"""
train_v2.py — 外层 5-fold 产 OOF + 全 80% refit final model + 测试集打分。

CLI:
  python train_v2.py --model lr  --sex female
  python train_v2.py --model xgb --sex female      # 需先跑 tune.py
  python train_v2.py --model dnn --sex male

输出 (outputs/v2/):
  {sex}_{model}_train_oof.parquet     训练区每人 OOF 预测 + acceleration
  {sex}_{model}_test_pred.parquet     测试区每人最终预测 + acceleration
  {sex}_{model}_metrics.json          OOF + test 双指标
  {sex}_{model}_final_model.joblib    final model + imputer + scaler bundle
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_modules import ALL_KEY, MODULE_KEYS, output_suffix, resolve_features
from v2_lib import (
    TARGET_COL,
    apply_acceleration, eval_one_fold, fit_acceleration_residual,
    imputed_feature_columns, load_config, load_imputed_features,
    make_en, make_kfold_iter, make_lr, make_xgb, metrics_dict,
    predict_dnn, train_dnn_fixed_epochs,
)


def load_best_params(out_dir: Path, sex_label: str, suffix: str,
                     model_name: str) -> dict:
    p = out_dir / f"{sex_label}{suffix}_{model_name}_best_params.json"
    if not p.exists():
        raise FileNotFoundError(f"先跑 tune.py 产 {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)["best_params"]


def run(config_path: Path, model_name: str, sex_label: str,
        features_arg: str, limit: int | None = None) -> None:
    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    out_dir = (base / cfg["paths"]["outputs_dir"]).resolve() / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_imp = load_imputed_features(data_dir, sex_label)
    df_train = df_imp[df_imp["split"] == "train"].reset_index(drop=True)
    df_test = df_imp[df_imp["split"] == "test"].reset_index(drop=True)
    if limit is not None:
        df_train = df_train.sample(n=min(limit, len(df_train)),
                                   random_state=0).reset_index(drop=True)
        df_test = df_test.sample(n=min(limit // 4, len(df_test)),
                                 random_state=0).reset_index(drop=True)
        print(f"[smoke] limit={limit} -> train={len(df_train)} test={len(df_test)}")
    full_feat_cols = imputed_feature_columns(df_imp)
    feat_cols = resolve_features(features_arg, full_feat_cols)
    suffix = output_suffix(features_arg)

    print(f"\n=== train_v2 | sex={sex_label} | features={features_arg} | "
          f"model={model_name} | n_train={len(df_train):,}  "
          f"n_test={len(df_test):,}  p={len(feat_cols)} ===")

    tcfg = cfg["tuning"]
    n_folds = tcfg["outer_folds"]
    seed = tcfg["seed"]
    xgb_cfg = tcfg["xgb"] if model_name == "xgb" else None
    dnn_cfg = tcfg["dnn"] if model_name == "dnn" else None
    en_cfg = cfg.get("elastic_net") if model_name == "en" else None

    # —— 取最佳超参 ——
    if model_name in ("lr", "en"):
        hp = {}
        if model_name == "lr":
            print("[train_v2] LR 不需要超参，使用 sklearn 默认")
        else:
            print(f"[train_v2] EN 用 ElasticNetCV 内置 CV 选参 "
                  f"(l1_ratio={en_cfg['l1_ratio_candidates']}, "
                  f"n_alphas={en_cfg['n_alphas']}, cv={en_cfg['cv']})")
    else:
        hp = load_best_params(out_dir, sex_label, suffix, model_name)
        print(f"[train_v2] best params from tune.py: {hp}")

    y_train = df_train[TARGET_COL].values.astype(np.float32)
    y_test = df_test[TARGET_COL].values.astype(np.float32)

    # —— Step 2: 外层 5-fold OOF ——
    oof = np.full(len(df_train), np.nan, dtype=np.float64)
    fold_metrics = []
    extras = []
    for fold_id, (tr, va) in enumerate(make_kfold_iter(y_train, n_splits=n_folds, seed=seed)):
        X_tr_df = df_train.iloc[tr]
        X_va_df = df_train.iloc[va]
        res = eval_one_fold(
            X_tr_df=X_tr_df, y_tr=y_train[tr],
            X_va_df=X_va_df, y_va=y_train[va],
            feat_cols=feat_cols,
            model_name=model_name,
            hp=hp,
            fold_seed=fold_id,
            xgb_cfg=xgb_cfg,
            dnn_cfg=dnn_cfg,
            en_cfg=en_cfg,
            val_idx=va,
        )
        oof[va] = res.val_pred
        m = dict(res.val_metrics)
        m["fold"] = fold_id
        fold_metrics.append(m)
        extras.append(res.extra)
        print(f"  fold {fold_id}: MAE={m['MAE']:.3f}  R2={m['R2']:.3f}  "
              f"r={m['pearson_r']:.3f}  extra={res.extra}")

    oof_metrics = metrics_dict(y_train, oof)
    print(f"  OOF: MAE={oof_metrics['MAE']:.3f}  R2={oof_metrics['R2']:.3f}  "
          f"r={oof_metrics['pearson_r']:.3f}")

    # —— Step 3: 全 80% refit final model ——
    print("\n[train_v2] refit final model on full train (80%) ...")
    X_full_arr = df_train[feat_cols].values.astype(np.float64)
    final_scaler = StandardScaler().fit(X_full_arr)
    X_full = final_scaler.transform(X_full_arr).astype(np.float32)

    if model_name == "lr":
        final_model = make_lr()
        final_model.fit(X_full, y_train)
    elif model_name == "en":
        final_model = make_en(
            l1_ratio_candidates=en_cfg["l1_ratio_candidates"],
            cv=en_cfg["cv"],
            n_alphas=en_cfg["n_alphas"],
            max_iter=en_cfg["max_iter"],
            random_state=en_cfg["random_state"],
        )
        final_model.fit(X_full, y_train)
        print(f"[train_v2] EN final: alpha={final_model.alpha_:.4g} "
              f"l1_ratio={final_model.l1_ratio_:.3g} "
              f"nnz={int(np.sum(np.abs(final_model.coef_) > 1e-12))}/{len(feat_cols)}")
    elif model_name == "xgb":
        # 用外层 fold best_iter 平均作为 final 的 n_estimators
        best_iters = [e.get("best_iter") for e in extras if e.get("best_iter")]
        n_est = int(round(np.mean(best_iters))) if best_iters else xgb_cfg["n_estimators_max"]
        print(f"[train_v2] xgb final n_estimators = mean(outer best_iter) = {n_est}")
        final_model = make_xgb(hp, n_estimators_max=n_est)
        final_model.fit(X_full, y_train)
    elif model_name == "dnn":
        best_eps = [e.get("best_epoch") for e in extras if e.get("best_epoch")]
        epochs = int(round(np.mean(best_eps))) if best_eps else dnn_cfg["epochs_max"]
        print(f"[train_v2] dnn final epochs = mean(outer best_epoch) = {epochs}")
        final_model = train_dnn_fixed_epochs(
            X_full, y_train, hp=hp, epochs=epochs,
        )
    else:
        raise ValueError(model_name)

    # —— Step 4: 测试集打分 ——
    print("\n[train_v2] predict test ...")
    X_test_arr = df_test[feat_cols].values.astype(np.float64)
    X_test = final_scaler.transform(X_test_arr).astype(np.float32)
    if model_name == "dnn":
        test_pred = predict_dnn(final_model, X_test).astype(np.float64)
    else:
        test_pred = final_model.predict(X_test).astype(np.float64)
    test_metrics = metrics_dict(y_test, test_pred)
    print(f"  TEST: MAE={test_metrics['MAE']:.3f}  R2={test_metrics['R2']:.3f}  "
          f"r={test_metrics['pearson_r']:.3f}")

    # —— acceleration 残差化 (LinearRegression(pred ~ age) 在 train 上 fit) ——
    accel_lr = fit_acceleration_residual(oof, y_train.astype(np.float64))
    train_accel = apply_acceleration(accel_lr, oof, y_train.astype(np.float64))
    test_accel = apply_acceleration(accel_lr, test_pred, y_test.astype(np.float64))

    # —— 输出 ——
    train_oof_df = pd.DataFrame({
        "eid": df_train["eid"].values,
        "sex": df_train["sex"].values,
        "age": y_train,
        "BioAge_pred": oof,
        "BioAge_acceleration": train_accel,
    })
    test_pred_df = pd.DataFrame({
        "eid": df_test["eid"].values,
        "sex": df_test["sex"].values,
        "age": y_test,
        "BioAge_pred": test_pred,
        "BioAge_acceleration": test_accel,
    })

    train_oof_path = out_dir / f"{sex_label}{suffix}_{model_name}_train_oof.parquet"
    test_pred_path = out_dir / f"{sex_label}{suffix}_{model_name}_test_pred.parquet"
    metrics_path = out_dir / f"{sex_label}{suffix}_{model_name}_metrics.json"
    bundle_path = out_dir / f"{sex_label}{suffix}_{model_name}_final_model.joblib"

    train_oof_df.to_parquet(train_oof_path, index=False)
    test_pred_df.to_parquet(test_pred_path, index=False)

    summary = {
        "model": model_name,
        "sex": sex_label,
        "features": features_arg,
        "feat_cols": feat_cols,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_features": len(feat_cols),
        "hp": hp,
        "oof": oof_metrics,
        "test": test_metrics,
        "folds": fold_metrics,
        "extras": extras,
        "notes": "v2.2: 80/20 split + global single miceforest + age excluded from imputer; "
                 f"feature set = {features_arg}",
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    bundle = {
        "model_name": model_name,
        "model": final_model if model_name != "dnn" else None,
        "model_state_dict": (final_model.state_dict() if model_name == "dnn" else None),
        "model_hp": hp,
        "scaler": final_scaler,
        "features": feat_cols,
        "accel_lr": accel_lr,
        "notes": "v2.1: imputer is global (see impute_v2.py); not bundled here",
    }
    joblib.dump(bundle, bundle_path)

    print(f"\n[train_v2] -> {train_oof_path}")
    print(f"[train_v2] -> {test_pred_path}")
    print(f"[train_v2] -> {metrics_path}")
    print(f"[train_v2] -> {bundle_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--model", choices=["lr", "en", "xgb", "dnn"], required=True)
    ap.add_argument("--sex", choices=["male", "female"], required=True)
    ap.add_argument("--features", choices=MODULE_KEYS, default=ALL_KEY,
                    help=f"特征集：{MODULE_KEYS}（默认 all = 综合 53 维）")
    ap.add_argument("--limit", type=int, default=None,
                    help="冒烟测试用：仅抽样 N 个训练样本")
    args = ap.parse_args()
    run(Path(args.config), args.model, args.sex, args.features, args.limit)
