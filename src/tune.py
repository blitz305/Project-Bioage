"""
tune.py — Optuna 调参（仅 xgb / dnn；lr 直接用默认）。

CLI:
  python tune.py --model xgb --sex female --n-trials 30
  python tune.py --model dnn --sex male  --n-trials 30

每个 trial：
  读已全局插补好的 features_imputed_v2_{sex}.parquet（仅 train 区）→
  切 5 折 → 每折 fit scaler + 模型 → 验证 R² →
  trial 分数 = mean(5 个 R²)

输出:
  outputs/v2/{sex}_{model}_best_params.json     最佳超参
  outputs/v2/{sex}_{model}_optuna_history.csv   全 trials 记录
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from feature_modules import ALL_KEY, MODULE_KEYS, output_suffix, resolve_features
from v2_lib import (
    TARGET_COL,
    eval_one_fold, imputed_feature_columns,
    load_config, load_imputed_features, make_kfold_iter,
)


def sample_hp(trial: optuna.Trial, space: dict) -> dict:
    hp = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "int":
            hp[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "float":
            hp[name] = trial.suggest_float(
                name, spec["low"], spec["high"],
                log=spec.get("log", False),
            )
        elif t == "cat":
            hp[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"unknown space type: {t}")
    return hp


def make_objective(*, df_train, feat_cols, target_col, model_name,
                   space, xgb_cfg, dnn_cfg, n_folds, seed):
    y = df_train[target_col].values.astype(np.float32)

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hp(trial, space)
        fold_r2 = []
        kfold = list(make_kfold_iter(y, n_splits=n_folds, seed=seed))
        for fold_id, (tr, va) in enumerate(kfold):
            X_tr_df = df_train.iloc[tr]
            X_va_df = df_train.iloc[va]
            res = eval_one_fold(
                X_tr_df=X_tr_df, y_tr=y[tr],
                X_va_df=X_va_df, y_va=y[va],
                feat_cols=feat_cols,
                model_name=model_name,
                hp=hp,
                fold_seed=fold_id,
                xgb_cfg=xgb_cfg,
                dnn_cfg=dnn_cfg,
                val_idx=va,
            )
            fold_r2.append(res.val_metrics["R2"])
            trial.report(float(np.mean(fold_r2)), step=fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_r2))

    return objective


def run(config_path: Path, model_name: str, sex_label: str,
        features_arg: str,
        n_trials: int | None, limit: int | None = None) -> None:
    if model_name not in ("xgb", "dnn"):
        raise ValueError(f"tune.py 仅支持 xgb / dnn，收到 {model_name}")
    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    out_dir = (base / cfg["paths"]["outputs_dir"]).resolve() / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_imp = load_imputed_features(data_dir, sex_label)
    df_train = df_imp[df_imp["split"] == "train"].reset_index(drop=True)
    if limit is not None:
        df_train = df_train.sample(n=min(limit, len(df_train)),
                                   random_state=0).reset_index(drop=True)
        print(f"[smoke] limit={limit} -> train={len(df_train)}")
    full_feat_cols = imputed_feature_columns(df_imp)
    feat_cols = resolve_features(features_arg, full_feat_cols)
    suffix = output_suffix(features_arg)

    print(f"\n=== tune | sex={sex_label} | features={features_arg} | "
          f"model={model_name} | n_train={len(df_train):,} | p={len(feat_cols)} ===")

    tcfg = cfg["tuning"]
    n_trials = n_trials or tcfg["n_trials"]
    n_folds = tcfg["inner_folds"]
    seed = tcfg["seed"]
    xgb_cfg = tcfg["xgb"] if model_name == "xgb" else None
    dnn_cfg = tcfg["dnn"] if model_name == "dnn" else None
    space = tcfg[model_name]["space"]

    pruner = MedianPruner(n_warmup_steps=2) if tcfg.get("pruner") == "median" else None
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{sex_label}_{model_name}",
    )

    objective = make_objective(
        df_train=df_train, feat_cols=feat_cols, target_col=TARGET_COL,
        model_name=model_name, space=space,
        xgb_cfg=xgb_cfg, dnn_cfg=dnn_cfg,
        n_folds=n_folds, seed=seed,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    print(f"\n[tune] best trial #{best_trial.number}: R²={best_trial.value:.4f}")
    print(f"[tune] best params: {best_trial.params}")

    best_path = out_dir / f"{sex_label}{suffix}_{model_name}_best_params.json"
    hist_path = out_dir / f"{sex_label}{suffix}_{model_name}_optuna_history.csv"

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "sex": sex_label,
            "features": features_arg,
            "n_features": len(feat_cols),
            "n_trials": n_trials,
            "inner_folds": n_folds,
            "best_value_R2": float(best_trial.value),
            "best_params": best_trial.params,
        }, f, ensure_ascii=False, indent=2)
    print(f"[tune] -> {best_path}")

    df_hist = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    df_hist.to_csv(hist_path, index=False)
    print(f"[tune] -> {hist_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--model", choices=["xgb", "dnn"], required=True)
    ap.add_argument("--sex", choices=["male", "female"], required=True)
    ap.add_argument("--features", choices=MODULE_KEYS, default=ALL_KEY,
                    help=f"特征集：{MODULE_KEYS}（默认 all = 综合 53 维）")
    ap.add_argument("--n-trials", type=int, default=None,
                    help="覆盖 config.tuning.n_trials")
    ap.add_argument("--limit", type=int, default=None,
                    help="冒烟测试用：仅抽样 N 个训练样本")
    args = ap.parse_args()
    run(Path(args.config), args.model, args.sex, args.features,
        args.n_trials, args.limit)
