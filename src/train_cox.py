"""
train_cox.py — v2.4：直接拟合死亡风险的 BioAge 训练流水线（EN-Cox + XGB-Cox）。

CLI:
  python train_cox.py --model encox  --sex female --features liver
  python train_cox.py --model xgbcox --sex male   --features all     # 需先跑 tune_cox.py

target: (death_event, death_time_years) from data/outcomes.parquet
        剔除 died_within_2yr=True 的样本

流程：
  1. 读 features_imputed_v2_{sex}.parquet ∩ outcomes (按 eid)
  2. resolve_features(features, full_cols) 取列子集
  3. 5-fold OOF（按 age 五分位分层）：每折独立 fit cox 模型 → predict val 的
     linear predictor → 拼 OOF
  4. 全 train refit final model
  5. apply 到 test → risk_score
  6. 风险分数 z-score（用 train OOF 的均值/方差），喂 gompertz.reverse_solve
     → bioage_cox
  7. BioAge_acceleration = bioage_cox 残差化 vs age（沿用 train_v2.py 口径，
     便于和 fit-age 模型在 survival_v2 / model_eval_v2 里横向比较）

输出（与 fit-age 输出同结构，方便扫描）：
  outputs/v2/{sex}{suffix}_{encox,xgbcox}_train_oof.parquet
      eid, sex, age, BioAge_pred(=bioage_cox), BioAge_acceleration,
      risk_score, bioage_cox
  outputs/v2/{sex}{suffix}_{encox,xgbcox}_test_pred.parquet  同上
  outputs/v2/{sex}{suffix}_{encox,xgbcox}_metrics.json
  outputs/v2/{sex}{suffix}_{encox,xgbcox}_final_model.joblib
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

try:
    import xgboost as xgb
except ImportError:
    xgb = None

import gompertz as gompertz_mod
from feature_modules import ALL_KEY, MODULE_KEYS, output_suffix, resolve_features
from v2_lib import (
    apply_acceleration, fit_acceleration_residual,
    imputed_feature_columns, load_config, load_imputed_features,
    log, make_kfold_iter, stratify_bins,
)


# --------------------------- 公共辅助 ---------------------------

def join_with_outcomes(df_train: pd.DataFrame, outcomes: pd.DataFrame,
                       drop_2yr: bool) -> pd.DataFrame:
    """把 features (含 age) 与 outcomes 在 eid 上 inner-join，剔除 died_within_2yr。
    返回的 DataFrame 保留 features 的所有列 + death_event/death_time_years/died_within_2yr。
    """
    out_cols = ["eid", "death_event", "death_time_years", "died_within_2yr"]
    merged = df_train.merge(outcomes[out_cols], on="eid", how="inner")
    n_before = len(merged)
    if drop_2yr:
        merged = merged[~merged["died_within_2yr"]].reset_index(drop=True)
    n_after = len(merged)
    log(f"  join+filter: {n_before:,} -> {n_after:,} "
        f"(drop_died_within_2yr={drop_2yr})")
    return merged


def to_xgb_label(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    """xgboost survival:cox 编码：label = time if event==1 else -time。"""
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=np.float64)
    # time 必须 > 0；遇到 0 时设个微小正数（极少见）
    t = np.where(t <= 0, 1e-3, t)
    return np.where(e == 1, t, -t)


# --------------------------- EN-Cox ---------------------------

def _normalize_l1_ratio(candidates):
    """sksurv CoxnetSurvivalAnalysis 接收 single float l1_ratio。我们选 candidates
    中 OOF C-index 最高的一个（外层会循环；这里给单值默认）。"""
    return list(candidates)


def fit_encox(X_tr_scaled: np.ndarray, surv_tr: np.ndarray,
              X_va_scaled: np.ndarray, surv_va: np.ndarray,
              *, l1_ratio: float, n_alphas: int, max_iter: int,
              tol: float) -> tuple[np.ndarray, CoxnetSurvivalAnalysis, dict]:
    """单 fold：给定 l1_ratio，由 CoxnetSurvivalAnalysis 内部走整条 alpha 路径，
    挑使 val C-index 最高的 alpha。返回 (val_risk, best_model, extras)。"""
    model = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        n_alphas=n_alphas,
        alpha_min_ratio=0.01,
        max_iter=max_iter,
        tol=tol,
        normalize=False,  # 我们外面已 StandardScaler 标准化
        fit_baseline_model=False,
    )
    model.fit(X_tr_scaled, surv_tr)
    alphas = model.alphas_
    best_c = -np.inf
    best_idx = -1
    e_va = surv_va["event"].astype(bool)
    t_va = surv_va["time"]
    for i, a in enumerate(alphas):
        try:
            pred = model.predict(X_va_scaled, alpha=a)
            c = concordance_index_censored(e_va, t_va, pred)[0]
        except Exception:
            continue
        if c > best_c:
            best_c = float(c)
            best_idx = i
    if best_idx < 0:
        raise RuntimeError("encox: 全部 alpha 计算 C-index 失败")
    best_alpha = float(alphas[best_idx])
    val_risk = model.predict(X_va_scaled, alpha=best_alpha)
    coef_at_best = model.coef_[:, best_idx]
    n_nz = int(np.sum(np.abs(coef_at_best) > 1e-12))
    extras = {
        "l1_ratio": float(l1_ratio),
        "best_alpha": best_alpha,
        "best_val_cindex": best_c,
        "n_nonzero": n_nz,
        "n_alphas_total": int(len(alphas)),
    }
    return val_risk.astype(np.float64), model, extras


def run_encox_fold(X_tr_df: pd.DataFrame, X_va_df: pd.DataFrame,
                   feat_cols: list[str], surv_tr: np.ndarray, surv_va: np.ndarray,
                   *, l1_ratio_candidates: list[float], n_alphas: int,
                   max_iter: int, tol: float) -> tuple[np.ndarray, dict, StandardScaler, CoxnetSurvivalAnalysis, float]:
    """一折 EN-Cox：在多个 l1_ratio 里挑 val C-index 最高那个。"""
    Xa_tr = X_tr_df[feat_cols].values.astype(np.float64)
    Xa_va = X_va_df[feat_cols].values.astype(np.float64)
    scaler = StandardScaler().fit(Xa_tr)
    X_tr_s = scaler.transform(Xa_tr).astype(np.float64)
    X_va_s = scaler.transform(Xa_va).astype(np.float64)

    best_c = -np.inf
    best_val = None
    best_extras = None
    best_model = None
    best_l1 = None
    for l1 in l1_ratio_candidates:
        try:
            val_pred, model, extras = fit_encox(
                X_tr_s, surv_tr, X_va_s, surv_va,
                l1_ratio=l1, n_alphas=n_alphas,
                max_iter=max_iter, tol=tol,
            )
        except Exception as e:
            log(f"    [encox l1_ratio={l1}] 失败: {e}")
            continue
        if extras["best_val_cindex"] > best_c:
            best_c = extras["best_val_cindex"]
            best_val = val_pred
            best_extras = extras
            best_model = model
            best_l1 = float(l1)
    if best_model is None:
        raise RuntimeError("encox: 全部 l1_ratio 失败")
    return best_val, best_extras, scaler, best_model, best_l1


# --------------------------- XGB-Cox ---------------------------

def _xgb_kwargs(hp: dict, n_estimators_max: int) -> dict:
    import os
    n_jobs = int(os.environ.get("THREADS_PER_JOB", "-1"))
    return {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "n_estimators": n_estimators_max,
        "max_depth": hp["max_depth"],
        "learning_rate": hp["learning_rate"],
        "subsample": hp["subsample"],
        "colsample_bytree": hp["colsample_bytree"],
        "min_child_weight": hp["min_child_weight"],
        "reg_alpha": hp["reg_alpha"],
        "reg_lambda": hp["reg_lambda"],
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": n_jobs,
    }


def run_xgbcox_fold(X_tr_df: pd.DataFrame, X_va_df: pd.DataFrame,
                    feat_cols: list[str], surv_tr: np.ndarray, surv_va: np.ndarray,
                    *, hp: dict, n_estimators_max: int,
                    early_stopping_rounds: int) -> tuple[np.ndarray, dict, StandardScaler, "xgb.XGBRegressor"]:
    if xgb is None:
        raise RuntimeError("xgboost 未安装")
    Xa_tr = X_tr_df[feat_cols].values.astype(np.float64)
    Xa_va = X_va_df[feat_cols].values.astype(np.float64)
    scaler = StandardScaler().fit(Xa_tr)
    X_tr_s = scaler.transform(Xa_tr).astype(np.float32)
    X_va_s = scaler.transform(Xa_va).astype(np.float32)

    y_tr = to_xgb_label(surv_tr["event"], surv_tr["time"])
    y_va = to_xgb_label(surv_va["event"], surv_va["time"])

    kw = _xgb_kwargs(hp, n_estimators_max)
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        kw["early_stopping_rounds"] = early_stopping_rounds
    model = xgb.XGBRegressor(**kw)
    model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
    # output_margin=True → raw linear predictor（log-hazard 空间）
    val_pred = model.predict(X_va_s, output_margin=True).astype(np.float64)
    e_va = surv_va["event"].astype(bool)
    t_va = surv_va["time"]
    c = float(concordance_index_censored(e_va, t_va, val_pred)[0])
    best_iter = getattr(model, "best_iteration", None) or model.n_estimators
    extras = {"best_iter": int(best_iter), "val_cindex": c}
    return val_pred, extras, scaler, model


# --------------------------- Main ---------------------------

def load_xgbcox_best_params(out_dir: Path, sex_label: str, suffix: str) -> dict:
    p = out_dir / f"{sex_label}{suffix}_xgbcox_best_params.json"
    if not p.exists():
        raise FileNotFoundError(f"先跑 tune_cox.py 产 {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)["best_params"]


def run(config_path: Path, model_name: str, sex_label: str,
        features_arg: str, limit: int | None = None) -> None:
    if model_name not in ("encox", "xgbcox"):
        raise ValueError(f"train_cox.py 只支持 encox / xgbcox，收到 {model_name}")
    t0_all = time.time()

    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    out_root = (base / cfg["paths"]["outputs_dir"]).resolve() / "v2"
    out_root.mkdir(parents=True, exist_ok=True)
    cox_dir = out_root / "cox"
    cox_dir.mkdir(parents=True, exist_ok=True)

    cox_cfg = cfg["cox"]
    drop_2yr = bool(cox_cfg.get("drop_died_within_2yr", True))
    n_folds = int(cox_cfg.get("outer_folds", 5))
    seed = int(cox_cfg.get("seed", 42))

    # —— 数据 ——
    df_imp = load_imputed_features(data_dir, sex_label)
    outcomes = pd.read_parquet(data_dir / "outcomes.parquet")
    df_train_full = df_imp[df_imp["split"] == "train"].reset_index(drop=True)
    df_test_full = df_imp[df_imp["split"] == "test"].reset_index(drop=True)
    if limit is not None:
        df_train_full = df_train_full.sample(n=min(limit, len(df_train_full)),
                                             random_state=0).reset_index(drop=True)
        df_test_full = df_test_full.sample(n=min(limit // 4, len(df_test_full)),
                                           random_state=0).reset_index(drop=True)
        log(f"[smoke] limit={limit} -> train={len(df_train_full)} test={len(df_test_full)}")
    full_feat_cols = imputed_feature_columns(df_imp)
    feat_cols = resolve_features(features_arg, full_feat_cols)
    suffix = output_suffix(features_arg)

    df_train = join_with_outcomes(df_train_full, outcomes, drop_2yr=drop_2yr)
    df_test = join_with_outcomes(df_test_full, outcomes, drop_2yr=drop_2yr)
    log(f"\n=== train_cox | sex={sex_label} features={features_arg} "
        f"model={model_name} | n_train={len(df_train):,} n_test={len(df_test):,} "
        f"p={len(feat_cols)} ===")

    age_train = df_train["Chronological_age"].values.astype(np.float64)
    age_test = df_test["Chronological_age"].values.astype(np.float64)
    event_train = df_train["death_event"].values.astype(np.int32)
    time_train = df_train["death_time_years"].values.astype(np.float64)
    event_test = df_test["death_event"].values.astype(np.int32)
    time_test = df_test["death_time_years"].values.astype(np.float64)
    log(f"  events: train={int(event_train.sum())}/{len(event_train)} "
        f"({100*event_train.mean():.2f}%)  "
        f"test={int(event_test.sum())}/{len(event_test)} "
        f"({100*event_test.mean():.2f}%)")

    # —— 5-fold OOF ——
    oof_risk = np.full(len(df_train), np.nan, dtype=np.float64)
    fold_metrics = []
    fold_extras = []
    encox_cfg = cox_cfg["encox"]
    xgbcox_cfg = cox_cfg["xgbcox"]

    hp_xgb = (load_xgbcox_best_params(out_root, sex_label, suffix)
              if model_name == "xgbcox" else None)
    if hp_xgb:
        log(f"  xgbcox best_params: {hp_xgb}")

    # 用 age 五分位分层
    for fold_id, (tr, va) in enumerate(make_kfold_iter(age_train.astype(np.float32),
                                                       n_splits=n_folds, seed=seed)):
        log(f"\n[fold {fold_id}] n_tr={len(tr):,} n_va={len(va):,}")
        X_tr_df = df_train.iloc[tr]
        X_va_df = df_train.iloc[va]
        surv_tr = Surv.from_arrays(event=event_train[tr].astype(bool),
                                   time=time_train[tr])
        surv_va = Surv.from_arrays(event=event_train[va].astype(bool),
                                   time=time_train[va])

        t0 = time.time()
        if model_name == "encox":
            val_risk, extras, _, _, _ = run_encox_fold(
                X_tr_df, X_va_df, feat_cols, surv_tr, surv_va,
                l1_ratio_candidates=encox_cfg["l1_ratio_candidates"],
                n_alphas=int(encox_cfg["n_alphas"]),
                max_iter=int(encox_cfg["max_iter"]),
                tol=float(encox_cfg["tol"]),
            )
            log(f"  encox fold done in {time.time()-t0:.1f}s | "
                f"l1={extras['l1_ratio']:.2g} alpha={extras['best_alpha']:.4g} "
                f"C={extras['best_val_cindex']:.4f} nnz={extras['n_nonzero']}")
        else:  # xgbcox
            val_risk, extras, _, _ = run_xgbcox_fold(
                X_tr_df, X_va_df, feat_cols, surv_tr, surv_va,
                hp=hp_xgb,
                n_estimators_max=int(xgbcox_cfg["n_estimators_max"]),
                early_stopping_rounds=int(xgbcox_cfg["early_stopping_rounds"]),
            )
            log(f"  xgbcox fold done in {time.time()-t0:.1f}s | "
                f"best_iter={extras['best_iter']} C={extras['val_cindex']:.4f}")
        oof_risk[va] = val_risk
        fold_metrics.append({"fold": fold_id,
                             "val_cindex": extras.get("best_val_cindex",
                                                      extras.get("val_cindex"))})
        fold_extras.append(extras)

    e_tr = event_train.astype(bool)
    oof_cindex = float(concordance_index_censored(e_tr, time_train, oof_risk)[0])
    log(f"\n[oof] C-index = {oof_cindex:.4f}  "
        f"(n={len(oof_risk):,} events={int(e_tr.sum()):,})")

    # —— Final refit on full train ——
    log("\n[refit] full train final model ...")
    X_full = df_train[feat_cols].values.astype(np.float64)
    final_scaler = StandardScaler().fit(X_full)
    X_full_s = final_scaler.transform(X_full)
    surv_full = Surv.from_arrays(event=e_tr, time=time_train)
    final_extras: dict = {}

    if model_name == "encox":
        # 用 OOF fold 投票出最频繁的 l1_ratio
        l1_votes = [e["l1_ratio"] for e in fold_extras]
        chosen_l1 = float(max(set(l1_votes), key=l1_votes.count))
        # 在 chosen_l1 内取 fold 选出的 alpha 几何均值（log 域更稳）
        same_l1_alphas = [e["best_alpha"] for e in fold_extras
                          if abs(e["l1_ratio"] - chosen_l1) < 1e-9]
        if not same_l1_alphas:
            same_l1_alphas = [e["best_alpha"] for e in fold_extras]
        chosen_alpha = float(np.exp(np.mean(np.log(same_l1_alphas))))
        log(f"  encox final l1_ratio (fold majority) = {chosen_l1}  "
            f"alpha (geo-mean of {len(same_l1_alphas)} fold-best) = {chosen_alpha:.4g}")
        # 用和 fold 完全相同的稳定配置：sksurv 自带 alpha 路径（alpha_max→alpha_max*ratio，
        # 逐档 warm-start，数值稳定）。再在路径上 snap 到最接近 chosen_alpha 的那一档。
        # 不要传自定义 alphas：单点/粗 geomspace 路径会让坐标下降退化成全零解。
        final_model = CoxnetSurvivalAnalysis(
            l1_ratio=chosen_l1,
            n_alphas=int(encox_cfg["n_alphas"]),
            alpha_min_ratio=0.01,
            max_iter=int(encox_cfg["max_iter"]),
            tol=float(encox_cfg["tol"]),
            normalize=False,
            fit_baseline_model=False,
        )
        final_model = final_model.fit(X_full_s.astype(np.float64), surv_full)
        path = final_model.alphas_
        idx_chosen = int(np.argmin(np.abs(path - chosen_alpha)))
        # 防退化：snap 命中的列若系数全零，沿路径挑离 chosen_alpha 最近的非零解
        if np.all(np.abs(final_model.coef_[:, idx_chosen]) <= 1e-12):
            nz_counts = np.sum(np.abs(final_model.coef_) > 1e-12, axis=0)
            nz_idx = np.where(nz_counts > 0)[0]
            if nz_idx.size:
                idx_chosen = int(nz_idx[np.argmin(np.abs(path[nz_idx] - chosen_alpha))])
        chosen_alpha_eff = float(path[idx_chosen])
        coef_final = final_model.coef_[:, idx_chosen]
        n_nz_final = int(np.sum(np.abs(coef_final) > 1e-12))
        train_risk = final_model.predict(X_full_s.astype(np.float64),
                                         alpha=chosen_alpha_eff)
        train_c = float(concordance_index_censored(e_tr, time_train, train_risk)[0])
        final_extras["final_l1_ratio"] = chosen_l1
        final_extras["final_alpha"] = chosen_alpha_eff
        final_extras["final_alpha_target"] = chosen_alpha
        final_extras["final_train_cindex"] = train_c
        final_extras["final_n_nonzero"] = n_nz_final
        log(f"  encox final: alpha={chosen_alpha_eff:.4g} (target {chosen_alpha:.4g})  "
            f"train C={train_c:.4f}  nnz={n_nz_final}/{len(feat_cols)}")
        X_te_s = final_scaler.transform(
            df_test[feat_cols].values.astype(np.float64)
        )
        test_risk = final_model.predict(X_te_s.astype(np.float64),
                                        alpha=chosen_alpha_eff)
    else:  # xgbcox
        # final n_estimators = mean(outer best_iter)
        bis = [e["best_iter"] for e in fold_extras if "best_iter" in e]
        n_est = int(round(np.mean(bis))) if bis else int(xgbcox_cfg["n_estimators_max"])
        log(f"  xgbcox final n_estimators = mean(outer best_iter) = {n_est}")
        kw = _xgb_kwargs(hp_xgb, n_est)
        final_model = xgb.XGBRegressor(**kw)
        y_full = to_xgb_label(event_train, time_train)
        final_model.fit(X_full_s.astype(np.float32), y_full, verbose=False)
        train_risk = final_model.predict(
            X_full_s.astype(np.float32), output_margin=True
        ).astype(np.float64)
        X_te_s = final_scaler.transform(
            df_test[feat_cols].values.astype(np.float64)
        ).astype(np.float32)
        test_risk = final_model.predict(
            X_te_s, output_margin=True
        ).astype(np.float64)
        final_extras["final_n_estimators"] = n_est
        train_c = float(concordance_index_censored(e_tr, time_train, train_risk)[0])
        final_extras["final_train_cindex"] = train_c
        log(f"  xgbcox final train C={train_c:.4f}")

    # test C-index（用 final 模型）
    e_te = event_test.astype(bool)
    test_cindex = float(concordance_index_censored(e_te, time_test, test_risk)[0])
    log(f"[test] C-index (final model on holdout) = {test_cindex:.4f}  "
        f"(n={len(test_risk):,} events={int(e_te.sum()):,})")

    # —— 标准化 risk_score ——
    # 用 OOF 的均值/方差 z-score → 喂 Gompertz；test 用同一组 (mu, sd)
    risk_mu = float(np.mean(oof_risk))
    risk_sd = float(np.std(oof_risk, ddof=0))
    if risk_sd <= 1e-12:
        risk_sd = 1.0
        log("  [warn] OOF risk sd≈0，回退到 1.0")
    train_eta = (oof_risk - risk_mu) / risk_sd
    test_eta = (test_risk - risk_mu) / risk_sd

    # —— Gompertz 参数：男女一次，缺则现拟 ——
    gp_path = cox_dir / f"gompertz_params_{sex_label}.json"
    if not gp_path.exists():
        log(f"[gompertz] 拟合 {sex_label} ...")
        # 用全 imputed train（含 died_within_2yr 也保留 — 拟合人群尽量大）
        gp_params = gompertz_mod.fit_and_save_from_train(
            df_train=df_imp[df_imp["split"] == "train"][["eid"]].assign(
                age=df_imp[df_imp["split"] == "train"]["Chronological_age"].values
            ),
            outcomes=outcomes,
            sex_label=sex_label,
            out_dir=cox_dir,
            bin_width=float(cox_cfg["gompertz"]["age_bin_width"]),
            age_min=float(cox_cfg["gompertz"]["fit_age_min"]),
            age_max=float(cox_cfg["gompertz"]["fit_age_max"]),
        )
    else:
        gp_params = gompertz_mod.load_params(sex_label, cox_dir)
        log(f"[gompertz] 复用 {gp_path.name} (a={gp_params['a']:.4g} "
            f"b={gp_params['b']:.4f} fit_r2={gp_params['fit_r2']:.4f})")

    ref_age = float(cox_cfg["gompertz"]["ref_age"])
    bioage_oof = gompertz_mod.reverse_solve(
        train_eta, a=gp_params["a"], b=gp_params["b"],
        ref_age=ref_age, horizon=10.0,
    )
    bioage_test = gompertz_mod.reverse_solve(
        test_eta, a=gp_params["a"], b=gp_params["b"],
        ref_age=ref_age, horizon=10.0,
    )
    log(f"[gompertz] reverse solve done | "
        f"OOF bioage mean={bioage_oof.mean():.2f} (vs age {age_train.mean():.2f}) | "
        f"TEST bioage mean={bioage_test.mean():.2f} (vs age {age_test.mean():.2f})")

    # —— acceleration: bioage_cox 残差化 vs age ——
    accel_lr = fit_acceleration_residual(bioage_oof, age_train)
    train_accel = apply_acceleration(accel_lr, bioage_oof, age_train)
    test_accel = apply_acceleration(accel_lr, bioage_test, age_test)

    # —— 落盘 ——
    train_oof_df = pd.DataFrame({
        "eid": df_train["eid"].values,
        "sex": df_train["sex"].values,
        "age": age_train,
        "BioAge_pred": bioage_oof,
        "BioAge_acceleration": train_accel,
        "risk_score": train_eta,
        "bioage_cox": bioage_oof,
    })
    test_pred_df = pd.DataFrame({
        "eid": df_test["eid"].values,
        "sex": df_test["sex"].values,
        "age": age_test,
        "BioAge_pred": bioage_test,
        "BioAge_acceleration": test_accel,
        "risk_score": test_eta,
        "bioage_cox": bioage_test,
    })

    train_oof_path = out_root / f"{sex_label}{suffix}_{model_name}_train_oof.parquet"
    test_pred_path = out_root / f"{sex_label}{suffix}_{model_name}_test_pred.parquet"
    metrics_path = out_root / f"{sex_label}{suffix}_{model_name}_metrics.json"
    bundle_path = out_root / f"{sex_label}{suffix}_{model_name}_final_model.joblib"

    train_oof_df.to_parquet(train_oof_path, index=False)
    test_pred_df.to_parquet(test_pred_path, index=False)

    summary = {
        "model": model_name,
        "target_type": "death",
        "sex": sex_label,
        "features": features_arg,
        "feat_cols": feat_cols,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_features": len(feat_cols),
        "n_events_train": int(event_train.sum()),
        "n_events_test": int(event_test.sum()),
        "hp": hp_xgb if model_name == "xgbcox" else {},
        "oof": {"cindex": oof_cindex},
        "test": {"cindex": test_cindex},
        "folds": fold_metrics,
        "extras": fold_extras,
        "final_extras": final_extras,
        "risk_score_normalization": {"mu": risk_mu, "sd": risk_sd},
        "gompertz": {"a": gp_params["a"], "b": gp_params["b"],
                     "ref_age": ref_age, "fit_r2": gp_params["fit_r2"]},
        "notes": f"v2.4: fit-to-death (drop_died_within_2yr={drop_2yr}), "
                 f"target=(death_event, death_time_years), feature set = {features_arg}",
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    bundle = {
        "model_name": model_name,
        "target_type": "death",
        "model": final_model,
        "model_hp": hp_xgb if model_name == "xgbcox" else None,
        "scaler": final_scaler,
        "features": feat_cols,
        "accel_lr": accel_lr,
        "risk_score_normalization": {"mu": risk_mu, "sd": risk_sd},
        "gompertz_params": gp_params,
        "notes": "v2.4 cox bundle",
    }
    joblib.dump(bundle, bundle_path)

    log(f"\n[train_cox] -> {train_oof_path.name}")
    log(f"[train_cox] -> {test_pred_path.name}")
    log(f"[train_cox] -> {metrics_path.name}")
    log(f"[train_cox] -> {bundle_path.name}")
    log(f"[train_cox] 总耗时 {time.time()-t0_all:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--model", choices=["encox", "xgbcox"], required=True)
    ap.add_argument("--sex", choices=["male", "female"], required=True)
    ap.add_argument("--features", choices=MODULE_KEYS, default=ALL_KEY,
                    help=f"特征集：{MODULE_KEYS}")
    ap.add_argument("--limit", type=int, default=None,
                    help="冒烟测试用：仅抽样 N 个训练样本")
    args = ap.parse_args()
    run(Path(args.config), args.model, args.sex, args.features, args.limit)
