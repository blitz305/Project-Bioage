"""
tune_cox.py — v2.4：XGB-Cox 调参（Optuna，C-index 内层 CV）。

CLI:
  python tune_cox.py --model xgbcox --sex female --features liver --n-trials 15

EN-Cox 不需要调参（CoxnetSurvivalAnalysis 内置 alpha + 我们外层挑 l1_ratio）。

每个 trial：
  读 features_imputed_v2_{sex}.parquet ∩ outcomes
  剔除 died_within_2yr=True
  按 age 五分位分 inner_folds 折
  每折 fit XGB-Cox（objective=survival:cox）
  评分 = mean(val C-index)

输出：
  outputs/v2/{sex}{suffix}_xgbcox_best_params.json
  outputs/v2/{sex}{suffix}_xgbcox_optuna_history.csv
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from feature_modules import ALL_KEY, MODULE_KEYS, output_suffix, resolve_features
from train_cox import _xgb_kwargs, join_with_outcomes, to_xgb_label
from v2_lib import (
    imputed_feature_columns, load_config, load_imputed_features,
    log, make_kfold_iter,
)


def sample_hp(trial: optuna.Trial, space: dict) -> dict:
    hp = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "int":
            hp[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "float":
            hp[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False),
            )
        elif t == "cat":
            hp[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"unknown space type: {t}")
    return hp


def make_objective(df_train: pd.DataFrame, feat_cols: list[str],
                   age_train: np.ndarray, event_train: np.ndarray,
                   time_train: np.ndarray, *, space: dict,
                   n_folds: int, seed: int, n_estimators_max: int,
                   early_stopping_rounds: int):
    def objective(trial: optuna.Trial) -> float:
        if xgb is None:
            raise RuntimeError("xgboost 未安装")
        hp = sample_hp(trial, space)
        fold_c = []
        kfold = list(make_kfold_iter(age_train.astype(np.float32),
                                     n_splits=n_folds, seed=seed))
        for fold_id, (tr, va) in enumerate(kfold):
            Xa_tr = df_train.iloc[tr][feat_cols].values.astype(np.float64)
            Xa_va = df_train.iloc[va][feat_cols].values.astype(np.float64)
            scaler = StandardScaler().fit(Xa_tr)
            X_tr = scaler.transform(Xa_tr).astype(np.float32)
            X_va = scaler.transform(Xa_va).astype(np.float32)

            y_tr = to_xgb_label(event_train[tr], time_train[tr])
            y_va = to_xgb_label(event_train[va], time_train[va])

            kw = _xgb_kwargs(hp, n_estimators_max)
            if early_stopping_rounds and early_stopping_rounds > 0:
                kw["early_stopping_rounds"] = early_stopping_rounds
            m = xgb.XGBRegressor(**kw)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            pred = m.predict(X_va, output_margin=True).astype(np.float64)
            c = float(concordance_index_censored(
                event_train[va].astype(bool), time_train[va], pred
            )[0])
            fold_c.append(c)
            trial.report(float(np.mean(fold_c)), step=fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_c))
    return objective


def run(config_path: Path, model_name: str, sex_label: str,
        features_arg: str, n_trials: int | None,
        limit: int | None = None) -> None:
    if model_name != "xgbcox":
        raise ValueError(f"tune_cox.py 仅支持 xgbcox，收到 {model_name}")

    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    out_dir = (base / cfg["paths"]["outputs_dir"]).resolve() / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    cox_cfg = cfg["cox"]
    xgbcox_cfg = cox_cfg["xgbcox"]
    drop_2yr = bool(cox_cfg.get("drop_died_within_2yr", True))
    n_folds = int(xgbcox_cfg.get("inner_folds", 5))
    seed = int(cox_cfg.get("seed", 42))
    n_trials = int(n_trials or xgbcox_cfg.get("n_trials", 15))

    df_imp = load_imputed_features(data_dir, sex_label)
    outcomes = pd.read_parquet(data_dir / "outcomes.parquet")
    df_tr_raw = df_imp[df_imp["split"] == "train"].reset_index(drop=True)
    if limit is not None:
        df_tr_raw = df_tr_raw.sample(n=min(limit, len(df_tr_raw)),
                                     random_state=0).reset_index(drop=True)
        log(f"[smoke] limit={limit} -> train={len(df_tr_raw)}")
    full_feat_cols = imputed_feature_columns(df_imp)
    feat_cols = resolve_features(features_arg, full_feat_cols)
    suffix = output_suffix(features_arg)

    df_train = join_with_outcomes(df_tr_raw, outcomes, drop_2yr=drop_2yr)
    log(f"\n=== tune_cox | sex={sex_label} features={features_arg} model={model_name} "
        f"| n_train={len(df_train):,} p={len(feat_cols)} trials={n_trials} ===")

    age_train = df_train["Chronological_age"].values.astype(np.float64)
    event_train = df_train["death_event"].values.astype(np.int32)
    time_train = df_train["death_time_years"].values.astype(np.float64)
    log(f"  events: {int(event_train.sum())}/{len(event_train)} "
        f"({100*event_train.mean():.2f}%)")

    pruner = MedianPruner(n_warmup_steps=2)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{sex_label}{suffix}_xgbcox",
    )
    objective = make_objective(
        df_train=df_train, feat_cols=feat_cols,
        age_train=age_train, event_train=event_train, time_train=time_train,
        space=xgbcox_cfg["space"],
        n_folds=n_folds, seed=seed,
        n_estimators_max=int(xgbcox_cfg["n_estimators_max"]),
        early_stopping_rounds=int(xgbcox_cfg["early_stopping_rounds"]),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log(f"[tune_cox] {n_trials} trials in {time.time()-t0:.1f}s")

    best = study.best_trial
    log(f"[tune_cox] best #{best.number}: C-index={best.value:.4f}  params={best.params}")

    best_path = out_dir / f"{sex_label}{suffix}_xgbcox_best_params.json"
    hist_path = out_dir / f"{sex_label}{suffix}_xgbcox_optuna_history.csv"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "xgbcox",
            "target_type": "death",
            "sex": sex_label,
            "features": features_arg,
            "n_features": len(feat_cols),
            "n_trials": n_trials,
            "inner_folds": n_folds,
            "best_value_C": float(best.value),
            "best_params": best.params,
        }, f, ensure_ascii=False, indent=2)
    df_hist = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    df_hist.to_csv(hist_path, index=False)
    log(f"[tune_cox] -> {best_path.name}")
    log(f"[tune_cox] -> {hist_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--model", choices=["xgbcox"], default="xgbcox")
    ap.add_argument("--sex", choices=["male", "female"], required=True)
    ap.add_argument("--features", choices=MODULE_KEYS, default=ALL_KEY)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(Path(args.config), args.model, args.sex, args.features,
        args.n_trials, args.limit)
