"""
v2_lib.py — v2 流程共享工具：
  • split / 数据切片
  • per-fold miceforest 包装
  • 模型工厂 (lr / xgb / dnn)
  • per-fold 评分（给 tune.py 和 train_v2.py 共用）

设计原则（v2.1 改动：放弃 per-fold MICE，改全局一次性插补）
--------
1. age 永远不进 imputer 预测器集合（avoid target leakage）。
2. miceforest 在 train 区 fit 一次（见 impute_v2.py），同时 transform train + test。
3. tune.py / train_v2.py 直接读 features_imputed_v2_{sex}.parquet，外层折不再做 MICE。
4. eval_one_fold 现在期望 X_*_df 已经是无 NaN 的（来自 imputed parquet）。
"""
from __future__ import annotations
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import miceforest as mf

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None


SEX_FEMALE = 0
SEX_MALE = 1
TARGET_COL = "Chronological_age"


# ---------- IO ----------

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(data_dir: Path) -> pd.DataFrame:
    p = data_dir / "split.parquet"
    if not p.exists():
        raise FileNotFoundError(f"先跑 make_split.py，缺 {p}")
    return pd.read_parquet(p)


def load_raw_features(data_dir: Path) -> pd.DataFrame:
    p = data_dir / "features_raw.parquet"
    if not p.exists():
        raise FileNotFoundError(f"先跑 preprocess.py，缺 {p}")
    return pd.read_parquet(p)


def load_imputed_features(data_dir: Path, sex_label: str) -> pd.DataFrame:
    """读 impute_v2.py 写出的全局一次性插补结果。"""
    p = data_dir / f"features_imputed_v2_{sex_label}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"先跑 impute_v2.py 生成 {p}")
    return pd.read_parquet(p)


def imputed_feature_columns(df: pd.DataFrame) -> list[str]:
    """从 imputed parquet 中取特征列名（排除元数据 + age）。"""
    return [c for c in df.columns if c not in ("eid", "sex", "split", TARGET_COL)]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """53 个 ML 输入特征名（排除 eid / sex / target）。MICE 也用这一套。"""
    return [c for c in df.columns if c not in ("eid", "sex", TARGET_COL)]


def split_by_sex_and_set(df: pd.DataFrame, split_df: pd.DataFrame,
                         sex_val: int, set_name: str) -> pd.DataFrame:
    """取指定性别 × 指定集合（train/test）的子集。"""
    sub_split = split_df[(split_df["sex"] == sex_val) & (split_df["split"] == set_name)]
    return df.merge(sub_split[["eid"]], on="eid", how="inner").reset_index(drop=True)


# ---------- 评估辅助 ----------

def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(stats.pearsonr(y_true, y_pred)[0]),
    }


def fit_acceleration_residual(pred: np.ndarray, age: np.ndarray) -> LinearRegression:
    """在训练区拟合 LinearRegression(pred ~ age)，返回 lr 对象供 test 复用。"""
    lr = LinearRegression().fit(age.reshape(-1, 1), pred)
    return lr


def apply_acceleration(lr: LinearRegression, pred: np.ndarray, age: np.ndarray) -> np.ndarray:
    return pred - lr.predict(age.reshape(-1, 1))


def stratify_bins(age: np.ndarray, n_bins: int = 5) -> np.ndarray:
    edges = np.quantile(age, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6
    return np.digitize(age, edges[1:-1])


# ---------- miceforest 包装 ----------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    """带时间戳 + 立即 flush 的 print。"""
    print(f"[{_ts()}] {msg}", flush=True)


@dataclass
class FittedImputer:
    """封装 miceforest kernel + 列顺序 + 训练列均值 fallback。"""
    kernel: mf.ImputationKernel
    feat_cols: list[str]
    col_means: np.ndarray  # shape (p,)，按 feat_cols 顺序

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        new_df = df[self.feat_cols].reset_index(drop=True).copy()
        completed = self.kernel.impute_new_data(new_df).complete_data(0)
        arr = completed[self.feat_cols].values.astype(np.float64)
        # miceforest 对训练时无缺失的列不做插补；用训练列均值兜底
        if np.isnan(arr).any():
            inds = np.where(np.isnan(arr))
            arr[inds] = np.take(self.col_means, inds[1])
        return arr


def fit_imputer(X_train_df: pd.DataFrame, *, feat_cols: list[str],
                iterations: int, random_state: int) -> FittedImputer:
    """在训练子集上 fit miceforest（单数据集）。X 必须是 DataFrame，含 NaN。"""
    sub = X_train_df[feat_cols].reset_index(drop=True).copy()
    log(f"  [imputer] fit start: rows={len(sub):,} cols={len(feat_cols)} iter={iterations}")
    t0 = time.time()
    kernel = mf.ImputationKernel(
        data=sub,
        num_datasets=1,
        random_state=random_state,
    )
    kernel.mice(iterations=iterations, verbose=True)
    log(f"  [imputer] fit done in {time.time() - t0:.1f}s")
    completed_train = kernel.complete_data(0)[feat_cols].values.astype(np.float64)
    col_means = np.nanmean(completed_train, axis=0)
    return FittedImputer(kernel=kernel, feat_cols=list(feat_cols),
                         col_means=col_means)


# ---------- 模型工厂 ----------

def _xgb_kwargs_from_hp(hp: dict, *, n_estimators_max: int) -> dict:
    import os
    n_jobs = int(os.environ.get("THREADS_PER_JOB", "-1"))
    return {
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


def make_lr() -> LinearRegression:
    return LinearRegression()


def make_en(*, l1_ratio_candidates: list[float], cv: int = 5,
            n_alphas: int = 100, max_iter: int = 5000,
            random_state: int = 42) -> ElasticNetCV:
    """ElasticNetCV：内置 CV 自动选 alpha + l1_ratio，不需要 Optuna。"""
    import os
    n_jobs = int(os.environ.get("THREADS_PER_JOB", "-1"))
    # sklearn 1.7+: 用 alphas=int 替代 n_alphas=int
    return ElasticNetCV(
        l1_ratio=list(l1_ratio_candidates),
        alphas=n_alphas,
        cv=cv,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def make_xgb(hp: dict, *, n_estimators_max: int = 2000,
             early_stopping_rounds: int | None = None):
    if xgb is None:
        raise RuntimeError("xgboost 未安装")
    kwargs = _xgb_kwargs_from_hp(hp, n_estimators_max=n_estimators_max)
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    return xgb.XGBRegressor(**kwargs)


# ---------- DNN ----------

if torch is not None:
    class MLP(nn.Module):
        def __init__(self, in_dim: int, n_layers: int, hidden_dim: int, dropout: float):
            super().__init__()
            layers = []
            prev = in_dim
            for _ in range(n_layers):
                layers += [nn.Linear(prev, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
                prev = hidden_dim
            layers += [nn.Linear(prev, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


def train_dnn(X_tr: np.ndarray, y_tr: np.ndarray,
              X_va: np.ndarray, y_va: np.ndarray,
              *, hp: dict, epochs_max: int, patience: int,
              device: str = "cpu") -> tuple[np.ndarray, int]:
    """训 DNN 到 X_va 上 early stopping。返回 (val_pred_at_best, best_epoch)。"""
    if torch is None:
        raise RuntimeError("pytorch 未安装")
    torch.manual_seed(42)

    model = MLP(
        in_dim=X_tr.shape[1],
        n_layers=hp["n_layers"],
        hidden_dim=hp["hidden_dim"],
        dropout=hp["dropout"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(),
                           lr=hp["learning_rate"],
                           weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True)

    Xv = torch.tensor(X_va, dtype=torch.float32, device=device)
    yv_np = y_va.astype(np.float32)

    best_mae = float("inf")
    best_pred = None
    best_epoch = 0
    bad = 0
    for ep in range(1, epochs_max + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xv).cpu().numpy()
        mae = float(np.mean(np.abs(pred - yv_np)))
        if mae < best_mae - 1e-5:
            best_mae = mae
            best_pred = pred
            best_epoch = ep
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_pred, best_epoch


def train_dnn_fixed_epochs(X_tr: np.ndarray, y_tr: np.ndarray,
                           *, hp: dict, epochs: int,
                           device: str = "cpu") -> "MLP":
    """用固定 epochs 训 DNN（用于 final refit；epochs 来自 outer fold 平均 best_epoch）。"""
    if torch is None:
        raise RuntimeError("pytorch 未安装")
    torch.manual_seed(42)
    model = MLP(
        in_dim=X_tr.shape[1],
        n_layers=hp["n_layers"],
        hidden_dim=hp["hidden_dim"],
        dropout=hp["dropout"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(),
                           lr=hp["learning_rate"],
                           weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()
    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def predict_dnn(model, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        return model(Xt).cpu().numpy()


# ---------- 一折评分 (tune.py + train_v2.py 共用) ----------

@dataclass
class FoldResult:
    val_pred: np.ndarray
    val_idx: np.ndarray
    val_metrics: dict
    extra: dict   # best_iter (xgb) / best_epoch (dnn)


def eval_one_fold(
    X_tr_df: pd.DataFrame, y_tr: np.ndarray,
    X_va_df: pd.DataFrame, y_va: np.ndarray,
    *,
    feat_cols: list[str],
    model_name: str,
    hp: dict,
    fold_seed: int,
    xgb_cfg: dict | None = None,
    dnn_cfg: dict | None = None,
    en_cfg: dict | None = None,
    val_idx: np.ndarray | None = None,
) -> FoldResult:
    """
    在一折内（v2.1：输入已全局插补好）：
      fit scaler (训) → fit model (训) → predict (验) → 算指标。
    """
    log(f"[fold seed={fold_seed}] start | model={model_name} | n_tr={len(X_tr_df):,} n_va={len(X_va_df):,}")
    X_tr_arr = X_tr_df[feat_cols].values.astype(np.float64)
    X_va_arr = X_va_df[feat_cols].values.astype(np.float64)
    if np.isnan(X_tr_arr).any() or np.isnan(X_va_arr).any():
        raise RuntimeError("eval_one_fold 收到含 NaN 的特征——应来自 features_imputed_v2_*.parquet")

    scaler = StandardScaler().fit(X_tr_arr)
    X_tr = scaler.transform(X_tr_arr).astype(np.float32)
    X_va = scaler.transform(X_va_arr).astype(np.float32)

    t0 = time.time()
    extra: dict[str, Any] = {}
    if model_name == "lr":
        m = make_lr()
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
    elif model_name == "en":
        cfg = en_cfg or {}
        m = make_en(
            l1_ratio_candidates=cfg.get("l1_ratio_candidates", [0.1, 0.5, 0.9, 1.0]),
            cv=cfg.get("cv", 5),
            n_alphas=cfg.get("n_alphas", 100),
            max_iter=cfg.get("max_iter", 5000),
            random_state=cfg.get("random_state", 42),
        )
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
        extra["best_alpha"] = float(m.alpha_)
        extra["best_l1_ratio"] = float(m.l1_ratio_)
        # 非零系数数（衡量稀疏度，方便后续报告）
        extra["n_nonzero"] = int(np.sum(np.abs(m.coef_) > 1e-12))
    elif model_name == "xgb":
        cfg = xgb_cfg or {}
        m = make_xgb(
            hp,
            n_estimators_max=cfg.get("n_estimators_max", 2000),
            early_stopping_rounds=cfg.get("early_stopping_rounds", 50),
        )
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        # XGBoost 的 best_iteration 在装 callbacks 时才有；不强求 early stopping，
        # 退化为 n_estimators_max。两种情况都正确。
        best_iter = getattr(m, "best_iteration", None)
        if best_iter is None:
            best_iter = m.n_estimators
        extra["best_iter"] = int(best_iter)
        pred = m.predict(X_va)
    elif model_name == "dnn":
        cfg = dnn_cfg or {}
        pred, best_ep = train_dnn(
            X_tr, y_tr, X_va, y_va,
            hp=hp,
            epochs_max=cfg.get("epochs_max", 200),
            patience=cfg.get("patience", 15),
        )
        extra["best_epoch"] = int(best_ep)
    else:
        raise ValueError(f"unknown model {model_name}")

    log(f"  [model:{model_name}] fit+predict in {time.time() - t0:.1f}s")
    return FoldResult(
        val_pred=pred.astype(np.float64),
        val_idx=val_idx if val_idx is not None else np.array([]),
        val_metrics=metrics_dict(y_va, pred),
        extra=extra,
    )


# ---------- StratifiedKFold 包装 (按 age 五分位) ----------

def make_kfold_iter(y: np.ndarray, n_splits: int, seed: int):
    strat = stratify_bins(y, n_bins=5)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return skf.split(np.zeros(len(y)), strat)
