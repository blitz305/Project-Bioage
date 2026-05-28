"""
impute_v2.py — 全局一次性 miceforest 插补（v2 流程的新前置步骤）。

设计：
  • 按性别独立处理。
  • 仅在 train 区 (split=='train') fit miceforest，age 不进 feat_cols。
  • 用 fitted kernel 同时 transform train + test，拼回元数据写出。
  • 输出 data/features_imputed_v2_{sex}.parquet
      列：eid, sex, split, Chronological_age, <53 个 ML 特征>

跑一次（两性别串行）大概 20-40 分钟，~3GB 内存。
之后 tune.py / train_v2.py 直接读，外层 5-fold 不再做 per-fold MICE。

CLI:
  python impute_v2.py            # 默认两性别都跑
  python impute_v2.py --sex female
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import miceforest as mf

from v2_lib import (
    SEX_FEMALE, SEX_MALE, TARGET_COL,
    feature_columns, load_config, load_raw_features, load_split,
    log,
)


SEX_MAP = {"female": SEX_FEMALE, "male": SEX_MALE}


def impute_one_sex(df_raw: pd.DataFrame, split_df: pd.DataFrame,
                   sex_val: int, sex_label: str,
                   feat_cols: list[str], iterations: int,
                   random_state: int) -> pd.DataFrame:
    sub_split = split_df[split_df["sex"] == sex_val][["eid", "split"]]
    df = df_raw.merge(sub_split, on="eid", how="inner").reset_index(drop=True)
    train_mask = (df["split"] == "train").values
    test_mask = (df["split"] == "test").values

    log(f"[{sex_label}] n_total={len(df):,} n_train={train_mask.sum():,} n_test={test_mask.sum():,}")
    log(f"[{sex_label}] feat_cols={len(feat_cols)} (age excluded)")

    X_train = df.loc[train_mask, feat_cols].reset_index(drop=True).copy()

    # —— fit miceforest on train ——
    log(f"[{sex_label}] miceforest fit start (iterations={iterations}) ...")
    t0 = time.time()
    kernel = mf.ImputationKernel(
        data=X_train,
        num_datasets=1,
        random_state=random_state,
        # 必须保留全部迭代数据，否则 impute_new_data() 会 assert 失败
    )
    kernel.mice(iterations=iterations, verbose=True)
    log(f"[{sex_label}] miceforest fit done in {time.time() - t0:.1f}s")

    # —— transform train + test ——
    completed_train = kernel.complete_data(0)
    log(f"[{sex_label}] transform test ...")
    t0 = time.time()
    X_test = df.loc[test_mask, feat_cols].reset_index(drop=True).copy()
    completed_test = kernel.impute_new_data(X_test).complete_data(0)
    log(f"[{sex_label}] transform test done in {time.time() - t0:.1f}s")

    # —— mean fallback：训练时无缺失列、test 上首次出现缺失的兜底 ——
    train_arr = completed_train[feat_cols].values.astype(np.float64)
    test_arr = completed_test[feat_cols].values.astype(np.float64)
    col_means = np.nanmean(train_arr, axis=0)
    if np.isnan(test_arr).any():
        inds = np.where(np.isnan(test_arr))
        n_fixed = len(inds[0])
        test_arr[inds] = np.take(col_means, inds[1])
        log(f"[{sex_label}] WARN: filled {n_fixed} residual NaNs in test with train col means")
    if np.isnan(train_arr).any():
        inds = np.where(np.isnan(train_arr))
        n_fixed = len(inds[0])
        train_arr[inds] = np.take(col_means, inds[1])
        log(f"[{sex_label}] WARN: filled {n_fixed} residual NaNs in train with col means")

    # —— 拼输出 ——
    train_out = pd.DataFrame(train_arr, columns=feat_cols)
    train_out.insert(0, "eid", df.loc[train_mask, "eid"].values)
    train_out.insert(1, "sex", df.loc[train_mask, "sex"].values)
    train_out.insert(2, "split", "train")
    train_out.insert(3, TARGET_COL, df.loc[train_mask, TARGET_COL].values)

    test_out = pd.DataFrame(test_arr, columns=feat_cols)
    test_out.insert(0, "eid", df.loc[test_mask, "eid"].values)
    test_out.insert(1, "sex", df.loc[test_mask, "sex"].values)
    test_out.insert(2, "split", "test")
    test_out.insert(3, TARGET_COL, df.loc[test_mask, TARGET_COL].values)

    out = pd.concat([train_out, test_out], ignore_index=True)

    # —— 完整性检查 ——
    n_nan = out[feat_cols].isna().sum().sum()
    if n_nan > 0:
        raise RuntimeError(f"[{sex_label}] still {n_nan} NaNs after imputation, abort")
    log(f"[{sex_label}] all good, n_rows={len(out):,} n_features={len(feat_cols)}")
    return out


def run(config_path: Path, sex_arg: str | None) -> None:
    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()

    df_raw = load_raw_features(data_dir)
    split_df = load_split(data_dir)
    feat_cols = feature_columns(df_raw)  # 已自动排除 eid / sex / Chronological_age

    mice_cfg = cfg["tuning"]["mice"]
    iterations = mice_cfg["iterations"]
    random_state = mice_cfg["random_state"]

    sex_labels = [sex_arg] if sex_arg else ["female", "male"]
    for sex_label in sex_labels:
        sex_val = SEX_MAP[sex_label]
        out_path = data_dir / f"features_imputed_v2_{sex_label}.parquet"
        log(f"=== imputing sex={sex_label} -> {out_path.name} ===")
        out = impute_one_sex(
            df_raw=df_raw, split_df=split_df,
            sex_val=sex_val, sex_label=sex_label,
            feat_cols=feat_cols,
            iterations=iterations,
            random_state=random_state + sex_val,  # 男女不同种子
        )
        out.to_parquet(out_path, index=False)
        log(f"-> {out_path}  (rows={len(out):,}, cols={len(out.columns)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--sex", choices=["male", "female"], default=None,
                    help="只跑指定性别；默认两性别都跑")
    args = ap.parse_args()
    run(Path(args.config), args.sex)
