"""
make_split.py — 一次性切分训练 / 测试，封存到 data/split.parquet。

输入: data/features_raw.parquet（preprocess.py 输出）
输出: data/split.parquet  列 = {eid, sex, age_quintile, split}
       split ∈ {train, test}
       比例 = config.split.test_ratio
       策略 = 男女各自独立切分；按 age 五分位分层

防护: 如果 split.parquet 已存在则报错退出，避免误覆盖。
       如需重切，手动 rm data/split.parquet 再跑。
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from v2_lib import (
    SEX_FEMALE, SEX_MALE, TARGET_COL,
    load_config, load_raw_features, stratify_bins,
)


def run(config_path: Path, force: bool = False) -> None:
    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    out_path = data_dir / "split.parquet"

    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} 已存在；如要重切请手动删除后再跑（保护测试集封存）"
        )

    df = load_raw_features(data_dir)
    print(f"[split] features_raw shape={df.shape}")

    sp_cfg = cfg["split"]
    test_ratio = sp_cfg["test_ratio"]
    seed = sp_cfg["seed"]
    n_bins = sp_cfg["stratify_age_bins"]
    by_sex = sp_cfg.get("by_sex", True)

    rows = []
    if by_sex:
        for sex_val, label in [(SEX_MALE, "male"), (SEX_FEMALE, "female")]:
            sub = df[df["sex"] == sex_val].reset_index(drop=True)
            if sub.empty:
                continue
            ages = sub[TARGET_COL].values.astype(np.float64)
            quintile = stratify_bins(ages, n_bins=n_bins)

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=test_ratio, random_state=seed,
            )
            tr_idx, te_idx = next(sss.split(np.zeros(len(sub)), quintile))
            split_arr = np.array(["train"] * len(sub), dtype=object)
            split_arr[te_idx] = "test"

            sub_out = pd.DataFrame({
                "eid": sub["eid"].values,
                "sex": sex_val,
                "age_quintile": quintile.astype(int),
                "split": split_arr,
            })
            rows.append(sub_out)
            n_tr, n_te = (split_arr == "train").sum(), (split_arr == "test").sum()
            print(f"[split:{label}] n={len(sub):,}  train={n_tr:,}  test={n_te:,}  "
                  f"(test ratio = {n_te / len(sub):.4f})")
    else:
        ages = df[TARGET_COL].values.astype(np.float64)
        quintile = stratify_bins(ages, n_bins=n_bins)
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed,
        )
        tr_idx, te_idx = next(sss.split(np.zeros(len(df)), quintile))
        split_arr = np.array(["train"] * len(df), dtype=object)
        split_arr[te_idx] = "test"
        rows.append(pd.DataFrame({
            "eid": df["eid"].values,
            "sex": df["sex"].values,
            "age_quintile": quintile.astype(int),
            "split": split_arr,
        }))

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(out_path, index=False)
    print(f"\n[split] -> {out_path}")
    print(f"[split] total rows = {len(out):,}")
    print(out.groupby(["sex", "split"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--force", action="store_true",
                    help="强制覆盖已有 split.parquet（默认拒绝以保护测试集）")
    args = ap.parse_args()
    run(Path(args.config), force=args.force)
