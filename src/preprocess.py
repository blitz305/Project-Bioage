"""
preprocess.py — 把 整合指标.csv 处理成 ML 可用的 wide-table。

流程：
  1. 按 config.yaml 列出的字段从大 CSV 中抽列（streaming，不读全表）
  2. 数组型字段聚合（mean / max / first / education_fold）
  3. 特殊编码 → NA
  4. 翻转 1031（大值 = 社会接触多）
  5. 构造 lnCRP
  6. 按 列缺失率 > 15% 自动剔除该特征（并落日志）
  7. 输出 features_raw.parquet（不做插补、不做标准化）

下一步由 impute.py 处理。
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------- IO ----------

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_raw_columns(cfg: dict) -> list[str]:
    cols = {"eid", cfg["sex_col"]}
    for feat in cfg["features"]:
        cols.update(feat["raw_cols"])
    return sorted(cols)


# ---------- 聚合策略 ----------

def aggregate_first(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols[0]]


def aggregate_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].mean(axis=1, skipna=True)


def aggregate_max(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].max(axis=1, skipna=True)


def fold_education(values: pd.Series, rule: dict) -> pd.Series:
    """
    按 R 参考代码逻辑折叠 p6138_i0 多选数组（CSV 中是字符串 '[1, 2, 3]'）。
    返回 0/1/2 的有序整数（Below / High / College），无效 → NaN。
    """
    drop = set(rule["drop_codes"])
    mapping_text_to_codes = rule["mapping"]
    levels = rule["ordered_levels"]
    text_to_int = {lvl: i for i, lvl in enumerate(levels)}
    code_to_text = {}
    for text, codes in mapping_text_to_codes.items():
        for c in codes:
            code_to_text[int(c)] = text

    def _one(raw):
        if pd.isna(raw):
            return np.nan
        try:
            arr = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except (ValueError, SyntaxError):
            return np.nan
        if not isinstance(arr, (list, tuple)):
            arr = [arr]
        # 步骤 1: 去掉 -3 (Prefer not to answer)
        arr = [int(x) for x in arr if pd.notna(x) and int(x) not in drop]
        if not arr:
            return np.nan
        # 步骤 2: 取最小编码 = 最高学历（含 -7 None of the above）
        # 注意：-7 比 1..6 都小，会被 min() 选中 → 单独处理
        non_neg = [x for x in arr if x > 0]
        if non_neg:
            top = min(non_neg)
        elif -7 in arr:
            top = -7
        else:
            return np.nan
        text = code_to_text.get(top)
        if text is None:
            return np.nan
        return text_to_int[text]

    return values.map(_one).astype("float64")


# ---------- 单特征 build ----------

def build_feature(df: pd.DataFrame, feat: dict, generic_na: list[int]) -> pd.Series:
    cols = feat["raw_cols"]
    agg = feat.get("aggregate", "first")
    name = feat["name"]

    if agg == "first":
        s = aggregate_first(df, cols)
    elif agg == "mean":
        s = aggregate_mean(df, cols)
    elif agg == "max":
        s = aggregate_max(df, cols)
    elif agg == "education_fold":
        s = fold_education(df[cols[0]], feat["fold_rule"])
        s.name = name
        return s
    else:
        raise ValueError(f"未知 aggregate={agg} for {name}")

    # 特殊编码 → NA（数值列）
    na_codes = set(feat.get("na_codes", []) or [])
    na_codes.update(generic_na)
    if na_codes and pd.api.types.is_numeric_dtype(s):
        s = s.where(~s.isin(na_codes))

    # 1031 方向翻转
    if feat.get("direction_flip"):
        # p1031: 1=daily ... 6=never, 7=no friends. 翻为大值=接触多。
        # 处理方式：把 7 视为最差（合并到原来的 6 → 翻转后的 1）
        # 简化：score = 8 - x, 然后 7 → 1（已自然成立）
        s = (8 - s).where(~s.isin([])).where(s.notna())

    # CRP → lnCRP（feature 名为 lnCRP 时单独取 ln）
    if name == "lnCRP":
        # CRP 不能为 0 / 负
        s = np.log(s.where(s > 0))

    s.name = name
    return s


# ---------- main ----------

def run(config_path: Path, nrows: int | None = None) -> None:
    cfg = load_config(config_path)
    base = config_path.parent
    raw_csv = (base / cfg["paths"]["raw_csv"]).resolve()
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess] raw_csv = {raw_csv}")
    print(f"[preprocess] data_dir = {data_dir}")

    use_cols = collect_raw_columns(cfg)
    print(f"[preprocess] reading {len(use_cols)} columns "
          f"(nrows={nrows if nrows else 'ALL'}) ...")
    df = pd.read_csv(raw_csv, usecols=use_cols, low_memory=False, nrows=nrows)
    print(f"[preprocess] loaded {len(df):,} rows")

    out = pd.DataFrame({"eid": df["eid"], "sex": df[cfg["sex_col"]]})
    for feat in cfg["features"]:
        s = build_feature(df, feat, cfg.get("generic_na_codes", []))
        out[feat["name"]] = s

    # 列缺失率
    feat_cols = [f["name"] for f in cfg["features"]]
    miss = out[feat_cols].isna().mean().sort_values(ascending=False)
    print("\n[preprocess] 列缺失率 (top 15):")
    print(miss.head(15).to_string())

    threshold = cfg.get("max_col_missing_rate", 0.15)
    drop_cols = miss[miss > threshold].index.tolist()
    if drop_cols:
        print(f"\n[preprocess] 缺失率 > {threshold:.0%} → 剔除特征: {drop_cols}")
        out = out.drop(columns=drop_cols)

    # 行缺失率（按剩余特征）
    remain_feats = [c for c in feat_cols if c not in drop_cols]
    row_miss = out[remain_feats].isna().mean(axis=1)
    keep_mask = row_miss <= cfg.get("max_row_missing_rate", 0.15)
    print(f"\n[preprocess] 行缺失率 > 15% 的样本: {(~keep_mask).sum():,} / {len(out):,}")
    out = out[keep_mask].reset_index(drop=True)

    # 性别拆分前先保存 raw（未插补、未标准化、未拆分）
    out_path = data_dir / "features_raw.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n[preprocess] saved {out_path}  shape={out.shape}")

    # 顺手存一个 dropped/kept 报告
    report = {
        "n_rows_in": int(len(df)),
        "n_rows_out": int(len(out)),
        "n_features_kept": len(remain_feats),
        "dropped_features": drop_cols,
        "col_missing_rate_before": miss.round(4).to_dict(),
    }
    rpt_path = data_dir / "preprocess_report.json"
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[preprocess] report  -> {rpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--nrows", type=int, default=None,
                    help="只读前 N 行做 smoke test")
    args = ap.parse_args()
    run(Path(args.config), nrows=args.nrows)
