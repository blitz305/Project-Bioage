"""
build_outcomes.py — 构造生存分析需要的 outcome 表。

输入: 整合指标.csv（取 eid, p34, p21022, p40000_i0, p40007_i0, p41270, p41280_a0..258）
输出: data/outcomes.parquet

每人一行，列：
    eid
    baseline_date           ≈ datetime(yob + age_at_recruitment, 7, 1)
    censor_date             全 cohort 用同一个 = max(p40000_i0)
    death_event             {0,1}
    death_time_years        距 baseline 的年数
    cvd_prevalent / cvd_event / cvd_time_years
    t2d_prevalent / t2d_event / t2d_time_years
    cancer_prevalent / cancer_event / cancer_time_years
    died_within_2yr         baseline 后 ≤2 年死亡

ICD10 前缀（无点）:
    CVD     = I20..I25, I60..I69
    T2D     = E11
    Cancer  = C00..C97
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------- ICD10 前缀 ----------

CVD_PREFIXES = tuple([f"I{i}" for i in range(20, 26)] + [f"I{i}" for i in range(60, 70)])
T2D_PREFIXES = ("E11",)
CANCER_PREFIXES = tuple(f"C{str(i).zfill(2)}" for i in range(0, 98))

OUTCOME_DEFS = {
    "cvd": CVD_PREFIXES,
    "t2d": T2D_PREFIXES,
    "cancer": CANCER_PREFIXES,
}


# ---------- 工具 ----------

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_codes(raw) -> list[str]:
    if pd.isna(raw):
        return []
    try:
        v = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return []
    if not isinstance(v, (list, tuple)):
        v = [v]
    return [str(x) for x in v if pd.notna(x)]


def first_match_date(codes: list[str], dates: list, prefixes: tuple[str, ...]):
    """对齐 codes 与 dates，返回 prefixes 命中的最早日期；都没命中返回 NaT。"""
    earliest = pd.NaT
    n = min(len(codes), len(dates))
    for c, d in zip(codes[:n], dates[:n]):
        if pd.isna(d):
            continue
        if c.startswith(prefixes):
            d = pd.to_datetime(d, errors="coerce")
            if pd.isna(d):
                continue
            if pd.isna(earliest) or d < earliest:
                earliest = d
    return earliest


# ---------- main ----------

def run(config_path: Path, nrows: int | None = None) -> None:
    cfg = load_config(config_path)
    base = config_path.parent
    raw_csv = (base / cfg["paths"]["raw_csv"]).resolve()
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    date_cols = [f"p41280_a{i}" for i in range(0, 259)]
    base_cols = ["eid", "p34", "p21022",
                 "p40000_i0", "p40007_i0",
                 "p41270"]
    use_cols = base_cols + date_cols

    print(f"[outcomes] reading {raw_csv} (nrows={nrows or 'ALL'})")
    df = pd.read_csv(raw_csv, usecols=use_cols, nrows=nrows, low_memory=False)
    print(f"[outcomes] loaded {len(df):,} rows")

    # baseline_date
    yob = df["p34"].astype("Int64")
    age0 = df["p21022"].astype("Int64")
    base_year = (yob + age0).astype("float")
    df["baseline_date"] = pd.to_datetime(
        base_year.astype("Int64").astype(str) + "-07-01",
        errors="coerce",
    )

    # death
    df["death_date"] = pd.to_datetime(df["p40000_i0"], errors="coerce")
    df["death_event"] = df["death_date"].notna().astype(int)

    # 行政 censoring date = max(death_date) 全 cohort 一致
    censor_date = df["death_date"].max()
    print(f"[outcomes] inferred censor_date = {censor_date.date()}")
    df["censor_date"] = censor_date

    # death_time_years
    end_date = df["death_date"].fillna(df["censor_date"])
    df["death_time_years"] = (end_date - df["baseline_date"]).dt.days / 365.25

    # died within 2yr
    df["died_within_2yr"] = (df["death_event"] == 1) & (df["death_time_years"] <= 2.0)

    # ICD10 outcomes
    print("[outcomes] parsing p41270 / p41280 ...")
    code_lists = df["p41270"].apply(parse_codes)
    date_arrays = df[date_cols].values.tolist()  # list-of-lists
    # 释放大列
    df = df.drop(columns=date_cols + ["p41270"])

    for name, prefixes in OUTCOME_DEFS.items():
        first = pd.Series(
            [first_match_date(c, d, prefixes) for c, d in zip(code_lists, date_arrays)],
            index=df.index, dtype="datetime64[ns]",
        )
        df[f"{name}_first_date"] = first
        df[f"{name}_prevalent"] = (
            first.notna() & (first <= df["baseline_date"])
        )
        # event = first 在 baseline 之后 且 ≤ censor，发生即 event=1
        is_incident = first.notna() & (first > df["baseline_date"])
        df[f"{name}_event"] = is_incident.astype(int)
        # time = min(first, censor or death) - baseline
        end = first.where(is_incident, df["censor_date"])
        # 死亡发生在结局之前则截尾到死亡
        end = end.where(~((df["death_event"] == 1) & (df["death_date"] < end)),
                        df["death_date"])
        df[f"{name}_time_years"] = (end - df["baseline_date"]).dt.days / 365.25

    keep_cols = [
        "eid", "baseline_date", "censor_date",
        "death_event", "death_time_years", "died_within_2yr",
        "cvd_event", "cvd_time_years", "cvd_prevalent",
        "t2d_event", "t2d_time_years", "t2d_prevalent",
        "cancer_event", "cancer_time_years", "cancer_prevalent",
    ]
    out = df[keep_cols].copy()

    # 简要统计
    print("\n[outcomes] summary")
    print(f"  n total              : {len(out):,}")
    print(f"  deaths               : {out.death_event.sum():,}")
    print(f"  died_within_2yr      : {out.died_within_2yr.sum():,}")
    for k in OUTCOME_DEFS:
        print(f"  {k:8s} prevalent   : {int(out[f'{k}_prevalent'].sum()):,}")
        print(f"  {k:8s} incident    : {int(out[f'{k}_event'].sum()):,}")

    out_path = data_dir / "outcomes.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n[outcomes] saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--nrows", type=int, default=None,
                    help="只读前 N 行做 smoke test")
    args = ap.parse_args()
    run(Path(args.config), nrows=args.nrows)
