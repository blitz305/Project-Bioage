"""
spearman_report.py — v2.2：53 个特征 vs Chronological_age 的 Spearman ρ + BH-FDR。

设计：
  • 只在 train 区算（≠test 集，避免泄漏）
  • 按性别独立报告（两组人群的相关性结构不同）
  • 只生成报告，不真的从特征列表里删（XGB/DNN 自带特征选择；模块内 7-8 维已属手工挑选）
  • 输出 outputs/v2/spearman_report_{sex}.csv，列：
      feature, n, rho, p, p_bh, abs_rho_rank, flag_pass

CLI:
  python spearman_report.py                    # 默认两性别都跑
  python spearman_report.py --sex female
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from v2_lib import (
    TARGET_COL,
    imputed_feature_columns, load_config, load_imputed_features, log,
)


def spearman_one_sex(df_imp: pd.DataFrame, *, abs_rho_thresh: float,
                     fdr_alpha: float) -> pd.DataFrame:
    df_train = df_imp[df_imp["split"] == "train"].reset_index(drop=True)
    feat_cols = imputed_feature_columns(df_train)
    age = df_train[TARGET_COL].values.astype(np.float64)

    rows = []
    for col in feat_cols:
        x = df_train[col].values.astype(np.float64)
        mask = ~(np.isnan(x) | np.isnan(age))
        if mask.sum() < 100:
            rows.append({"feature": col, "n": int(mask.sum()),
                         "rho": np.nan, "p": np.nan})
            continue
        rho, p = stats.spearmanr(x[mask], age[mask])
        rows.append({"feature": col, "n": int(mask.sum()),
                     "rho": float(rho), "p": float(p)})

    out = pd.DataFrame(rows)
    valid = out["p"].notna()
    out["p_bh"] = np.nan
    if valid.any():
        _, p_bh, _, _ = multipletests(out.loc[valid, "p"].values,
                                      alpha=fdr_alpha, method="fdr_bh")
        out.loc[valid, "p_bh"] = p_bh

    out["abs_rho_rank"] = out["rho"].abs().rank(ascending=False, method="min")
    out["flag_pass"] = (out["rho"].abs() > abs_rho_thresh) & (out["p_bh"] < fdr_alpha)
    out = out.sort_values("abs_rho_rank").reset_index(drop=True)
    return out


def run(config_path: Path, sex_arg: str | None) -> None:
    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    out_dir = (base / cfg["paths"]["outputs_dir"]).resolve() / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    sp_cfg = cfg.get("spearman", {})
    abs_rho_thresh = float(sp_cfg.get("abs_rho_threshold", 0.1))
    fdr_alpha = float(sp_cfg.get("fdr_alpha", 0.05))

    sex_labels = [sex_arg] if sex_arg else ["female", "male"]
    for sex_label in sex_labels:
        log(f"=== spearman | sex={sex_label} ===")
        df_imp = load_imputed_features(data_dir, sex_label)
        rep = spearman_one_sex(df_imp, abs_rho_thresh=abs_rho_thresh,
                               fdr_alpha=fdr_alpha)
        out_path = out_dir / f"spearman_report_{sex_label}.csv"
        rep.to_csv(out_path, index=False)
        n_pass = int(rep["flag_pass"].sum())
        log(f"  [{sex_label}] {n_pass}/{len(rep)} features pass "
            f"(|rho|>{abs_rho_thresh} & p_bh<{fdr_alpha})")
        log(f"  top 5: {list(rep.head(5)['feature'])}")
        log(f"  -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--sex", choices=["male", "female"], default=None,
                    help="只跑指定性别；默认两性别都跑")
    args = ap.parse_args()
    run(Path(args.config), args.sex)
