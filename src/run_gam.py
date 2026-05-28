"""
run_gam.py — Python 包装：调 Rscript gam_trajectories.R 跑 GAM 年龄轨迹。

流程：
  1. 读 features_imputed_v2_{sex}.parquet 的 train 区
  2. 按 feature_modules.gam_feature_columns() 取 43 个体测/生化（跳过 SDOH）
  3. 写到 tmp parquet（只含 age + 43 列）
  4. subprocess.run(["Rscript", "gam_trajectories.R", tmp, sex, out_dir, ...])
  5. 校验产物齐全

CLI:
  python run_gam.py            # 默认两性别都跑
  python run_gam.py --sex female
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from feature_modules import gam_feature_columns
from v2_lib import (
    TARGET_COL,
    imputed_feature_columns, load_config, load_imputed_features, log,
)


def gam_one_sex(*, data_dir: Path, out_gam_dir: Path, sex_label: str,
                r_script: Path, gam_cfg: dict) -> None:
    df_imp = load_imputed_features(data_dir, sex_label)
    df_train = df_imp[df_imp["split"] == "train"].reset_index(drop=True)
    full_feat_cols = imputed_feature_columns(df_imp)
    gam_cols = gam_feature_columns(full_feat_cols)
    log(f"[{sex_label}] n_train={len(df_train):,} | n_gam_features={len(gam_cols)}")

    sub = df_train[[TARGET_COL, *gam_cols]].copy()

    with tempfile.NamedTemporaryFile(
        prefix=f"gam_input_{sex_label}_", suffix=".parquet", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sub.to_parquet(tmp_path, index=False)
        cmd = [
            "Rscript", str(r_script),
            str(tmp_path), sex_label, str(out_gam_dir),
            TARGET_COL,
            str(int(gam_cfg.get("n_splines", 20))),
            str(float(gam_cfg.get("fdr_alpha", 0.01))),
            str(float(gam_cfg.get("edf_threshold", 3.0))),
            str(float(gam_cfg.get("age_grid_min", 38))),
            str(float(gam_cfg.get("age_grid_max", 73))),
            str(int(gam_cfg.get("age_grid_n", 50))),
        ]
        log(f"[{sex_label}] running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Rscript 退出码 {proc.returncode}")
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    # —— 校验产物 ——
    expected = [
        out_gam_dir / f"gam_summary_{sex_label}.csv",
        out_gam_dir / f"gam_curves_{sex_label}.png",
        out_gam_dir / f"gam_heatmap_{sex_label}.png",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(f"[{sex_label}] 缺产物：{missing}")
    log(f"[{sex_label}] all GAM outputs OK")


def run(config_path: Path, sex_arg: str | None) -> None:
    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    out_dir = (base / cfg["paths"]["outputs_dir"]).resolve() / "v2"
    out_gam_dir = out_dir / "gam"
    out_gam_dir.mkdir(parents=True, exist_ok=True)

    gam_cfg = cfg.get("gam", {})
    r_script = base / "gam_trajectories.R"
    if not r_script.exists():
        raise FileNotFoundError(f"缺 R 脚本：{r_script}")
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript 不在 PATH；请确认 conda activate bioage")

    sex_labels = [sex_arg] if sex_arg else ["female", "male"]
    for sex_label in sex_labels:
        log(f"=== GAM | sex={sex_label} ===")
        gam_one_sex(
            data_dir=data_dir, out_gam_dir=out_gam_dir,
            sex_label=sex_label, r_script=r_script, gam_cfg=gam_cfg,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--sex", choices=["male", "female"], default=None,
                    help="只跑指定性别；默认两性别都跑")
    args = ap.parse_args()
    run(Path(args.config), args.sex)
