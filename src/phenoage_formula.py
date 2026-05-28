"""
phenoage_formula.py — 按 Levine 2018 原始公式算 PhenoAge。

输入: data/features_imputed_{male,female}.parquet
输出:
    data/phenoage_formula_male.parquet
    data/phenoage_formula_female.parquet

每行: eid, sex, age, PhenoAge_baseline, PhenoAge_accel
其中 PhenoAge_accel = PhenoAge - LinearReg(PhenoAge ~ age) 的残差
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def phenoage(df: pd.DataFrame) -> pd.Series:
    """
    Levine 2018 PhenoAge 公式，按 BioAge R 包（dayoonkwon/BioAge）的规范实现。

    单位约定（系数已为这些单位校准，无需换算）：
      Albumin: g/L
      Creatinine: μmol/L
      Glucose: mmol/L
      CRP: mg/L → ln
      Lymphocyte%: %
      MCV: fL（用 30270 Mean sphered cell volume）
      RDW: %
      ALP: U/L
      WBC: 10^9/L
      age: years
    """
    albumin   = df["Albumin"]
    creat     = df["Creatinine"]
    glucose   = df["Glucose"]
    # MICE 偶尔会给正值变量插补出负值；ln 前 clip 到 UKB 检测下限
    lncrp     = np.log(np.clip(df["CRP"], 0.08, None))
    lymph     = df["Lymphocyte_pct"]
    mcv       = df["Mean_sphered_cell_volume"]
    rdw       = df["RDW"]
    alp       = df["ALP"]
    wbc       = df["WBC"]
    age       = df["Chronological_age"]

    xb = (
        -19.907
        - 0.0336 * albumin
        + 0.0095 * creat
        + 0.1953 * glucose
        + 0.0954 * lncrp
        - 0.0120 * lymph
        + 0.0268 * mcv
        + 0.3306 * rdw
        + 0.00188 * alp
        + 0.0554 * wbc
        + 0.0804 * age
    )
    # 注意分母是 +0.0076927；BioAge R 包源码为准
    M = 1.0 - np.exp(-1.51714 * np.exp(xb) / 0.0076927)
    M = np.clip(M, 1e-12, 1 - 1e-12)
    pheno = 141.50225 + np.log(-0.00553 * np.log(1.0 - M)) / 0.090165
    return pheno


def add_acceleration(pheno: np.ndarray, age: np.ndarray) -> np.ndarray:
    """对 chronological age 取残差。"""
    lr = LinearRegression().fit(age.reshape(-1, 1), pheno)
    return pheno - lr.predict(age.reshape(-1, 1))


def run(config_path: Path) -> None:
    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()

    for label in ("male", "female"):
        path = data_dir / f"features_imputed_{label}.parquet"
        if not path.exists():
            print(f"[phenoage] 缺 {path}，跳过")
            continue
        df = pd.read_parquet(path)
        pheno = phenoage(df).values.astype(np.float64)
        accel = add_acceleration(pheno, df["Chronological_age"].values.astype(np.float64))

        out = pd.DataFrame({
            "eid": df["eid"].values,
            "sex": df["sex"].values,
            "age": df["Chronological_age"].values,
            "PhenoAge_baseline": pheno,
            "PhenoAge_accel": accel,
        })
        out_path = data_dir / f"phenoage_formula_{label}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"[phenoage:{label}] n={len(out):,}")
        print(f"  PhenoAge_baseline  mean={pheno.mean():.2f}  sd={pheno.std():.2f}  "
              f"corr_with_age={np.corrcoef(pheno, df['Chronological_age'])[0,1]:.3f}")
        print(f"  PhenoAge_accel     mean={accel.mean():+.3f}  sd={accel.std():.2f}")
        print(f"  -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = ap.parse_args()
    run(Path(args.config))
