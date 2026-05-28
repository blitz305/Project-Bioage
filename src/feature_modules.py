"""
feature_modules.py — v2.2 模块化时钟的特征清单（单一权威定义）。

模块映射依据：
  • metabolic / liver / kidney / immune 四个器官系统模块
    按《生物学年龄小程序设计方案(3)》§8.2 选指标，但用 UKB baseline 实际可用的列名
  • all = 综合模型，等于现有 v2 流程的 53 维全特征
  • SDOH (Townsend / NO2 / PM10 / Education / Income / Friend_visits 等)
    属于社会暴露，不进任何器官模块；只在 all 里出现

特征命名严格对齐 features_imputed_v2_{sex}.parquet 的列名（来自 preprocess.py
按 feature_dictionary.md 命名）。如果 parquet 列名变化，这里必须同步改。
"""
from __future__ import annotations


MODULE_FEATURES: dict[str, list[str]] = {
    "metabolic": [
        "BMI",
        "Glucose",
        "HbA1c",
        "Triglycerides",
        "Cholesterol",
        "LDL_direct",
        "ApolipoproteinA",
    ],
    "liver": [
        "ALT",
        "AST",
        "GGT",
        "ALP",
        "Total_bilirubin",
        "Direct_bilirubin",
        "Albumin",
        "Total_protein",
    ],
    "kidney": [
        "Creatinine",
        "Urea",
        "CystatinC",
        "Urine_creatinine",
        "Urine_potassium",
        "Urine_sodium",
        "Calcium",
        "Phosphate",
    ],
    "immune": [
        "WBC",
        "Neutrophill_count",
        "Lymphocyte_pct",
        "Monocyte_pct",
        "Platelet_count",
        "RDW",
        "CRP",
    ],
}

ALL_KEY = "all"
MODULE_KEYS = list(MODULE_FEATURES.keys()) + [ALL_KEY]


def resolve_features(features_arg: str, full_feat_cols: list[str]) -> list[str]:
    """
    根据 --features 参数返回实际特征列表。
      features_arg == 'all' → full_feat_cols（综合模型，53 维）
      其它 key → MODULE_FEATURES[key]，并校验每个特征都在 full_feat_cols 里
    """
    if features_arg == ALL_KEY:
        return list(full_feat_cols)
    if features_arg not in MODULE_FEATURES:
        raise ValueError(
            f"unknown features key: {features_arg!r}; "
            f"options = {MODULE_KEYS}"
        )
    cols = MODULE_FEATURES[features_arg]
    missing = [c for c in cols if c not in full_feat_cols]
    if missing:
        raise KeyError(
            f"模块 {features_arg!r} 缺特征 {missing}（imputed parquet 里没有这些列）"
        )
    return list(cols)


def output_suffix(features_arg: str) -> str:
    """文件名约定：features=='all' 时不加后缀（沿用旧名），其他模块加 _{key}。"""
    if features_arg == ALL_KEY:
        return ""
    return f"_{features_arg}"


def gam_feature_columns(full_feat_cols: list[str]) -> list[str]:
    """
    GAM 轨迹分析的范围 = 43 个体测/生化（≈ 4 个器官模块的并集）。
    去重保留 parquet 出现顺序。SDOH 不画 GAM（社会变量与年龄是采样 artifact）。
    """
    in_modules = set()
    for cols in MODULE_FEATURES.values():
        in_modules.update(cols)
    # 加 4 个器官模块未覆盖、但仍属体测/生化的指标（按 feature_dictionary.md §1）
    extra_biomarkers = {
        "DBP", "SBP", "FEV1",
        "RBC", "Haemoglobin", "MCV", "Platelet_crit",
        "Mean_sphered_cell_volume",
        "IGF1", "SHBG", "Testosterone", "VitaminD",
        "lnCRP",  # PhenoAge 用的 log-CRP（与 CRP 并存）
    }
    in_modules.update(extra_biomarkers)
    return [c for c in full_feat_cols if c in in_modules]
