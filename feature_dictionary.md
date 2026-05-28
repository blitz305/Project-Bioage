# 机器学习特征字典 — PhenoAge ∪ SDOHAge

> 输入：PhenoAge 模型 (10) ∪ SDOHAge 最终输入 (51) 的并集，共 **53 个候选特征**。
> 输出：生物学预测年龄。
> 范围：仅整理「进入 ML 模型的输入特征」，不含目标年龄、协变量、结局变量。

## 0. 全局约定（来自两份指导文件）

| 项目 | 说明 |
|---|---|
| 时点 | 全部使用 UKB **baseline assessment** (instance 0) |
| 缺失值 | 单变量缺失率 < 15% → multiple imputation；> 15% 通常剔除该样本/特征。原文未指定算法（可选 MICE / missForest，需自己定） |
| 标准化 | 所有连续 + 编码后的有序变量统一 z-score（μ=0, σ=1） |
| 性别建模 | SDOHAge 原文男女分开；如果统一建模需把 sex (field 31) 作为特征加入 |
| 特征筛选 | SDOHAge 原文对体测/生化做 Boruta + RFE；SDOH 直接全入。若我们自己跑 ML，可选择保留全部 53 维或复刻 Boruta+RFE |
| 编码方向问题 | field 1031 数值越大越孤立、field 2110 数值越大越亲密 — 方向相反，建议统一翻转其中一个 |

---

## 1. 体测 / 血生化 / 尿生化（Physical & Biomarkers）

| # | 字段名（建议列名） | UKB field id | 单位 | 数据类型 | 变换 | 缺失处理 | 出现于 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 1 | BMI | 21001 | kg/m² | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 2 | DBP | 4079 | mmHg | 连续 | z-score | MI(<15%) | SDOH(M+F) | 自动测量 |
| 3 | SBP | 4080 | mmHg | 连续 | z-score | MI(<15%) | SDOH(M+F) | 自动测量 |
| 4 | FEV1 | 3063 | L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 5 | Leukocyte_count (WBC) | 30000 | 10⁹/L | 连续 | z-score | MI(<15%) | SDOH(M+F), Pheno | |
| 6 | Erythrocyte_count (RBC) | 30010 | 10¹²/L | 连续 | z-score | MI(<15%) | SDOH(M only) | |
| 7 | Haemoglobin | 30020 | g/dL | 连续 | z-score | MI(<15%) | SDOH(F only) | |
| 8 | MCV | 30040 | fL | 连续 | z-score | MI(<15%) | SDOH(M+F) | 与下方 30270 不同 |
| 9 | Erythrocyte_distribution_width (RDW) | 30070 | % | 连续 | z-score | MI(<15%) | SDOH(M only), Pheno | |
| 10 | Platelet_count | 30080 | 10⁹/L | 连续 | z-score | MI(<15%) | SDOH(M only) | |
| 11 | Platelet_crit | 30090 | % | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 12 | Neutrophill_count | 30140 | 10⁹/L | 连续 | z-score | MI(<15%) | SDOH(F only) | |
| 13 | Lymphocyte_percentage | 30180 | % | 连续 | z-score | MI(<15%) | SDOH(M only), Pheno | |
| 14 | Monocyte_percentage | 30190 | % | 连续 | z-score | MI(<15%) | SDOH(M only) | |
| 15 | Mean_sphered_cell_volume | 30270 | fL | 连续 | z-score | MI(<15%) | Pheno only | PhenoAge 论文补充表 S13 把 "MCV" 映射到 30270，注意 ≠ 30040 |
| 16 | Urine_creatinine | 30510 | μmol/L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 17 | Urine_potassium | 30520 | mmol/L | 连续 | z-score | MI(<15%) | SDOH(M only) | |
| 18 | Urine_sodium | 30530 | mmol/L | 连续 | z-score | MI(<15%) | SDOH(M only) | |
| 19 | Albumin | 30600 | g/L | 连续 | z-score | MI(<15%) | SDOH(M+F), Pheno | |
| 20 | Alkaline_phosphatase (ALP) | 30610 | U/L | 连续 | z-score | MI(<15%) | SDOH(F only), Pheno | |
| 21 | ALT | 30620 | U/L | 连续 | z-score | MI(<15%) | SDOH(M only) | |
| 22 | ApolipoproteinA | 30630 | g/L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 23 | AST | 30650 | U/L | 连续 | z-score | MI(<15%) | SDOH(F only) | |
| 24 | Direct_bilirubin | 30660 | μmol/L | 连续 | z-score | MI(<15%) | SDOH(F only) | |
| 25 | Urea | 30670 | mmol/L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 26 | Calcium | 30680 | mmol/L | 连续 | z-score | MI(<15%) | SDOH(F only) | |
| 27 | Cholesterol | 30690 | mmol/L | 连续 | z-score | MI(<15%) | SDOH(F only) | |
| 28 | Creatinine | 30700 | μmol/L | 连续 | z-score | MI(<15%) | SDOH(M only), Pheno | |
| 29 | CRP | 30710 | mg/L (raw) | 连续 | **PhenoAge 用 ln(CRP)；SDOHAge 用原始值**。建议保留两列：`crp` 与 `lncrp` | MI(<15%) | SDOH(M+F), Pheno | 唯一变换不一致的特征 |
| 30 | Cystatin_C | 30720 | mg/L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 31 | GGT | 30730 | U/L | 连续 | z-score | MI(<15%) | SDOH(M only) | |
| 32 | Glucose | 30740 | mmol/L | 连续 | z-score | MI(<15%) | SDOH(M+F), Pheno | |
| 33 | HbA1c | 30750 | mmol/mol | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 34 | IGF1 | 30770 | nmol/L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 35 | LDL_direct | 30780 | mmol/L | 连续 | z-score | MI(<15%) | SDOH(M only) | |
| 36 | Phosphate | 30810 | mmol/L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 37 | SHBG | 30830 | nmol/L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 38 | Total_bilirubin | 30840 | μmol/L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 39 | Testosterone | 30850 | nmol/L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 40 | Total_protein | 30860 | g/L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |
| 41 | Triglycerides | 30870 | mmol/L | 连续 | z-score | MI(<15%) | SDOH(M only) | |
| 42 | VitaminD | 30890 | nmol/L | 连续 | z-score | MI(<15%) | SDOH(M+F) | |

> 体测/生化合计：**42** 个 (SDOHAge) + 1 个 PhenoAge 新增 (Mean sphered cell volume) = **43** 个。

---

## 2. SDOH（10 个，全部直接入模，不做 Boruta/RFE）

| # | 字段名 | UKB field id | 单位/类型 | 编码 | 变换 | 缺失处理 | 备注 |
|---|---|---|---|---|---|---|---|
| 43 | Townsend_deprivation_index | 22189 | index, 连续 | 连续 | z-score | MI(<15%) | 越高越贫困 |
| 44 | NO2_2007 | 24018 | μg/m³, 连续 | 连续 | z-score | MI(<15%) | 24016/17/18 中按论文取 2007 |
| 45 | PM10_2010 | 24005 | μg/m³, 连续 | 连续 | z-score | MI(<15%) | |
| 46 | PM25_2010 | 24006 | μg/m³, 连续 | 连续 | z-score | MI(<15%) | |
| 47 | Daytime_noise | 24020 | dB, 连续 | 连续 | z-score | MI(<15%) | |
| 48 | Nighttime_noise | 24022 | dB, 连续 | 连续 | z-score | MI(<15%) | |
| 49 | Education | 6138 | 多选数组 | 1=Degree, 2=A levels, 3=O levels/GCSE, 4=CSE, 5=NVQ/HND/HNC, 6=Other prof, -7=None, -3=NA | **先折叠为 6 个互斥等级（论文未明示规则，常用做法 = 取最高资格），再 z-score**。-7 → 单独一档（=最低）；-3 → 当缺失 | MI(<15%) | 论文遗留模糊点 |
| 50 | Household_income | 738 | 单选有序 | 1=<18k; 2=18-31k; 3=31-52k; 4=52-100k; 5=>100k; -1/-3=NA | 视为有序整数 → z-score（论文未明确是否 one-hot） | MI(<15%) | -1/-3 当缺失 |
| 51 | Friend_family_visits | 1031 | 单选有序 | 1=Almost daily … 6=Never; 7=No friends/family outside; -1/-3=NA | **先翻转方向使大值=社会接触多**（与 2110 一致），再 z-score；7 通常合并为最差档或单列 | MI(<15%) | 方向问题见 Ambiguities |
| 52 | Able_to_confide | 2110 | 单选有序 | 5=Almost daily … 0=Never; -1/-3=NA | 视为有序整数 → z-score | MI(<15%) | 大值=社会支持高 |

---

## 3. 仅 PhenoAge 新增

| # | 字段名 | UKB field id | 单位 | 变换 | 缺失处理 | 备注 |
|---|---|---|---|---|---|---|
| 53 | Chronological_age | 21022 | years | 直接进入 PhenoAge 公式（系数 0.0804）；ML 模型中也作为输入 | 通常无缺失 | 必须包含；最终 PhenoAge_acceleration = PhenoAge 对 age 取残差 |

---

## 4. 复现风险清单（必须自己拍板的项）

1. **Education (6138) 折叠规则**：原文未给。默认按"最高资格" → {Degree, A-level, O-level/GCSE, CSE, NVQ/HND/HNC, Other prof, None}。
2. **类别变量编码**：原文只说"标准化"，未说 one-hot。建议有序变量 → 整数 → z-score；如果想严格区分类别，再加一组 one-hot 做对照实验。
3. **多重插补算法**：原文未指定。推荐 `sklearn.experimental.IterativeImputer`（MICE）或 `missForest`，按性别分别插补更稳。
4. **NO2 字段年份**：24016(2005)/24017(2006)/24018(2007) 三选一，论文按 EU-wide average 暗指 24018，需到 UKB 与补充表 6 描述统计核对。
5. **CRP 单位**：UKB 是 mg/L；PhenoAge 原始公式定义是 mg/dL。如果完全复刻 PhenoAge 公式，需要先 ÷10 再 ln。如果只把 lncrp 当 ML 特征，二者只差常数，不影响树模型与神经网。
6. **1031 方向翻转**：建议训练前统一为 "数值越大社会越融入"，与 2110 方向一致。
7. **PhenoAge 与 SDOHAge 的 MCV 定义不同**：保留 30040（SDOHAge 用）和 30270（PhenoAge 用）两个独立特征，不要合并。

---

## 5. 一行汇总（可直接拷进代码做特征列表）

```python
UKB_FIELDS = [
    21001, 4079, 4080, 3063,
    30000, 30010, 30020, 30040, 30070, 30080, 30090, 30140,
    30180, 30190, 30270,
    30510, 30520, 30530,
    30600, 30610, 30620, 30630, 30650, 30660, 30670, 30680, 30690,
    30700, 30710, 30720, 30730, 30740, 30750, 30770, 30780,
    30810, 30830, 30840, 30850, 30860, 30870, 30890,
    22189, 24018, 24005, 24006, 24020, 24022,
    6138, 738, 1031, 2110,
    21022,
]
assert len(UKB_FIELDS) == 53
```
