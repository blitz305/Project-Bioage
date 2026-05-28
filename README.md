# Bioage — UK Biobank 生物学年龄建模流水线

本仓库实现一套基于 UK Biobank 的生物学年龄（BioAge）建模与评估流水线，包含两条主线：

- **fit-to-age**：以实足年龄为监督目标，回归出生物学年龄，残差即 `BioAge_acceleration`。
- **fit-to-death**：以生存结局为目标的 Cox 模型（EN-Cox + XGB-Cox），经 Gompertz 反解为年龄单位。

> 本 README 只介绍**仓库结构**。53 个特征的口径见 `feature_dictionary.md`。

---

## 仓库结构

```
Bioage/
│
├── README.md                       本文件
├── .gitignore                      数据 / 模型 / 结果一律不上传
├── environment_minimal.yml         conda 环境依赖
├── feature_dictionary.md           53 个特征字典：UKB id / 单位 / 变换 / 缺失策略
│
├── src/                            ★ 全部源码（仓库唯一上传的代码目录）
│   │
│   │  — 数据准备 —
│   ├── preprocess.py               原始 CSV → features_raw.parquet
│   ├── make_split.py               80/20 分层切分 → split.parquet
│   ├── impute_v2.py                miceforest（LightGBM）单次插补，按折独立
│   ├── build_outcomes.py           构建 4 个结局：death / CVD / T2D / cancer
│   ├── phenoage_formula.py         PhenoAge（Levine 2018）固定权重基线
│   │
│   │  — 特征 / 配置 / 公共库 —
│   ├── feature_modules.py          模块特征定义：metabolic/liver/kidney/immune/all
│   ├── config.yaml                 全流程超参与路径配置
│   ├── v2_lib.py                   共享工具：切分加载、建模、指标、日志
│   │
│   │  — fit-to-age 主线 —
│   ├── tune.py                     XGB / DNN 的 Optuna 调参（评分=5-fold R²）
│   ├── train_v2.py                 训练 lr / xgb / dnn / en，产 OOF + test 预测
│   │
│   │  — 横断面分析 —
│   ├── spearman_report.py          特征-年龄 Spearman + BH-FDR 预筛报告
│   ├── run_gam.py                  GAM 轨迹 Python 包装（调用下方 R 脚本）
│   ├── gam_trajectories.R          mgcv::gam 年龄轨迹拟合 + 出图
│   │
│   │  — fit-to-death 主线（Cox）—
│   ├── tune_cox.py                 XGB-Cox 的 Optuna 调参（评分=C-index）
│   ├── train_cox.py                EN-Cox / XGB-Cox 训练 + Gompertz 反解
│   ├── gompertz.py                 Gompertz 死亡率反解工具（风险→年龄单位）
│   │
│   │  — 评估 —
│   ├── survival_v2.py              Cox 生存分析：HR per SD / C-index / 森林图
│   ├── model_eval_v2.py            横向评估主脚本：master 表 + 多张评估图
│   │
│   │  — 编排 / 提交 —
│   ├── run_all_v2.sh               v2 全流程一键编排
│   ├── run_cox_v24.sh              v2.4 Cox 增量编排（不重跑 fit-age）
│   └── submit_v2.sh                集群 SLURM 提交脚本（本地，不上传）
│
└── （以下目录均在 .gitignore 中，留本地不上传）
    ├── data/                       中间产物：features_raw / imputed / outcomes / split (parquet, joblib)
    ├── outputs/                    模型 / 指标 / 图：v1_pilot/、v2/、master 评估表
    ├── 指标csv文件/                UKB 原始 CSV（受控数据）
    ├── 指导文件/                   论文调研整理 + 医院数据（受控）
    ├── UKB数据结果/                UKB 模型结果备份
    └── 粤北医院数据结果/           医院模型结果备份
```

---

## 环境

```bash
conda env create -f environment_minimal.yml
conda activate bioage
```

GAM 部分另需 R + mgcv（`r-base` + `r-mgcv` + `r-arrow`，通过 conda-forge 安装）。

---

## 运行入口

| 命令 | 用途 |
|---|---|
| `cd src && TOTAL_CORES=8 bash run_all_v2.sh` | fit-to-age 全流程（切分→调参→训练→评估） |
| `cd src && TOTAL_CORES=8 bash run_cox_v24.sh` | fit-to-death Cox 增量（EN-Cox + XGB-Cox） |

> 数据、模型、结果均不随仓库分发。需先准备好 UKB 原始数据并按 `src/config.yaml` 中的路径放置，再运行流水线生成 `data/` 与 `outputs/`。
