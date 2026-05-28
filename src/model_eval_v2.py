"""
model_eval_v2.py — v2.3 横向模型评估（22 个 sex × features × model 组合）。

分 5 层评估，对齐 BioAge 文献范式（PhenoAge / KDM-BA / MileAge / DBA-EnBA / MethylNet）：
  L1 回归内禀：MAE / RMSE / R² / Pearson r（OOF + TEST + gap）
  L2 校准：age 5 段 MAE + calibration slope（pred~age 斜率）
  L3 一致性：与综合 XGB pred 的 Pearson r
  L4 二分类临床效度：fast (top20%) vs slow (bot20%) ager → 4 outcome × 2 horizon 的 AUC/F1/Sens/Spec/PPV
  L5 生存（金标准）：从 survival_v2 --loop-all 产出的 cindex_all_models.csv / hr_all_models.csv join

依赖输入：
  outputs/v2/{sex}{_features}_{lr,en,xgb,dnn}_{train_oof,test_pred}.parquet
  outputs/v2/cindex_all_models.csv  （survival_v2.py --loop-all 已跑）
  outputs/v2/hr_all_models.csv
  data/outcomes.parquet

输出：
  outputs/v2/model_eval_master.csv          22 行 × 全指标
  outputs/v2/model_eval_L1_regression.csv
  outputs/v2/model_eval_L2_calibration.csv
  outputs/v2/model_eval_L3_agreement.csv    宽表（22 行 × 22 列，pred 间相关）
  outputs/v2/model_eval_L4_classification.csv  长表
  outputs/v2/eval_forest_HR_per_SD_death.png
  outputs/v2/eval_cindex_bar_4outcomes.png
  outputs/v2/eval_classification_auc.png
  outputs/v2/eval_module_agreement_heatmap.png
  outputs/v2/eval_calibration_scatter_per_model.png

CLI:
  python model_eval_v2.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)

from survival_v2 import scan_pred_files
from v2_lib import load_config, log, metrics_dict


OUTCOMES = ["death", "cvd", "t2d", "cancer"]
HORIZONS = [5, 10]
AGE_BINS = [(40, 50), (50, 55), (55, 60), (60, 65), (65, 73)]
REFERENCE_KEY = ("all", "xgb")  # L3 用综合 XGB 当参考
FIT_AGE_MODELS = {"lr", "en", "xgb", "dnn"}
FIT_DEATH_MODELS = {"encox", "xgbcox"}


def combo_key(sex: str, features: str, model: str) -> str:
    return f"{sex}_{features}_{model}"


def target_type_of(model: str) -> str:
    """根据 model 名称判断 target_type：fit-age vs fit-death。"""
    if model in FIT_DEATH_MODELS:
        return "death"
    return "age"


# ============================ L1: 回归内禀 ============================

def compute_L1(items: list[tuple]) -> pd.DataFrame:
    """L1 回归内禀：对 fit-death 模型仍计算（数字"不好看"但保持表结构一致）。"""
    rows = []
    for sex, features, model, tr_path, te_path in items:
        oof = pd.read_parquet(tr_path)
        te = pd.read_parquet(te_path)
        tt = target_type_of(model)
        if tt == "death":
            # cox 模型 R²/MAE 没意义（target 不是年龄），写 NaN 防误读
            rows.append({
                "sex": sex, "features": features, "model": model,
                "target_type": tt,
                "n_train": len(oof), "n_test": len(te),
                "OOF_MAE": np.nan, "OOF_RMSE": np.nan,
                "OOF_R2": np.nan, "OOF_pearson_r": np.nan,
                "TEST_MAE": np.nan, "TEST_RMSE": np.nan,
                "TEST_R2": np.nan, "TEST_pearson_r": np.nan,
                "OOF_TEST_R2_gap": np.nan,
            })
            continue
        oof_m = metrics_dict(oof["age"].values, oof["BioAge_pred"].values)
        te_m = metrics_dict(te["age"].values, te["BioAge_pred"].values)
        rows.append({
            "sex": sex, "features": features, "model": model,
            "target_type": tt,
            "n_train": len(oof), "n_test": len(te),
            "OOF_MAE": oof_m["MAE"], "OOF_RMSE": oof_m["RMSE"],
            "OOF_R2": oof_m["R2"], "OOF_pearson_r": oof_m["pearson_r"],
            "TEST_MAE": te_m["MAE"], "TEST_RMSE": te_m["RMSE"],
            "TEST_R2": te_m["R2"], "TEST_pearson_r": te_m["pearson_r"],
            "OOF_TEST_R2_gap": oof_m["R2"] - te_m["R2"],
        })
    return pd.DataFrame(rows)


# ============================ L2: 校准 ============================

def compute_L2(items: list[tuple]) -> pd.DataFrame:
    """L2 校准：fit-death 模型的 calibration_slope 仍能算（pred=bioage_cox vs age），
    数值低很正常（bioage_cox 不刻意贴 age）。分年龄段 MAE 对 cox 写 NaN。"""
    rows = []
    for sex, features, model, _, te_path in items:
        te = pd.read_parquet(te_path)
        age = te["age"].values.astype(np.float64)
        pred = te["BioAge_pred"].values.astype(np.float64)
        tt = target_type_of(model)
        row: dict = {"sex": sex, "features": features, "model": model,
                     "target_type": tt}
        for lo, hi in AGE_BINS:
            mask = (age >= lo) & (age < hi)
            key = f"MAE_{lo}_{hi}"
            if tt == "death":
                row[key] = np.nan
            else:
                row[key] = (float(np.mean(np.abs(pred[mask] - age[mask])))
                            if mask.sum() >= 30 else np.nan)
        # calibration slope: pred ~ age（理想斜率=1，<1 表示回归到均值）
        lr = LinearRegression().fit(age.reshape(-1, 1), pred)
        row["calibration_slope"] = float(lr.coef_[0])
        row["calibration_intercept"] = float(lr.intercept_)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================ L3: 模型间一致性 ============================

def compute_L3(items: list[tuple]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回：
      L3_long: 每个模型 vs 综合 XGB 的 Pearson r（一行一模型）
      L3_matrix: 22×22 模型间相关矩阵（每性别内部）"""
    # 读所有 test pred 到一个大 DataFrame（key=eid）
    sex_preds: dict[str, pd.DataFrame] = {"female": pd.DataFrame(), "male": pd.DataFrame()}
    for sex, features, model, _, te_path in items:
        col = combo_key(sex, features, model)
        te = pd.read_parquet(te_path)[["eid", "BioAge_pred"]].rename(
            columns={"BioAge_pred": col}
        )
        if sex_preds[sex].empty:
            sex_preds[sex] = te
        else:
            sex_preds[sex] = sex_preds[sex].merge(te, on="eid", how="outer")

    long_rows = []
    matrix_blocks = []
    for sex, df in sex_preds.items():
        if df.empty:
            continue
        pred_cols = [c for c in df.columns if c != "eid"]
        ref_col = combo_key(sex, *REFERENCE_KEY)
        if ref_col not in pred_cols:
            log(f"  [L3] {sex} 缺综合 XGB（{ref_col}），跳过 corr_with_all_xgb")
            ref_series = None
        else:
            ref_series = df[ref_col].values

        for col in pred_cols:
            parts = col.split("_")
            sx, mdl = parts[0], parts[-1]
            ft = "_".join(parts[1:-1]) if len(parts) > 2 else "all"
            if ref_series is None:
                r = np.nan
            else:
                v = df[col].values
                mask = ~(np.isnan(v) | np.isnan(ref_series))
                r = float(stats.pearsonr(v[mask], ref_series[mask])[0]) if mask.sum() > 10 else np.nan
            long_rows.append({"sex": sx, "features": ft, "model": mdl,
                              "corr_with_all_xgb": r})
        # 矩阵：只算这个性别内的相关方阵
        corr = df[pred_cols].corr(method="pearson")
        corr.index.name = "model_combo"
        corr_reset = corr.reset_index()
        corr_reset.insert(0, "sex_block", sex)
        matrix_blocks.append(corr_reset)

    long_df = pd.DataFrame(long_rows)
    matrix_df = pd.concat(matrix_blocks, ignore_index=True) if matrix_blocks else pd.DataFrame()
    return long_df, matrix_df


# ============================ L4: fast/slow ager 二分类 ============================

def compute_L4(items: list[tuple], outcomes: pd.DataFrame) -> pd.DataFrame:
    """对每个模型：accel top 20% vs bot 20% → 4 outcome × 2 horizon 二分类。

    对每个 outcome 独立切 top/bot 20%（在排除该 outcome prevalent 之后），
    避免 prevalent 过滤导致的索引错位。
    """
    rows = []
    for sex, features, model, _, te_path in items:
        te = pd.read_parquet(te_path)
        df = te.merge(outcomes, on="eid", how="inner")
        df = df[~df["died_within_2yr"]].reset_index(drop=True)
        if len(df) < 100:
            continue

        for outcome in OUTCOMES:
            sub = df.copy()
            if outcome != "death":
                sub = sub[~sub[f"{outcome}_prevalent"]].reset_index(drop=True)
            if len(sub) < 100:
                continue

            accel = sub["BioAge_acceleration"].values
            q_lo, q_hi = np.quantile(accel, [0.20, 0.80])
            fast_mask = accel >= q_hi
            slow_mask = accel <= q_lo
            keep_fs = fast_mask | slow_mask
            sub_fs = sub[keep_fs].reset_index(drop=True)
            sub_fs["fast_label"] = (sub_fs["BioAge_acceleration"].values >= q_hi).astype(int)

            for horizon in HORIZONS:
                t = sub_fs[f"{outcome}_time_years"].values
                e = sub_fs[f"{outcome}_event"].values.astype(int)
                pos = (e == 1) & (t <= horizon)
                neg_no_event = (e == 0) & (t >= horizon)
                neg_event_after = (e == 1) & (t > horizon)
                neg = neg_no_event | neg_event_after
                keep2 = pos | neg
                if keep2.sum() < 100 or pos[keep2].sum() < 20:
                    continue
                y_true = pos[keep2].astype(int)
                y_pred = sub_fs["fast_label"].values[keep2]
                score = sub_fs["BioAge_acceleration"].values[keep2]
                rows.append({
                    "sex": sex, "features": features, "model": model,
                    "outcome": outcome, "horizon_years": horizon,
                    "n": int(keep2.sum()),
                    "n_pos": int(y_true.sum()),
                    "AUC": float(roc_auc_score(y_true, score)),
                    "F1": float(f1_score(y_true, y_pred, zero_division=0)),
                    "Sens": float(recall_score(y_true, y_pred, zero_division=0)),
                    "Spec": float(recall_score(1 - y_true, 1 - y_pred, zero_division=0)),
                    "PPV": float(precision_score(y_true, y_pred, zero_division=0)),
                })
    return pd.DataFrame(rows)


# ============================ L5: 从 survival --loop-all join ============================

def load_L5(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cidx_path = out_dir / "cindex_all_models.csv"
    hr_path = out_dir / "hr_all_models.csv"
    if not cidx_path.exists() or not hr_path.exists():
        log(f"[L5] 缺 {cidx_path.name} / {hr_path.name}，先跑 `python survival_v2.py --loop-all`")
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(cidx_path), pd.read_csv(hr_path)


# ============================ Master 拼接 ============================

def build_master(L1: pd.DataFrame, L2: pd.DataFrame, L3: pd.DataFrame,
                 L4: pd.DataFrame, L5_cidx: pd.DataFrame, L5_hr: pd.DataFrame) -> pd.DataFrame:
    keys = ["sex", "features", "model"]
    # 先把 target_type 从 L1 提出来，merge 时只保留 L1 那份
    L2_use = L2.drop(columns=[c for c in ["target_type"] if c in L2.columns])
    df = L1.merge(L2_use, on=keys, how="outer")
    df = df.merge(L3, on=keys, how="outer")

    # L4 → 宽表：每 outcome × horizon 一列 AUC/F1/Sens/Spec/PPV
    if not L4.empty:
        L4_wide = L4.pivot_table(
            index=keys, columns=["outcome", "horizon_years"],
            values=["AUC", "F1", "Sens", "Spec"], aggfunc="first",
        )
        L4_wide.columns = [f"fast_{m}_{o}_{h}y" for m, o, h in L4_wide.columns]
        df = df.merge(L4_wide.reset_index(), on=keys, how="left")

    # L5 cindex → 宽表
    if not L5_cidx.empty:
        c_wide = L5_cidx.pivot_table(
            index=keys, columns="outcome",
            values=["C_age", "C_phenoage_age", "C_bioage_age",
                    "C_both_age", "deltaC_bioage_vs_age"], aggfunc="first",
        )
        c_wide.columns = [f"{m}_{o}" for m, o in c_wide.columns]
        df = df.merge(c_wide.reset_index(), on=keys, how="left")

    # L5 hr → 每 outcome 的 m1_bioage / bioage_z 的 HR
    if not L5_hr.empty:
        hr_bio = L5_hr[
            (L5_hr["cox_spec"] == "m1_bioage") & (L5_hr["term"] == "bioage_z")
        ].copy()
        hr_bio["HR_CI"] = hr_bio.apply(
            lambda r: f"{r['HR']:.3f} ({r['HR_lower']:.3f}-{r['HR_upper']:.3f})", axis=1
        )
        hr_wide = hr_bio.pivot_table(
            index=keys, columns="outcome",
            values=["HR", "HR_CI"], aggfunc="first",
        )
        hr_wide.columns = [f"HR_per_SD_{o}" if m == "HR" else f"HR_CI_{o}"
                           for m, o in hr_wide.columns]
        df = df.merge(hr_wide.reset_index(), on=keys, how="left")

    # 兜底：缺 target_type 列时按 model 名补
    if "target_type" not in df.columns:
        df["target_type"] = df["model"].map(target_type_of)
    else:
        df["target_type"] = df["target_type"].fillna(df["model"].map(target_type_of))
    # 列顺序：把 target_type 放到 model 后面
    cols = list(df.columns)
    if "target_type" in cols:
        cols.remove("target_type")
        ins = cols.index("model") + 1
        cols = cols[:ins] + ["target_type"] + cols[ins:]
        df = df[cols]
    df = df.sort_values(["sex", "features", "target_type", "model"]).reset_index(drop=True)
    return df


# ============================ 可视化 ============================


def plot_age_vs_death_target(master: pd.DataFrame, out_path: Path) -> None:
    """v2.4 新图：fit-age 模型 vs fit-death 模型在死亡 HR 上的对比。
    每个 (sex, features) 一列，列内 fit-age 多模型为蓝点，fit-death 多模型为红点。
    """
    if "HR_per_SD_death" not in master.columns:
        return
    sub = master.dropna(subset=["HR_per_SD_death"]).copy()
    if sub.empty:
        return
    sub["group"] = sub["sex"].str[:1] + "/" + sub["features"]
    groups = sorted(sub["group"].unique())
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(groups)), 6))
    for i, g in enumerate(groups):
        gd = sub[sub["group"] == g]
        for _, r in gd.iterrows():
            color = "darkred" if r["target_type"] == "death" else "steelblue"
            marker = "s" if r["target_type"] == "death" else "o"
            ax.scatter(i, r["HR_per_SD_death"], color=color, marker=marker,
                       s=60, alpha=0.85, edgecolors="black", linewidths=0.4)
            ax.annotate(r["model"], (i, r["HR_per_SD_death"]),
                        fontsize=6, xytext=(4, 0), textcoords="offset points",
                        va="center")
    ax.axhline(1.0, color="red", lw=0.5, ls="--")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("HR per SD BioAge_accel — all-cause death (TEST)")
    ax.set_title("fit-age (蓝圆) vs fit-death (红方) — HR per SD across 5 features × 2 sex")
    # 图例
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", mfc="steelblue",
               mec="black", markersize=8, label="fit-age (lr/en/xgb/dnn)"),
        Line2D([0], [0], marker="s", color="w", mfc="darkred",
               mec="black", markersize=8, label="fit-death (encox/xgbcox)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"  -> {out_path.name}")

def plot_forest_HR_death(master: pd.DataFrame, hr_long: pd.DataFrame,
                         out_path: Path) -> None:
    if hr_long.empty:
        return
    sub = hr_long[
        (hr_long["cox_spec"] == "m1_bioage")
        & (hr_long["term"] == "bioage_z")
        & (hr_long["outcome"] == "death")
    ].copy()
    if sub.empty:
        return
    sub["label"] = sub.apply(lambda r: f"{r['sex'][:1]}/{r['features']}/{r['model']}", axis=1)
    sub = sub.sort_values(["sex", "features", "model"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(sub))))
    y = np.arange(len(sub))
    colors = ["steelblue" if s == "female" else "darkorange" for s in sub["sex"]]
    ax.errorbar(sub["HR"], y,
                xerr=[sub["HR"] - sub["HR_lower"], sub["HR_upper"] - sub["HR"]],
                fmt="o", ecolor="gray", capsize=2, mfc="none")
    for i, c in enumerate(colors):
        ax.plot(sub["HR"].iloc[i], i, "o", color=c)
    ax.axvline(1.0, color="red", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=8)
    ax.set_xlabel("HR per SD BioAge_accel (95% CI) — TEST, all-cause death")
    ax.set_title("Forest plot — HR per SD across all 22 models")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log(f"  -> {out_path.name}")


def plot_cindex_bar(master: pd.DataFrame, out_path: Path) -> None:
    cols_b = [f"C_bioage_age_{o}" for o in OUTCOMES]
    cols_a = [f"C_age_{o}" for o in OUTCOMES]
    if not all(c in master.columns for c in cols_b):
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, outcome in zip(axes.flat, OUTCOMES):
        col_b = f"C_bioage_age_{outcome}"
        col_a = f"C_age_{outcome}"
        sub = master.dropna(subset=[col_b]).copy()
        sub = sub.sort_values(["sex", "features", "model"]).reset_index(drop=True)
        x = np.arange(len(sub))
        colors = ["steelblue" if s == "female" else "darkorange" for s in sub["sex"]]
        ax.bar(x, sub[col_b], color=colors, edgecolor="black", lw=0.3)
        # baseline line = C(age only) 平均
        baseline = sub[col_a].mean()
        ax.axhline(baseline, color="red", lw=0.8, ls="--",
                   label=f"avg C(age only)={baseline:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['sex'][:1]}/{r['features']}/{r['model']}" for _, r in sub.iterrows()],
            rotation=75, fontsize=7,
        )
        ax.set_ylim(0.5, max(0.85, sub[col_b].max() + 0.02))
        ax.set_ylabel("C-index (BioAge + age)")
        ax.set_title(f"{outcome}")
        ax.legend(fontsize=7)
    fig.suptitle("C-index by model — 4 outcomes (TEST)", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"  -> {out_path.name}")


def plot_classification_auc(L4: pd.DataFrame, out_path: Path) -> None:
    if L4.empty:
        return
    L4 = L4.copy()
    L4["model_combo"] = L4.apply(
        lambda r: f"{r['sex'][:1]}/{r['features']}/{r['model']}", axis=1
    )
    L4["col_label"] = L4["outcome"] + "_" + L4["horizon_years"].astype(str) + "y"
    piv = L4.pivot_table(index="model_combo", columns="col_label",
                         values="AUC", aggfunc="first")
    piv = piv.sort_index()
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * piv.shape[1] + 3),
                                    max(5, 0.32 * piv.shape[0])))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=0.85)
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    plt.colorbar(im, ax=ax, label="AUC")
    ax.set_title("Fast/slow ager binary AUC — outcome × horizon (TEST)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log(f"  -> {out_path.name}")


def plot_module_agreement(L3_matrix: pd.DataFrame, out_path: Path) -> None:
    if L3_matrix.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, sex in zip(axes, ["female", "male"]):
        sub = L3_matrix[L3_matrix["sex_block"] == sex].copy()
        if sub.empty:
            continue
        mat = sub.drop(columns=["sex_block", "model_combo"]).copy()
        mat.index = sub["model_combo"].values
        # 只保留这个性别下的列
        cols = [c for c in mat.columns if c.startswith(sex)]
        mat = mat.loc[mat.index.isin(cols), cols]
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=75, fontsize=7)
        ax.set_yticks(np.arange(len(cols)))
        ax.set_yticklabels(cols, fontsize=7)
        ax.set_title(f"{sex} | pred-pred Pearson r")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log(f"  -> {out_path.name}")


def plot_calibration_scatter(items: list[tuple], out_path: Path) -> None:
    n = len(items)
    n_col = 6
    n_row = (n + n_col - 1) // n_col
    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(2.4 * n_col, 2.1 * n_row))
    axes = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, (sex, features, model, _, te_path) in zip(axes, items):
        te = pd.read_parquet(te_path)
        idx = np.random.default_rng(0).choice(len(te), size=min(3000, len(te)), replace=False)
        age = te["age"].values[idx]
        pred = te["BioAge_pred"].values[idx]
        ax.scatter(age, pred, s=3, alpha=0.25, color="steelblue", edgecolors="none")
        lo, hi = min(age.min(), pred.min()), max(age.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.6)
        ax.set_title(f"{sex[:1]}/{features}/{model}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Calibration scatter — predicted vs chronological age (TEST)", y=1.0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"  -> {out_path.name}")


# ============================ Main ============================

def run(config_path: Path) -> None:
    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    out_dir = (base / cfg["paths"]["outputs_dir"]).resolve() / "v2"

    items = scan_pred_files(out_dir)
    log(f"[eval] 扫到 {len(items)} 个 (sex, features, model) 组合")
    if not items:
        return

    log("[eval] 算 L1 回归内禀 ...")
    L1 = compute_L1(items)
    L1.to_csv(out_dir / "model_eval_L1_regression.csv", index=False)

    log("[eval] 算 L2 校准 ...")
    L2 = compute_L2(items)
    L2.to_csv(out_dir / "model_eval_L2_calibration.csv", index=False)

    log("[eval] 算 L3 一致性 ...")
    L3_long, L3_matrix = compute_L3(items)
    L3_long.to_csv(out_dir / "model_eval_L3_agreement_long.csv", index=False)
    L3_matrix.to_csv(out_dir / "model_eval_L3_agreement.csv", index=False)

    log("[eval] 算 L4 fast/slow 二分类 ...")
    outcomes = pd.read_parquet(data_dir / "outcomes.parquet")
    L4 = compute_L4(items, outcomes)
    L4.to_csv(out_dir / "model_eval_L4_classification.csv", index=False)

    log("[eval] 读 L5 生存 ...")
    L5_cidx, L5_hr = load_L5(out_dir)

    log("[eval] 拼 master ...")
    master = build_master(L1, L2, L3_long, L4, L5_cidx, L5_hr)
    master.to_csv(out_dir / "model_eval_master.csv", index=False)
    log(f"  -> model_eval_master.csv  ({len(master)} 行)")

    log("[eval] 画图 ...")
    plot_forest_HR_death(master, L5_hr, out_dir / "eval_forest_HR_per_SD_death.png")
    plot_cindex_bar(master, out_dir / "eval_cindex_bar_4outcomes.png")
    plot_classification_auc(L4, out_dir / "eval_classification_auc.png")
    plot_module_agreement(L3_matrix, out_dir / "eval_module_agreement_heatmap.png")
    plot_calibration_scatter(items, out_dir / "eval_calibration_scatter_per_model.png")
    plot_age_vs_death_target(master, out_dir / "eval_age_vs_death_target.png")
    log("[eval] done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = ap.parse_args()
    run(Path(args.config))
