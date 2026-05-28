"""
survival_v2.py — v2 双轨生存分析 + 3-way C-index + ΔC-index。

依赖（按 model_name = xgb / dnn / lr）:
  outputs/v2/{sex}_{model}_train_oof.parquet
  outputs/v2/{sex}_{model}_test_pred.parquet
  data/phenoage_formula_{sex}.parquet
  data/outcomes.parquet
  data/split.parquet

CLI:
  python survival_v2.py --model xgb

输出 (outputs/v2/):
  survival_train_summary.md      train OOF 上跑的 Cox（探索）
  survival_test_summary.md       test 上跑的 Cox（★ 最终对外指标）
  cindex_table.csv               3-way + ΔC（age-only / PhenoAge / ML / 全） × train/test
  forest_test.png                test 上的 forest plot
  km_test_{sex}_{outcome}.png    test 上的 KM
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from feature_modules import ALL_KEY, MODULE_KEYS, output_suffix
from v2_lib import SEX_FEMALE, SEX_MALE, load_config, load_split


OUTCOMES = ["death", "cvd", "t2d", "cancer"]
SEXES = [("male", SEX_MALE), ("female", SEX_FEMALE)]


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def fit_cox(sub: pd.DataFrame, duration: str, event: str, formula: str) -> dict:
    cph = CoxPHFitter()
    cph.fit(sub, duration_col=duration, event_col=event, formula=formula,
            robust=False, show_progress=False)
    summ = cph.summary
    out = {"_concordance": float(cph.concordance_index_),
           "_n": int(len(sub)), "_events": int(sub[event].sum())}
    for var in summ.index:
        out[var] = {
            "HR": float(summ.loc[var, "exp(coef)"]),
            "HR_lower": float(summ.loc[var, "exp(coef) lower 95%"]),
            "HR_upper": float(summ.loc[var, "exp(coef) upper 95%"]),
            "p": float(summ.loc[var, "p"]),
        }
    return out


def cindex_only(sub: pd.DataFrame, duration: str, event: str, formula: str) -> float:
    cph = CoxPHFitter()
    cph.fit(sub, duration_col=duration, event_col=event, formula=formula,
            robust=False, show_progress=False)
    return float(cph.concordance_index_)


def km_quartile_plot(sub: pd.DataFrame, outcome: str, sex_label: str, out_dir: Path) -> None:
    dur = f"{outcome}_time_years"
    evt = f"{outcome}_event"
    try:
        q = pd.qcut(sub["bioage_z"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    for label in ["Q1", "Q2", "Q3", "Q4"]:
        mask = q == label
        if mask.sum() < 50:
            continue
        kmf.fit(sub.loc[mask, dur], sub.loc[mask, evt], label=str(label))
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title(f"KM by BioAge_z quartile — {sex_label} / {outcome} (TEST)")
    ax.set_xlabel("Years from baseline")
    ax.set_ylabel("Survival prob")
    plt.tight_layout()
    fp = out_dir / f"km_test_{sex_label}_{outcome}.png"
    fig.savefig(fp, dpi=120)
    plt.close(fig)


def forest_plot(rows: list[dict], out_dir: Path, fname: str = "forest_test.png") -> None:
    df = pd.DataFrame(rows)
    df = df[(df["model_kind"] == "m1_bioage") & (df["term"] == "bioage_z")].reset_index(drop=True)
    if df.empty:
        return
    df["label"] = df["sex"] + " / " + df["outcome"]
    df = df.sort_values(["outcome", "sex"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(df))))
    y = np.arange(len(df))
    ax.errorbar(df["HR"], y,
                xerr=[df["HR"] - df["HR_lower"], df["HR_upper"] - df["HR"]],
                fmt="o", color="steelblue", ecolor="gray", capsize=3)
    ax.axvline(1.0, color="red", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("HR per SD BioAge_accel (95% CI)  — test set")
    ax.set_title("BioAge acceleration → outcomes (test)")
    ax.invert_yaxis()
    plt.tight_layout()
    fp = out_dir / fname
    fig.savefig(fp, dpi=120)
    plt.close(fig)


def write_cox_summary(rows: list[dict], n_info: dict, out_path: Path, title: str) -> None:
    lines = [f"# {title}", ""]
    lines.append("## 队列信息")
    for sex, info in n_info.items():
        lines.append(f"- **{sex}**: n_total={info['n_total']:,}, "
                     f"剔除 ≤2 年死亡={info['drop_2yr']:,}, "
                     f"final n={info['n_final']:,}")
    lines.append("")

    def section(model_kind: str, term: str, header: str):
        lines.append(f"## {header}")
        lines.append("| Sex | Outcome | n | events | HR | 95% CI | p | C-index |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["model_kind"] != model_kind or r["term"] != term:
                continue
            lines.append(
                f"| {r['sex']} | {r['outcome']} | {r['n']:,} | {r['events']:,} | "
                f"{r['HR']:.3f} | ({r['HR_lower']:.3f}, {r['HR_upper']:.3f}) | "
                f"{r['p']:.2e} | {r['concordance']:.3f} |"
            )
        lines.append("")

    section("m1_bioage", "bioage_z", "Model 1: bioage_z + age")
    section("m2_phenoage", "phenoage_z", "Model 2: phenoage_z + age")
    lines.append("## Model 3: bioage_z + phenoage_z + age")
    lines.append("| Sex | Outcome | term | HR | 95% CI | p |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["model_kind"] != "m3_both" or r["term"] not in ("bioage_z", "phenoage_z"):
            continue
        lines.append(
            f"| {r['sex']} | {r['outcome']} | {r['term']} | {r['HR']:.3f} | "
            f"({r['HR_lower']:.3f}, {r['HR_upper']:.3f}) | {r['p']:.2e} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def cox_one_outcome(sub: pd.DataFrame, outcome: str) -> dict:
    dur = f"{outcome}_time_years"
    evt = f"{outcome}_event"
    res = {}
    res["m1_bioage"] = fit_cox(sub, dur, evt, "bioage_z + age")
    res["m2_phenoage"] = fit_cox(sub, dur, evt, "phenoage_z + age")
    res["m3_both"] = fit_cox(sub, dur, evt, "bioage_z + phenoage_z + age")
    return res


def cindex_three_way(sub: pd.DataFrame, outcome: str) -> dict:
    """C-index 4 个对照：age only / PhenoAge+age / BioAge+age / both+age。"""
    dur = f"{outcome}_time_years"
    evt = f"{outcome}_event"
    return {
        "C_age": cindex_only(sub, dur, evt, "age"),
        "C_phenoage_age": cindex_only(sub, dur, evt, "phenoage_z + age"),
        "C_bioage_age": cindex_only(sub, dur, evt, "bioage_z + age"),
        "C_both_age": cindex_only(sub, dur, evt, "bioage_z + phenoage_z + age"),
    }


def collect_rows_from_cox(res_dict: dict, sex_label: str, outcome: str,
                          n: int, ev: int) -> list[dict]:
    rows = []
    for model_kind, m_res in res_dict.items():
        cidx = m_res["_concordance"]
        for term, v in m_res.items():
            if term.startswith("_"):
                continue
            rows.append({
                "sex": sex_label, "outcome": outcome,
                "model_kind": model_kind, "term": term,
                "HR": v["HR"], "HR_lower": v["HR_lower"], "HR_upper": v["HR_upper"],
                "p": v["p"], "n": n, "events": ev, "concordance": cidx,
            })
    return rows


def assemble_dataset(pred_df: pd.DataFrame, ph_df: pd.DataFrame,
                     outcomes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = pred_df.merge(
        ph_df[["eid", "PhenoAge_accel"]], on="eid"
    ).merge(outcomes, on="eid")
    n_total = len(df)
    keep = ~df["died_within_2yr"]
    n_drop = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    df["bioage_z"] = zscore(df["BioAge_acceleration"])
    df["phenoage_z"] = zscore(df["PhenoAge_accel"])
    info = {"n_total": n_total, "drop_2yr": n_drop, "n_final": len(df)}
    return df, info


def run_track(pred_files: dict, label: str, *, data_dir: Path, out_dir: Path,
              outcomes: pd.DataFrame, ph: dict[str, pd.DataFrame],
              draw_km: bool) -> tuple[list[dict], dict, list[dict]]:
    """label='train' 或 'test'，pred_files = {sex_label: parquet_path}。"""
    all_rows: list[dict] = []
    cindex_rows: list[dict] = []
    n_info: dict = {}

    for sex_label, _ in SEXES:
        if sex_label not in pred_files:
            continue
        pred_df = pd.read_parquet(pred_files[sex_label])
        df, info = assemble_dataset(pred_df, ph[sex_label], outcomes)
        n_info[sex_label] = info
        print(f"[{label}:{sex_label}] n_total={info['n_total']:,}  "
              f"drop_2yr={info['drop_2yr']:,}  final={info['n_final']:,}")

        for outcome in OUTCOMES:
            sub = df.copy()
            if outcome != "death":
                sub = sub[~sub[f"{outcome}_prevalent"]].reset_index(drop=True)
            n = len(sub)
            ev = int(sub[f"{outcome}_event"].sum())
            print(f"  [{label}/{sex_label}/{outcome}] n={n:,}  events={ev:,}")
            if ev < 30:
                print(f"    events 太少，跳过")
                continue
            cox_input = sub[["bioage_z", "phenoage_z", "age",
                             f"{outcome}_time_years", f"{outcome}_event"]]
            try:
                res = cox_one_outcome(cox_input, outcome)
                all_rows += collect_rows_from_cox(res, sex_label, outcome, n, ev)
                cindex = cindex_three_way(cox_input, outcome)
                cindex_rows.append({
                    "track": label, "sex": sex_label, "outcome": outcome,
                    "n": n, "events": ev,
                    **cindex,
                    "deltaC_bioage_vs_age": cindex["C_bioage_age"] - cindex["C_age"],
                    "deltaC_both_vs_phenoage": cindex["C_both_age"] - cindex["C_phenoage_age"],
                })
                if draw_km and label == "test":
                    km_quartile_plot(sub, outcome, sex_label, out_dir)
            except Exception as e:
                print(f"    Cox 失败: {e}")
    return all_rows, n_info, cindex_rows


def parse_pred_filename(path: Path) -> tuple[str, str, str] | None:
    """从 '{sex}{_features}_{model}_test_pred.parquet' 解析 (sex, features, model)。
    features 缺省（综合）时返回 'all'。返回 None 表示无法解析。"""
    stem = path.stem
    if not stem.endswith("_test_pred"):
        return None
    parts = stem[:-len("_test_pred")].split("_")
    if len(parts) < 2:
        return None
    sex, model = parts[0], parts[-1]
    if sex not in ("female", "male"):
        return None
    if model not in ("lr", "en", "xgb", "dnn", "encox", "xgbcox"):
        return None
    features = "_".join(parts[1:-1]) if len(parts) > 2 else ALL_KEY
    if features not in MODULE_KEYS:
        return None
    return sex, features, model


def scan_pred_files(out_dir: Path) -> list[tuple[str, str, str, Path, Path]]:
    """扫描 outputs/v2/ 下所有 *_test_pred.parquet。返回 [(sex, features, model, train_oof, test_pred)]。"""
    items: list[tuple[str, str, str, Path, Path]] = []
    for te in sorted(out_dir.glob("*_test_pred.parquet")):
        parsed = parse_pred_filename(te)
        if parsed is None:
            continue
        sex, features, model = parsed
        suffix = output_suffix(features)
        tr = out_dir / f"{sex}{suffix}_{model}_train_oof.parquet"
        if not tr.exists():
            print(f"  [scan] skip {te.name}: 缺对应 train_oof")
            continue
        items.append((sex, features, model, tr, te))
    return items


def cox_loop_all(test_files_by_combo: list[tuple[str, str, str, Path]],
                 ph: dict, outcomes: pd.DataFrame,
                 out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对所有 (sex, features, model) 组合跑 Cox + 4-way C-index。
    输出：
      cindex_all_models.csv (长表)：sex, features, model, outcome, n, events, C_age,
        C_phenoage_age, C_bioage_age, C_both_age, deltaC_bioage_vs_age, deltaC_both_vs_phenoage
      hr_all_models.csv (长表)：sex, features, model, outcome, term, HR, HR_lower, HR_upper, p
    """
    cindex_rows: list[dict] = []
    hr_rows: list[dict] = []

    for sex, features, model, te_path in test_files_by_combo:
        if sex not in ph:
            print(f"  [skip] {sex}/{features}/{model}: 缺 phenoage_formula_{sex}.parquet")
            continue
        pred_df = pd.read_parquet(te_path)
        df, info = assemble_dataset(pred_df, ph[sex], outcomes)
        print(f"[loop] {sex}/{features}/{model}: n={info['n_final']:,}")

        for outcome in OUTCOMES:
            sub = df.copy()
            if outcome != "death":
                sub = sub[~sub[f"{outcome}_prevalent"]].reset_index(drop=True)
            n = len(sub)
            ev = int(sub[f"{outcome}_event"].sum())
            if ev < 30:
                continue
            cox_input = sub[["bioage_z", "phenoage_z", "age",
                             f"{outcome}_time_years", f"{outcome}_event"]]
            try:
                cidx = cindex_three_way(cox_input, outcome)
                cindex_rows.append({
                    "sex": sex, "features": features, "model": model,
                    "outcome": outcome, "n": n, "events": ev,
                    **cidx,
                    "deltaC_bioage_vs_age": cidx["C_bioage_age"] - cidx["C_age"],
                    "deltaC_both_vs_phenoage": cidx["C_both_age"] - cidx["C_phenoage_age"],
                })
                res = cox_one_outcome(cox_input, outcome)
                for model_kind, m_res in res.items():
                    c_m = m_res["_concordance"]
                    for term, v in m_res.items():
                        if term.startswith("_"):
                            continue
                        hr_rows.append({
                            "sex": sex, "features": features, "model": model,
                            "outcome": outcome, "cox_spec": model_kind, "term": term,
                            "HR": v["HR"], "HR_lower": v["HR_lower"],
                            "HR_upper": v["HR_upper"], "p": v["p"],
                            "C_index": c_m,
                            "n": n, "events": ev,
                        })
            except Exception as e:
                print(f"  [warn] {sex}/{features}/{model}/{outcome} Cox 失败: {e}")

    cidx_df = pd.DataFrame(cindex_rows)
    hr_df = pd.DataFrame(hr_rows)
    cidx_df.to_csv(out_dir / "cindex_all_models.csv", index=False)
    hr_df.to_csv(out_dir / "hr_all_models.csv", index=False)
    print(f"[loop] -> cindex_all_models.csv  ({len(cidx_df)} rows)")
    print(f"[loop] -> hr_all_models.csv  ({len(hr_df)} rows)")
    return cidx_df, hr_df


def run_loop_all(config_path: Path) -> None:
    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    out_dir = (base / cfg["paths"]["outputs_dir"]).resolve() / "v2"

    outcomes = pd.read_parquet(data_dir / "outcomes.parquet")
    ph: dict = {}
    for sex_label, _ in SEXES:
        p = data_dir / f"phenoage_formula_{sex_label}.parquet"
        if p.exists():
            ph[sex_label] = pd.read_parquet(p)

    items = scan_pred_files(out_dir)
    print(f"[loop-all] 扫到 {len(items)} 个 (sex, features, model) 组合")
    for sex, features, model, _, _ in items:
        print(f"  {sex}/{features}/{model}")
    test_files_by_combo = [(s, f, m, te) for s, f, m, _, te in items]
    cox_loop_all(test_files_by_combo, ph, outcomes, out_dir)


def run(config_path: Path, model_name: str, features_arg: str) -> None:
    cfg = load_config(config_path)
    base = config_path.parent
    data_dir = (base / cfg["paths"]["data_dir"]).resolve()
    out_dir = (base / cfg["paths"]["outputs_dir"]).resolve() / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = output_suffix(features_arg)
    outcomes = pd.read_parquet(data_dir / "outcomes.parquet")
    print(f"[surv_v2] features={features_arg} model={model_name} | outcomes shape: {outcomes.shape}")

    ph = {}
    for sex_label, _ in SEXES:
        p = data_dir / f"phenoage_formula_{sex_label}.parquet"
        if p.exists():
            ph[sex_label] = pd.read_parquet(p)

    train_files, test_files = {}, {}
    for sex_label, _ in SEXES:
        tr = out_dir / f"{sex_label}{suffix}_{model_name}_train_oof.parquet"
        te = out_dir / f"{sex_label}{suffix}_{model_name}_test_pred.parquet"
        if tr.exists() and sex_label in ph:
            train_files[sex_label] = tr
        if te.exists() and sex_label in ph:
            test_files[sex_label] = te

    if not train_files and not test_files:
        print("[surv_v2] 找不到 train_oof / test_pred 文件，先跑 train_v2.py")
        return

    print("\n=== Track: TRAIN OOF (探索) ===")
    train_rows, train_n, train_cidx = run_track(
        train_files, "train",
        data_dir=data_dir, out_dir=out_dir,
        outcomes=outcomes, ph=ph, draw_km=False,
    )

    print("\n=== Track: TEST (★ 最终对外) ===")
    test_rows, test_n, test_cidx = run_track(
        test_files, "test",
        data_dir=data_dir, out_dir=out_dir,
        outcomes=outcomes, ph=ph, draw_km=True,
    )

    # —— 输出 ——
    if train_rows:
        write_cox_summary(train_rows, train_n,
                          out_dir / "survival_train_summary.md",
                          f"v2 Survival (TRAIN OOF / {model_name})")
        print(f"[surv_v2] -> survival_train_summary.md")

    if test_rows:
        write_cox_summary(test_rows, test_n,
                          out_dir / "survival_test_summary.md",
                          f"v2 Survival (TEST / {model_name}) ★ 最终对外指标")
        print(f"[surv_v2] -> survival_test_summary.md")
        forest_plot(test_rows, out_dir, fname="forest_test.png")
        print(f"[surv_v2] -> forest_test.png")

    cindex_all = train_cidx + test_cidx
    if cindex_all:
        cidx_df = pd.DataFrame(cindex_all)
        cidx_path = out_dir / "cindex_table.csv"
        cidx_df.to_csv(cidx_path, index=False)
        print(f"[surv_v2] -> {cidx_path}")

        # 长表 → markdown 摘要拼到 test summary 末尾
        if test_rows:
            md = ["", "## 3-way C-index (TEST)",
                  "| sex | outcome | C(age) | C(PhenoAge+age) | C(BioAge+age) | C(both+age) | ΔC(bioage−age) | ΔC(both−phenoage) |",
                  "|---|---|---|---|---|---|---|---|"]
            for r in test_cidx:
                md.append(
                    f"| {r['sex']} | {r['outcome']} | {r['C_age']:.4f} | "
                    f"{r['C_phenoage_age']:.4f} | {r['C_bioage_age']:.4f} | "
                    f"{r['C_both_age']:.4f} | {r['deltaC_bioage_vs_age']:+.4f} | "
                    f"{r['deltaC_both_vs_phenoage']:+.4f} |"
                )
            sumf = out_dir / "survival_test_summary.md"
            with open(sumf, "a", encoding="utf-8") as f:
                f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--model", default="xgb",
                    help="哪一个模型的 OOF / test_pred 用作 BioAge 来源")
    ap.add_argument("--features", choices=MODULE_KEYS, default=ALL_KEY,
                    help=f"特征集：{MODULE_KEYS}（默认 all = 综合，配合单模型模式）")
    ap.add_argument("--loop-all", action="store_true",
                    help="扫描 outputs/v2/ 下所有 (sex, features, model) 组合，遍历跑 Cox+C-index")
    args = ap.parse_args()
    if args.loop_all:
        run_loop_all(Path(args.config))
    else:
        run(Path(args.config), args.model, args.features)
