"""
gompertz.py — v2.4：Gompertz 反解工具。

把 Cox 模型输出的 linear predictor / 累积风险，翻译成"年龄单位"——
这是 PhenoAge (Levine 2018) 表面看是年龄、底层是死亡风险的关键步骤。

参考人群：UKB train 区按性别独立拟合
    mortality_rate(age) = a * exp(b * age)
观测死亡率按 5 岁分箱算 events/person-year。
拟合 log(rate) ~ age 的线性回归得到 (log_a, b)。

反解公式（参考 Levine 2018 Methods §2.4 / Liu 2018 PhenoAge eMethods）：
    给定 Cox linear predictor η（标准化后，均值 0），
    个体 baseline cumulative hazard at ref_age = H_ref（拟合时算）
    个体 10y 死亡概率 = 1 - exp(-H_ref * exp(η))
    反 Gompertz：bioage = ref_age + ln(M_indiv / M_ref) / b
    其中 M_ref 是 ref_age 平均人的 10y 死亡概率。

无 CLI，供 train_cox.py 调用。

输出：outputs/v2/cox/gompertz_params_{sex}.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_gompertz(
    ages: np.ndarray,
    events: np.ndarray,
    follow_years: np.ndarray,
    *,
    bin_width: float = 5.0,
    age_min: float = 40.0,
    age_max: float = 73.0,
) -> dict:
    """按 bin_width 岁分箱算观测死亡率，拟合 log(rate) ~ age。

    参数：
      ages: 每人基线年龄
      events: 每人死亡 event 0/1
      follow_years: 每人随访年数（事件时间或截尾时间）
      bin_width: 分箱宽度（岁）
      age_min/age_max: 用作拟合的年龄范围

    返回：
      dict {a, b, log_a, fit_r2, n_bins, bins: [...]}
      其中 mortality_rate(age) = a * exp(b * age)
    """
    edges = np.arange(age_min, age_max + bin_width, bin_width)
    centers = []
    rates = []
    bin_log = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (ages >= lo) & (ages < hi)
        if mask.sum() < 100:
            continue
        person_years = float(follow_years[mask].sum())
        n_deaths = int(events[mask].sum())
        if person_years <= 0 or n_deaths < 5:
            continue
        rate = n_deaths / person_years
        centers.append((lo + hi) / 2.0)
        rates.append(rate)
        bin_log.append({"age_lo": float(lo), "age_hi": float(hi),
                        "n": int(mask.sum()), "deaths": n_deaths,
                        "person_years": person_years, "rate": rate})

    if len(centers) < 3:
        raise RuntimeError(
            f"Gompertz 拟合的可用 bin 数 < 3（age_min={age_min}, age_max={age_max}, "
            f"bin_width={bin_width}），样本不够")

    x = np.array(centers, dtype=np.float64)
    y_log = np.log(np.array(rates, dtype=np.float64))
    lr = LinearRegression().fit(x.reshape(-1, 1), y_log)
    log_a = float(lr.intercept_)
    b = float(lr.coef_[0])
    a = float(np.exp(log_a))
    fit_r2 = float(lr.score(x.reshape(-1, 1), y_log))

    return {
        "a": a,
        "b": b,
        "log_a": log_a,
        "fit_r2": fit_r2,
        "n_bins": len(centers),
        "bins": bin_log,
    }


def mortality_rate(age: np.ndarray, a: float, b: float) -> np.ndarray:
    """Gompertz 死亡率：a * exp(b * age)"""
    return a * np.exp(b * np.asarray(age))


def cumulative_hazard(age: np.ndarray, *, a: float, b: float,
                      horizon: float = 10.0) -> np.ndarray:
    """从 age 起 horizon 年累积风险（Gompertz 积分）：
        H = (a / b) * exp(b * age) * (exp(b * horizon) - 1)
    """
    age = np.asarray(age, dtype=np.float64)
    return (a / b) * np.exp(b * age) * (np.exp(b * horizon) - 1.0)


def death_prob_horizon(age: np.ndarray, *, a: float, b: float,
                       horizon: float = 10.0) -> np.ndarray:
    """从 age 起 horizon 年死亡概率：1 - exp(-H)"""
    H = cumulative_hazard(age, a=a, b=b, horizon=horizon)
    return 1.0 - np.exp(-H)


def reverse_solve(
    risk_score: np.ndarray,
    *,
    a: float,
    b: float,
    ref_age: float = 60.0,
    horizon: float = 10.0,
    clip_age: tuple[float, float] = (20.0, 110.0),
) -> np.ndarray:
    """把 Cox linear predictor (η) 翻译成"年龄"单位。

    步骤：
      1. ref_age 的基础累积风险 H_ref = (a/b) * exp(b*ref_age) * (exp(b*horizon)-1)
      2. 个体 horizon 年死亡概率 M_i = 1 - exp(-H_ref * exp(η_i))
         （PhenoAge 假设：个体相对参考的风险倍数 = exp(η)）
      3. 反 Gompertz：solve M_i = 1 - exp(-(a/b)*exp(b*bioage)*(exp(b*horizon)-1))
         得：bioage = ln(-ln(1-M_i) * b / (a*(exp(b*horizon)-1))) / b

    参数：
      risk_score: 标准化的 Cox 线性预测器（z-score 后），均值≈0
      ref_age: 锚点参考年龄（默认 60）
      horizon: 时间窗口（默认 10 年）

    返回：bioage（np.ndarray，岁），裁剪到 clip_age 范围
    """
    eta = np.asarray(risk_score, dtype=np.float64)
    H_ref = (a / b) * np.exp(b * ref_age) * (np.exp(b * horizon) - 1.0)
    H_indiv = H_ref * np.exp(eta)
    # 反 Gompertz
    # H_indiv = (a/b) * exp(b*bioage) * (exp(b*horizon) - 1)
    # => exp(b*bioage) = H_indiv * b / (a * (exp(b*horizon) - 1))
    # => bioage = ln(H_indiv * b / (a * (exp(b*horizon)-1))) / b
    denom = a * (np.exp(b * horizon) - 1.0)
    bioage = np.log(np.clip(H_indiv * b / denom, 1e-12, None)) / b
    return np.clip(bioage, clip_age[0], clip_age[1])


def save_params(params: dict, sex_label: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"gompertz_params_{sex_label}.json"
    payload = {**params, "sex": sex_label}
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fp


def load_params(sex_label: str, out_dir: Path) -> dict:
    fp = out_dir / f"gompertz_params_{sex_label}.json"
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def fit_and_save_from_train(
    df_train: pd.DataFrame,
    outcomes: pd.DataFrame,
    sex_label: str,
    out_dir: Path,
    *,
    bin_width: float = 5.0,
    age_min: float = 40.0,
    age_max: float = 73.0,
) -> dict:
    """便捷入口：从 train_oof DataFrame（含 eid, age）+ outcomes 拟合并落盘。"""
    keys = ["eid", "death_event", "death_time_years", "died_within_2yr"]
    join = df_train[["eid", "age"]].merge(outcomes[keys], on="eid", how="inner")
    join = join[~join["died_within_2yr"]].reset_index(drop=True)
    params = fit_gompertz(
        ages=join["age"].values.astype(np.float64),
        events=join["death_event"].values.astype(np.float64),
        follow_years=join["death_time_years"].values.astype(np.float64),
        bin_width=bin_width,
        age_min=age_min,
        age_max=age_max,
    )
    params["n_train"] = int(len(join))
    params["ref_age"] = 60.0
    fp = save_params(params, sex_label, out_dir)
    print(f"[gompertz/{sex_label}] a={params['a']:.4g}  b={params['b']:.4f}  "
          f"fit_r2={params['fit_r2']:.4f}  n_bins={params['n_bins']}  -> {fp}")
    return params
