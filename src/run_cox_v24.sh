#!/usr/bin/env bash
# run_cox_v24.sh — 只跑 v2.4 增量：EN-Cox + XGB-Cox × 5 features × 2 sex
#
# 用法：
#   TOTAL_CORES=8  bash run_cox_v24.sh        # 本地 WSL
#   TOTAL_CORES=144 bash run_cox_v24.sh       # 集群
#
# 前提：v2.3 全量已跑完，依赖以下既有产物（read-only 复用）：
#   data/features_imputed_v2_{sex}.parquet
#   data/outcomes.parquet
#   data/split.parquet
#   outputs/v2/{22 个 fit-age 模型}_test_pred.parquet
#
# 编排：
#   Wave 8a: sksurv 自检
#   Wave 8b: EN-Cox 训练 × 10（全并发）
#   Wave 8c: XGB-Cox 调参 × 10（6 路并发，TPE 15 trials）
#   Wave 8d: XGB-Cox 训练 × 10（4 路并发）
#   Wave 8e: survival_v2 --loop-all + model_eval_v2（重写 master 至 42 行）

set -euo pipefail
cd "$(dirname "$0")"

TOTAL_CORES=${TOTAL_CORES:-8}
LOG_DIR="../outputs/v2/_logs"
COX_DIR="../outputs/v2/cox"
mkdir -p "$LOG_DIR" "$COX_DIR"

W_8B=$(( TOTAL_CORES / 10 )); [[ $W_8B -lt 1 ]] && W_8B=1
W_8C=$(( TOTAL_CORES / 6  )); [[ $W_8C -lt 1 ]] && W_8C=1
W_8D=$(( TOTAL_CORES / 4  )); [[ $W_8D -lt 1 ]] && W_8D=1

echo "=== v2.4 cox run | TOTAL_CORES=$TOTAL_CORES ==="
echo "    Wave 8b encox parallel=10×${W_8B}c"
echo "    Wave 8c xgbcox tune  parallel=6×${W_8C}c"
echo "    Wave 8d xgbcox train parallel=4×${W_8D}c"

run_bg() {
    local name="$1"; shift
    local threads="$1"; shift
    THREADS_PER_JOB="$threads" \
    OMP_NUM_THREADS="$threads" \
    OPENBLAS_NUM_THREADS="$threads" \
    MKL_NUM_THREADS="$threads" \
    LIGHTGBM_NUM_THREADS="$threads" \
        stdbuf -oL -eL "$@" > "$LOG_DIR/cox_${name}.log" 2>&1 &
    echo "  [pid $!] cox_${name} (threads=$threads)"
}

# ---------- Wave 8a: 自检 ----------
echo
echo "=== Wave 8a: sksurv check ==="
python -c "import sksurv; print('sksurv', sksurv.__version__)" \
    || { echo "ERROR: 缺 sksurv，先 conda install -c conda-forge scikit-survival"; exit 1; }

# ---------- Wave 8b: EN-Cox × 10 ----------
echo
echo "=== Wave 8b: EN-Cox train × 10 (skip existing) ==="
OUT_DIR="../outputs/v2"
launched=0
for FEAT in all metabolic liver kidney immune; do
    case "$FEAT" in
        all) SUFFIX="" ;;
        *)   SUFFIX="_${FEAT}" ;;
    esac
    for SX in female male; do
        DONE_FILE="${OUT_DIR}/${SX}${SUFFIX}_encox_metrics.json"
        if [[ -f "$DONE_FILE" ]]; then
            echo "  [skip] ${SX}${SUFFIX}_encox (metrics.json exists)"
            continue
        fi
        run_bg "train_encox_${SX}_${FEAT}" "$W_8B" \
            python -u train_cox.py --model encox --sex "$SX" --features "$FEAT"
        launched=$((launched + 1))
    done
done
if [[ $launched -gt 0 ]]; then
    wait
fi
echo "Wave 8b done (launched=$launched)."

# ---------- Wave 8c: XGB-Cox tune × 10 ----------
echo
echo "=== Wave 8c: XGB-Cox tune × 10 (6 parallel, skip existing) ==="
i=0
launched=0
for FEAT in all metabolic liver kidney immune; do
    case "$FEAT" in all) SUFFIX="" ;; *) SUFFIX="_${FEAT}" ;; esac
    for SX in female male; do
        DONE_FILE="${OUT_DIR}/${SX}${SUFFIX}_xgbcox_best_params.json"
        if [[ -f "$DONE_FILE" ]]; then
            echo "  [skip] ${SX}${SUFFIX}_xgbcox tune (best_params.json exists)"
            continue
        fi
        run_bg "tune_xgbcox_${SX}_${FEAT}" "$W_8C" \
            python -u tune_cox.py --model xgbcox --sex "$SX" --features "$FEAT"
        i=$((i+1))
        launched=$((launched + 1))
        if [[ $((i % 6)) -eq 0 ]]; then wait; fi
    done
done
[[ $launched -gt 0 ]] && wait
echo "Wave 8c done (launched=$launched)."

# ---------- Wave 8d: XGB-Cox train × 10 ----------
echo
echo "=== Wave 8d: XGB-Cox train × 10 (4 parallel, skip existing) ==="
i=0
launched=0
for FEAT in all metabolic liver kidney immune; do
    case "$FEAT" in all) SUFFIX="" ;; *) SUFFIX="_${FEAT}" ;; esac
    for SX in female male; do
        DONE_FILE="${OUT_DIR}/${SX}${SUFFIX}_xgbcox_metrics.json"
        if [[ -f "$DONE_FILE" ]]; then
            echo "  [skip] ${SX}${SUFFIX}_xgbcox train (metrics.json exists)"
            continue
        fi
        run_bg "train_xgbcox_${SX}_${FEAT}" "$W_8D" \
            python -u train_cox.py --model xgbcox --sex "$SX" --features "$FEAT"
        i=$((i+1))
        launched=$((launched + 1))
        if [[ $((i % 4)) -eq 0 ]]; then wait; fi
    done
done
[[ $launched -gt 0 ]] && wait
echo "Wave 8d done (launched=$launched)."

# ---------- Wave 8e: 评估刷新 ----------
echo
echo "=== Wave 8e: survival_v2 --loop-all + model_eval_v2 (master 42 行) ==="
# 备份现有 master 表
if [[ -f ../outputs/v2/model_eval_master.csv ]]; then
    cp ../outputs/v2/model_eval_master.csv \
       ../outputs/v2/model_eval_master_v23_backup.csv
    echo "  backed up old master -> model_eval_master_v23_backup.csv"
fi

OMP_NUM_THREADS="$TOTAL_CORES" \
OPENBLAS_NUM_THREADS="$TOTAL_CORES" \
MKL_NUM_THREADS="$TOTAL_CORES" \
    python -u survival_v2.py --loop-all 2>&1 | tee "$LOG_DIR/cox_survival_loop_all.log"

OMP_NUM_THREADS="$TOTAL_CORES" \
OPENBLAS_NUM_THREADS="$TOTAL_CORES" \
MKL_NUM_THREADS="$TOTAL_CORES" \
    python -u model_eval_v2.py 2>&1 | tee "$LOG_DIR/cox_model_eval_v2.log"

echo
echo "v2.4 cox ALL DONE. see ../outputs/v2/"
echo "★ master (42 rows): outputs/v2/model_eval_master.csv"
echo "★ 新图: outputs/v2/eval_age_vs_death_target.png"
echo "★ Gompertz params: outputs/v2/cox/gompertz_params_{female,male}.json"
echo "logs: $LOG_DIR/cox_*.log"
