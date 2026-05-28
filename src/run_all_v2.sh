#!/usr/bin/env bash
# v2 并行版（自适应核数）
# 用法：
#   本地 16 核：    TOTAL_CORES=16 bash run_all_v2.sh
#   集群 cnode03： TOTAL_CORES=144 bash run_all_v2.sh   （submit_v2.sh 已设）
#   不设默认 16
#
# 编排：
#   Wave 1: 4 tune + 2 LR train 并发 (6 路)，每路 = TOTAL_CORES / 6
#   Wave 2: 4 train (xgb/dnn × m/f) 并发，每路 = TOTAL_CORES / 4
#   Wave 3: survival_v2
set -euo pipefail
cd "$(dirname "$0")"

TOTAL_CORES=${TOTAL_CORES:-16}
W1_THREADS=$(( TOTAL_CORES / 6 ))
W2_THREADS=$(( TOTAL_CORES / 4 ))
[[ $W1_THREADS -lt 1 ]] && W1_THREADS=1
[[ $W2_THREADS -lt 1 ]] && W2_THREADS=1

LOG_DIR="../outputs/v2/_logs"
mkdir -p "$LOG_DIR"

run_bg() {
    local name="$1"; shift
    local threads="$1"; shift
    THREADS_PER_JOB="$threads" \
    OMP_NUM_THREADS="$threads" \
    OPENBLAS_NUM_THREADS="$threads" \
    MKL_NUM_THREADS="$threads" \
    LIGHTGBM_NUM_THREADS="$threads" \
        stdbuf -oL -eL "$@" > "$LOG_DIR/${name}.log" 2>&1 &
    echo "  [pid $!] $name (threads=$threads)"
}

echo "=== TOTAL_CORES=$TOTAL_CORES → wave1=${W1_THREADS}c×6, wave2=${W2_THREADS}c×4 ==="

# Wave 0: 切分（已存在则跳过）
if [[ ! -f ../data/split.parquet ]]; then
    python -u make_split.py
fi

# Wave 0.5: 全局一次性 miceforest 插补（v2.1 改动）
# 两性别串行；之后所有 tune/train 都直接读 features_imputed_v2_*.parquet
if [[ ! -f ../data/features_imputed_v2_female.parquet || ! -f ../data/features_imputed_v2_male.parquet ]]; then
    echo "=== Wave 0.5: global miceforest imputation (sequential) ==="
    OMP_NUM_THREADS="$TOTAL_CORES" \
    OPENBLAS_NUM_THREADS="$TOTAL_CORES" \
    MKL_NUM_THREADS="$TOTAL_CORES" \
    LIGHTGBM_NUM_THREADS="$TOTAL_CORES" \
        python -u impute_v2.py 2>&1 | tee "$LOG_DIR/impute_v2.log"
fi

# Wave 0.6: Spearman + BH-FDR 预筛报告 (2 并发，秒级)
if [[ ! -f ../outputs/v2/spearman_report_female.csv || ! -f ../outputs/v2/spearman_report_male.csv ]]; then
    echo "=== Wave 0.6: Spearman+BH report (2 parallel) ==="
    run_bg "spearman_female" 2 python -u spearman_report.py --sex female
    run_bg "spearman_male"   2 python -u spearman_report.py --sex male
    wait
fi

# Wave 1: 4 tune + 2 LR train (6 路并发)
echo "=== Wave 1: tune × 4 + LR train × 2 (parallel) ==="
run_bg "tune_xgb_female" "$W1_THREADS" python -u tune.py --model xgb --sex female
run_bg "tune_xgb_male"   "$W1_THREADS" python -u tune.py --model xgb --sex male
run_bg "tune_dnn_female" "$W1_THREADS" python -u tune.py --model dnn --sex female
run_bg "tune_dnn_male"   "$W1_THREADS" python -u tune.py --model dnn --sex male
run_bg "train_lr_female" "$W1_THREADS" python -u train_v2.py --model lr --sex female
run_bg "train_lr_male"   "$W1_THREADS" python -u train_v2.py --model lr --sex male
wait
echo "Wave 1 done."

# Wave 2: 4 train (xgb/dnn × m/f)，4 路并发
echo "=== Wave 2: xgb/dnn train × 4 (parallel) ==="
run_bg "train_xgb_female" "$W2_THREADS" python -u train_v2.py --model xgb --sex female
run_bg "train_xgb_male"   "$W2_THREADS" python -u train_v2.py --model xgb --sex male
run_bg "train_dnn_female" "$W2_THREADS" python -u train_v2.py --model dnn --sex female
run_bg "train_dnn_male"   "$W2_THREADS" python -u train_v2.py --model dnn --sex male
wait
echo "Wave 2 done."

# Wave 3: survival
echo "=== Wave 3: survival_v2 ==="
python -u survival_v2.py --model xgb

# ==================== v2.2 增量 ====================

# Wave 4a: 8 个模块化 EN 训练（4 模块 × 2 性别，EN 不调参，全并发）
echo "=== Wave 4a: module EN train × 8 (parallel) ==="
W4_THREADS=$(( TOTAL_CORES / 8 ))
[[ $W4_THREADS -lt 1 ]] && W4_THREADS=1
for MOD in metabolic liver kidney immune; do
    for SX in female male; do
        run_bg "train_en_${SX}_${MOD}" "$W4_THREADS" \
            python -u train_v2.py --model en --sex "$SX" --features "$MOD"
    done
done
wait
echo "Wave 4a done."

# Wave 4b: 8 个模块化 XGB 调参（4 模块 × 2 性别，6 路并发）
echo "=== Wave 4b: module XGB tune × 8 (6 parallel) ==="
W4B_THREADS=$(( TOTAL_CORES / 6 ))
[[ $W4B_THREADS -lt 1 ]] && W4B_THREADS=1
i=0
for MOD in metabolic liver kidney immune; do
    for SX in female male; do
        run_bg "tune_xgb_${SX}_${MOD}" "$W4B_THREADS" \
            python -u tune.py --model xgb --sex "$SX" --features "$MOD"
        i=$((i+1))
        if [[ $((i % 6)) -eq 0 ]]; then wait; fi
    done
done
wait
echo "Wave 4b done."

# Wave 4c: 8 个模块化 XGB 训练（5-fold OOF + refit，4 路并发）
echo "=== Wave 4c: module XGB train × 8 (4 parallel) ==="
W4C_THREADS=$(( TOTAL_CORES / 4 ))
[[ $W4C_THREADS -lt 1 ]] && W4C_THREADS=1
i=0
for MOD in metabolic liver kidney immune; do
    for SX in female male; do
        run_bg "train_xgb_${SX}_${MOD}" "$W4C_THREADS" \
            python -u train_v2.py --model xgb --sex "$SX" --features "$MOD"
        i=$((i+1))
        if [[ $((i % 4)) -eq 0 ]]; then wait; fi
    done
done
wait
echo "Wave 4c done."

# Wave 5: GAM 年龄轨迹（R + mgcv scat，2 并发）
echo "=== Wave 5: GAM trajectories (2 parallel) ==="
if [[ ! -f ../outputs/v2/gam/gam_summary_female.csv || ! -f ../outputs/v2/gam/gam_summary_male.csv ]]; then
    run_bg "gam_female" $(( TOTAL_CORES / 2 )) python -u run_gam.py --sex female
    run_bg "gam_male"   $(( TOTAL_CORES / 2 )) python -u run_gam.py --sex male
    wait
fi
echo "Wave 5 done."

# Wave 7: v2.3 横向评估
echo "=== Wave 7a: survival_v2 --loop-all (Cox + C-index for all 22 models) ==="
OMP_NUM_THREADS="$TOTAL_CORES" \
OPENBLAS_NUM_THREADS="$TOTAL_CORES" \
MKL_NUM_THREADS="$TOTAL_CORES" \
    python -u survival_v2.py --loop-all 2>&1 | tee "$LOG_DIR/survival_loop_all.log"

echo "=== Wave 7b: model_eval_v2 (master 表 + 5 张图) ==="
OMP_NUM_THREADS="$TOTAL_CORES" \
OPENBLAS_NUM_THREADS="$TOTAL_CORES" \
MKL_NUM_THREADS="$TOTAL_CORES" \
    python -u model_eval_v2.py 2>&1 | tee "$LOG_DIR/model_eval_v2.log"
echo "Wave 7 done."

echo
echo "v2 ALL DONE.  see ../outputs/v2/"
echo "logs in $LOG_DIR/"
echo "★ 横向对比看 outputs/v2/model_eval_master.csv"
