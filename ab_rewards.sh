#!/usr/bin/env bash
# 奖励 A/B 实验：等当前 45 分钟基线跑完，再顺序跑 4 个配置。
# 关键：全部按"环境步数"停止并按步数进度退火 lr，保证四个配置在**同一步数**上可比。
#
#   base    基线（与当前配置完全一致，只是换成按步数停止 → 作为公平参照）
#   area    势函数里加入"可达空间"项：Φ = -0.25·bfs_norm - 0.2·(1 - area_frac)
#   doomed  "注定死亡"（可达格数 < 长度）时的每步 -0.3 提前惩罚
#   both    两者叠加
#
# 结果：runs/ab_*/metrics.csv、eval.csv（每 2.6M 步一次 200 局贪心评估）
set -u
cd /home/lzl/snake-rl
PY=.venv/bin/python
STEPS=20000000
COMMON="--n-envs 2048 --rollout 128 --train-steps $STEPS --minutes 240 \
        --eval-every 10 --eval-episodes 200 --shaping 0.25"

echo "[ab] 等待基线（45 分钟那次）结束 ... $(date '+%F %H:%M:%S')"
while pgrep -f "train.py --minutes 45 --out runs/ppo20" >/dev/null 2>&1; do sleep 20; done
echo "[ab] 基线已结束，开始奖励 A/B $(date '+%F %H:%M:%S')"

run() {
  local name="$1"; shift
  echo "[ab] ==================== 开始 $name  $(date '+%H:%M:%S') ===================="
  $PY -u train.py $COMMON --out "runs/ab_$name" "$@"
  echo "[ab] ==================== 结束 $name  $(date '+%H:%M:%S') ===================="
}

run base
run area   --area-shaping 0.2
run doomed --doomed-penalty 0.3
run both   --area-shaping 0.2 --doomed-penalty 0.3

echo "[ab] 全部完成 $(date '+%F %H:%M:%S')"
