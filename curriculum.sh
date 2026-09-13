#!/usr/bin/env bash
# B 部分：安全掩码 + 棋盘尺寸课程。
#
# 为什么需要课程：掩码拿掉了"死亡"这个高频信号后回合极长，20x20 上单位时间拿到的
# 经验回合数只有无掩码时的 ~1/80，食物奖励太稀疏，策略起不来（8x8 上同一套配置
# 几分钟就能把 61 分满分吃满，证明方法本身没问题）。
#
# 关键便利：观测维度与棋盘尺寸无关（特征按盘面归一化、patch 固定 7x7），
# 所以权重可以跨尺寸直接迁移，只需要 --init-ckpt 接上。
#
# 回合上限由环境自动按 4*格数 取：8->256, 12->576, 16->1024, 20->1600
set -u
cd /home/lzl/snake-rl
PY=.venv/bin/python
COMMON="--n-envs 1024 --rollout 128 --eval-every 10 --eval-episodes 100 --minutes 90 \
        --shaping 0.25 --area-shaping 0.2 --doomed-penalty 0.3 --mask-safety"

stage () {
  local grid=$1 steps=$2 init=$3 out=$4
  local extra=""
  [ -n "$init" ] && extra="--init-ckpt $init"
  echo "[curr] ================= grid=${grid}x${grid}  ${steps} 步  $(date '+%H:%M:%S') ================="
  $PY -u train.py $COMMON --grid "$grid" --train-steps "$steps" $extra --out "$out"
  echo "[curr] ---------------- grid=${grid} 完成  $(date '+%H:%M:%S') ----------------"
}

stage 8  4000000  ""                     runs/cur_g8
stage 12 6000000  runs/cur_g8/ckpt.pt    runs/cur_g12
stage 16 8000000  runs/cur_g12/ckpt.pt   runs/cur_g16
stage 20 12000000 runs/cur_g16/ckpt.pt   runs/cur_g20

echo "[curr] 全部阶段完成 $(date '+%F %H:%M:%S')"
