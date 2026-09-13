"""
评估"吃满"：用足够大的回合上限跑到结束，统计吃满率与吃满步数。

对照基线：
  * 纯 Hamilton 回路跟随：吃满率 100%，平均 39,802 步（每颗食物 100.3 步）
  * 理论下界：397 步（每步都在吃）
  * 旧 RL + 尾巴安全盾：吃满率 0%（会死），均分 105

用法： .venv/bin/python eval_fill.py <ckpt> [局数] [回合上限] [shield_mode]
例：   .venv/bin/python eval_fill.py runs/arc_rl/ckpt.pt 200 60000 arc
"""

import sys
import time

import numpy as np
import torch

from snake_env import VectorSnake
from train import ActorCritic, N_ACTIONS, apply_mask


def main():
    ckpt = sys.argv[1]
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
    mode = sys.argv[4] if len(sys.argv) > 4 else "arc"
    grid = 20
    full = grid * grid - 3

    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    a = ck.get("args", {})
    env = VectorSnake(n_envs=64, rows=grid, cols=grid, max_steps=cap, seed=12345,
                      use_patch=not a.get("no_patch", False), safety_mask=True,
                      shield_mode=mode)
    env.food_reward = 1.0
    env.death_penalty = 0.0
    env.step_penalty = 0.0
    env.shaping_scale = 0.0
    env.area_shaping = 0.0
    env.doomed_penalty = 0.0
    net = ActorCritic(env.obs_dim, N_ACTIONS, a.get("hidden", 256))
    net.load_state_dict(ck["model"])
    net.eval()

    print(f"评估 {ckpt}（{n_ep} 局，{grid}x{grid}，满分 {full}，回合上限 {cap}，盾={mode}）")
    obs = env.observe()
    scores, steps, causes = [], [], []
    t0 = time.time()
    while len(scores) < n_ep:
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(obs))
            logits = apply_mask(logits, torch.from_numpy(env.safe))
            act = logits.argmax(dim=-1).numpy()
        obs, _, _, _, info = env.step(act)
        for i in np.flatnonzero(info["finished"]):
            scores.append(int(info["final_score"][i]))
            steps.append(int(info["final_steps"][i]))
            causes.append(int(info["cause"][i]))
    s = np.array(scores[:n_ep], dtype=np.float64)
    st = np.array(steps[:n_ep], dtype=np.float64)
    c = np.array(causes[:n_ep])
    filled = s >= full
    print(f"  均分 {s.mean():.1f}  中位 {np.median(s):.0f}  最高 {s.max():.0f}  "
          f"p10 {np.percentile(s,10):.0f}")
    print(f"  吃满率 {100*filled.mean():.1f}%（{int(filled.sum())}/{n_ep}）")
    if filled.any():
        print(f"  吃满步数: 均值 {st[filled].mean():,.0f}  中位 {np.median(st[filled]):,.0f}  "
              f"最少 {st[filled].min():,.0f}  ← 对比纯 Hamilton 39,802 / 下界 397")
        print(f"  每颗食物步数: {st[filled].mean()/full:.1f}  ← 对比 Hamilton 100.3")
    print(f"  结束原因: 墙 {100*(c==0).mean():.1f}%  自撞 {100*(c==1).mean():.1f}%  "
          f"通关 {100*(c==3).mean():.1f}%  撞步数上限 {100*(c==2).mean():.1f}%")
    print(f"  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
