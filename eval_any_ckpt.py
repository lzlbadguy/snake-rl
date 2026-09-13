"""
评估任意 ckpt：安全层 on/off × 回合上限，同一种子。

用法： .venv/bin/python eval_any_ckpt.py <ckpt> [局数] [grid]

例：   .venv/bin/python eval_any_ckpt.py runs/cur_g20/ckpt.pt 200 20
"""

import sys
import time

import numpy as np
import torch

from snake_env import VectorSnake
from train import ActorCritic, N_ACTIONS, apply_mask


def ev(ckpt, grid, n_ep, mask, cap, seed=777, tier=True):
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    a = ck.get("args", {})
    env = VectorSnake(n_envs=64, rows=grid, cols=grid, max_steps=cap, seed=seed,
                      use_patch=not a.get("no_patch", False), safety_mask=mask,
                      tier_shield=tier)
    env.food_reward = 0.0
    env.death_penalty = 0.0
    env.step_penalty = 0.0
    env.shaping_scale = 0.0
    env.area_shaping = 0.0
    env.doomed_penalty = 0.0
    net = ActorCritic(env.obs_dim, N_ACTIONS, a.get("hidden", 256))
    net.load_state_dict(ck["model"])
    net.eval()

    obs = env.observe()
    scores, causes, steps = [], [], []
    while len(scores) < n_ep:
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(obs))
            if mask:
                logits = apply_mask(logits, torch.from_numpy(env.safe))
            act = logits.argmax(dim=-1).numpy()
        obs, _, _, _, info = env.step(act)
        for i in np.flatnonzero(info["finished"]):
            scores.append(int(info["final_score"][i]))
            causes.append(int(info["cause"][i]))
            steps.append(int(info["final_steps"][i]))
    s = np.array(scores[:n_ep])
    c = np.array(causes[:n_ep])
    st = np.array(steps[:n_ep])
    n = len(s)
    return dict(mean=s.mean(), median=float(np.median(s)), mx=s.max(), p10=float(np.percentile(s, 10)),
                steps=st.mean(), wall=100 * (c == 0).mean(), self_=100 * (c == 1).mean(),
                cap=100 * (c == 2).mean(), win=100 * (c == 3).mean())


def main():
    ckpt = sys.argv[1]
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    grid = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    print(f"评估 {ckpt}（{n_ep} 局，seed=777，{grid}x{grid}，满分 {grid*grid-3}）")
    print(f"{'安全层':>6} {'上限':>6} {'均分':>8} {'中位':>6} {'最高':>6} {'p10':>6} "
          f"{'均步数':>8} {'墙%':>6} {'己%':>6} {'超时%':>7}")
    for mask, cap, tier, tag in ((False, 4000, True, "无盾"), (True, 4000, False, "旧盾"),
                                 (True, 4000, True, "新盾(两级)")):
        t0 = time.time()
        r = ev(ckpt, grid, n_ep, mask, cap, tier=tier)
        print(f"{tag if mask else '—':>10} {cap:>6} {r['mean']:>8.2f} {r['median']:>6.0f} {r['mx']:>6} "
              f"{r['p10']:>6.0f} {r['steps']:>8.0f} {r['wall']:>6.1f} {r['self_']:>6.1f} "
              f"{r['cap']:>7.1f}   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
