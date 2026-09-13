"""
评估：把"无掩码训练出来的策略"在**推理时**套上安全层，分数会怎样？

动机：
  * 掩码训练那条线在 20x20 上学不会觅食（贪心 0.2 分，比"掩码+随机"的 2.4 分还差）
  * 而无掩码训练出的策略贪心 62.8 分、死因 74% 自撞 —— 正是安全层能兜住的那类死亡
  * 所以更合理的打法是 train soft / shield hard：训练时不用掩码（保留死亡信号与探索），
    推理时用安全层拦掉自陷动作

这里直接拿现成 checkpoint 测 with/without 掩码，以及不同回合上限（掩码会拖长回合，
1600 步的上限可能成为新的瓶颈）。

用法： .venv/bin/python eval_shield.py [局数]
"""

import os
import sys

import numpy as np
import torch

from snake_env import VectorSnake, N_ACTIONS, CAUSE_WALL, CAUSE_SELF, CAUSE_CAP, CAUSE_WIN
from train import ActorCritic, apply_mask

CKPTS = [("ppo20 (55M, 无掩码)", "runs/ppo20/ckpt.pt"),
         ("ab_both (20M, 无掩码)", "runs/ab_both/ckpt.pt")]


def evaluate(ckpt_path, mask, max_steps, n_ep=300, n_env=64, grid=20, seed=777):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    a = ck.get("args", {})
    env = VectorSnake(n_envs=n_env, rows=grid, cols=grid, max_steps=max_steps, seed=seed,
                      use_patch=not a.get("no_patch", False),
                      food_reward=0.0, death_penalty=0.0, step_penalty=0.0,
                      shaping_scale=0.0, area_shaping=0.0, doomed_penalty=0.0,
                      safety_mask=mask)
    model = ActorCritic(env.obs_dim, N_ACTIONS, a.get("hidden", 256))
    model.load_state_dict(ck["model"])
    model.eval()
    obs = env.observe()
    scores, causes, lens, stp = [], [], [], []
    while len(scores) < n_ep:
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(obs))
            if mask:
                logits = apply_mask(logits, torch.from_numpy(env.safe))
            act = logits.argmax(dim=-1).numpy()
        obs, _, _, _, info = env.step(act)
        for i in np.flatnonzero(info["finished"]):
            scores.append(int(info["final_score"][i]))
            causes.append(int(info["cause"][i]))
            lens.append(int(info["final_len"][i]))
            stp.append(int(info["final_steps"][i]))
    return (np.array(scores[:n_ep]), np.array(causes[:n_ep]),
            np.array(lens[:n_ep]), np.array(stp[:n_ep]))


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print(f"贪心评估 {n_ep} 局（seed=777，20x20，满分 397）\n")
    print(f"{'checkpoint':<24}{'安全层':>7}{'上限':>7}{'均分':>9}{'中位':>7}{'最高':>7}"
          f"{'均长':>7}{'均步数':>8}{'墙%':>7}{'己%':>7}{'超时%':>8}")
    for name, path in CKPTS:
        if not os.path.exists(path):
            print(f"{name:<24}  缺 ckpt")
            continue
        for mask in (False, True):
            for cap in ((1600, 4000) if mask else (1600,)):
                s, c, l, st = evaluate(path, mask, cap, n_ep)
                print(f"{name:<24}{'开' if mask else '关':>7}{cap:>7}"
                      f"{s.mean():>9.2f}{np.median(s):>7.0f}{s.max():>7}"
                      f"{l.mean():>7.0f}{st.mean():>8.0f}"
                      f"{100*(c==CAUSE_WALL).mean():>7.1f}"
                      f"{100*(c==CAUSE_SELF).mean():>7.1f}"
                      f"{100*(c==CAUSE_CAP).mean():>8.1f}")


if __name__ == "__main__":
    main()
