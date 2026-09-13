"""
A/B 汇总：读四条曲线，并在**相同种子集**上对四个 checkpoint 做 1000 局贪心评估。

为什么要重评一遍：eval.csv 里的均值没有逐局分布，无法判断差异是否超出噪声。
这里统一用 seed=777、128 个并行环境、1000 局，给出均分 ± 标准误 / 中位 / p10 / p90 / 死因分解。

用法： .venv/bin/python eval_ckpts.py
"""

import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snake_env import VectorSnake, N_ACTIONS, CAUSE_WALL, CAUSE_SELF, CAUSE_CAP
from train import ActorCritic

RUNS = [("base", "runs/ab_base"), ("area", "runs/ab_area"),
        ("doomed", "runs/ab_doomed"), ("both", "runs/ab_both")]


def load_eval(path):
    rows = []
    p = os.path.join(path, "eval.csv")
    if not os.path.exists(p):
        return rows
    with open(p) as f:
        for r in csv.reader(f):
            if r:
                rows.append([float(x) for x in r])
    return rows          # [update, steps, mean, median, max, wall, self, cap]


def evaluate(ckpt_path, n_episodes=1000, n_env=128, seed=777):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    a = ck.get("args", {})
    env = VectorSnake(n_envs=n_env, seed=seed, use_patch=not a.get("no_patch", False),
                      food_reward=0.0, death_penalty=0.0, step_penalty=0.0,
                      shaping_scale=0.0, area_shaping=0.0, doomed_penalty=0.0)
    model = ActorCritic(env.obs_dim, N_ACTIONS, a.get("hidden", 256))
    model.load_state_dict(ck["model"])
    model.eval()
    obs = env.observe()
    scores, causes, lens = [], [], []
    while len(scores) < n_episodes:
        with torch.no_grad():
            act = model(torch.from_numpy(obs))[0].argmax(-1).numpy()
        obs, _, _, _, info = env.step(act)
        for i in np.flatnonzero(info["finished"]):
            scores.append(int(info["final_score"][i]))
            lens.append(int(info["final_len"][i]))
            causes.append(int(info["cause"][i]))
    s = np.array(scores[:n_episodes])
    return s, np.array(causes[:n_episodes]), np.array(lens[:n_episodes]), ck.get("update")


def main():
    print("=== 训练过程中的贪心评估曲线（每 10 次更新一次，200 局）===")
    print(f"{'upd':>5}{'步数(M)':>8}" + "".join(f"{n:>16}" for n, _ in RUNS))
    curves = {n: load_eval(p) for n, p in RUNS}
    # 按更新序号对齐（同一 update 号在所有 run 里代表相同的环境步数），避免浮点匹配踩坑
    upd_axis = sorted({int(r[0]) for rows in curves.values() for r in rows})
    for up in upd_axis:
        hits = {n: next((r for r in curves[n] if int(r[0]) == up), None) for n, _ in RUNS}
        if not any(hits.values()):
            continue
        step_m = next((r[1] for r in hits.values() if r), 0) / 1e6
        line = f"{up:>5}{step_m:>8.1f}"
        for n, _ in RUNS:
            r = hits[n]
            line += f"{r[2]:>16.2f}" if r else f"{'-':>16}"
        print(line)

    print("\n=== 四个 checkpoint 的 1000 局重评（同种子 seed=777，贪心）===")
    print(f"{'配置':<8}{'均分':>9}{'±SE':>7}{'中位':>7}{'p10':>7}{'p90':>7}{'最高':>7}"
          f"{'均长':>8}{'墙%':>7}{'己%':>7}{'上限%':>8}")
    res = {}
    for name, path in RUNS:
        ck = os.path.join(path, "ckpt.pt")
        if not os.path.exists(ck):
            print(f"{name:<8} 缺 ckpt.pt")
            continue
        s, c, l, upd = evaluate(ck)
        se = s.std(ddof=1) / np.sqrt(len(s))
        print(f"{name:<8}{s.mean():>9.2f}{se:>7.2f}{np.median(s):>7.0f}"
              f"{np.percentile(s, 10):>7.0f}{np.percentile(s, 90):>7.0f}{s.max():>7.0f}"
              f"{l.mean():>8.1f}{100 * (c == CAUSE_WALL).mean():>7.1f}"
              f"{100 * (c == CAUSE_SELF).mean():>7.1f}{100 * (c == CAUSE_CAP).mean():>8.1f}")
        res[name] = (s.mean(), se, upd)

    if "base" in res:
        print("\n--- 相对 base 的差值（合并标准误 + 近似 t）---")
        for name, (m, se, upd) in res.items():
            if name == "base":
                continue
            d = m - res["base"][0]
            sd = float(np.sqrt(se ** 2 + res["base"][1] ** 2))
            verdict = "显著" if abs(d) > 2 * sd else "与噪声无法区分"
            print(f"  {name:<7} {d:+6.2f} 分   合并SE {sd:.2f}   t≈{d/sd:+5.1f}   {verdict}"
                  f"   (ckpt update={upd})")


if __name__ == "__main__":
    main()
