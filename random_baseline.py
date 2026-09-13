"""
对照测试：8x8 上"均匀随机动作"能拿多少分？

目的：检验 curriculum 阶段 1 那个"训练态均分 55.05/61"到底是**学到了觅食**，
还是**小棋盘上随机游走本来就能撞到食物**（如果是后者，那 55 分是噪声的功劳，
贪心评估 median 0 / max 28 才是真相）。

同时给出：开/关安全掩码、1600 步上限下的随机策略分数。

用法： .venv/bin/python random_baseline.py
"""

import sys

import numpy as np

from snake_env import VectorSnake


def run(grid, n_env, episodes, mask, max_steps, seed=5):
    env = VectorSnake(n_envs=n_env, rows=grid, cols=grid, max_steps=max_steps, seed=seed,
                      food_reward=0.0, death_penalty=0.0, step_penalty=0.0,
                      shaping_scale=0.0, safety_mask=mask)
    rng = np.random.default_rng(seed)
    scores, steps, causes = [], [], []
    # 掩码开启时只用允许的动作，模拟"策略完全随机但被安全层兜着"
    obs = env.observe()
    while len(scores) < episodes:
        if mask:
            allowed = env.safe
            a = np.zeros(env.n, dtype=np.int64)
            for i in range(env.n):
                idx = np.flatnonzero(allowed[i])
                if idx.size == 0:          # 与训练器一致：三个动作全禁时放开全部
                    idx = np.arange(3)
                a[i] = int(rng.choice(idx))
        else:
            a = rng.integers(0, 3, size=env.n)
        _, _, _, _, info = env.step(a)
        for i in np.flatnonzero(info["finished"]):
            scores.append(int(info["final_score"][i]))
            steps.append(int(info["final_steps"][i]))
            causes.append(int(info["cause"][i]))
    s = np.array(scores[:episodes])
    return s, np.array(steps[:episodes]), np.array(causes[:episodes])


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    FULL = {8: 61, 20: 397}
    print("均匀随机策略（不吃任何学习成果）—— 用来判断『高分』是不是噪声的功劳\n")
    print(f"{'棋盘':>5}{'掩码':>6}{'上限':>7}{'均分':>9}{'中位':>7}{'最高':>7}"
          f"{'均长':>7}{'满分':>7}{'每局步数':>10}")
    for grid in (8, 20):
        for mask in (False, True):
            for max_steps in (1600,):
                s, st, c = run(grid, 128, episodes, mask, max_steps)
                print(f"{grid:>5}{'开' if mask else '关':>6}{max_steps:>7}"
                      f"{s.mean():>9.2f}{np.median(s):>7.0f}{s.max():>7}"
                      f"{st.mean():>7.0f}{FULL[grid]:>7}{st.mean():>10.0f}")
                if grid == 8 and mask:
                    print(f"      ↑ 若这个数接近 55，说明之前 curriculum 阶段 1 的"
                          f"'均分 55' 主要是小棋盘随机游走的产物，而不是学会了觅食")


if __name__ == "__main__":
    main()
