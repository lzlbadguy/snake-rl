"""
性能基准 —— 回答"并行多环境到底有没有用"。

同口径对比（都要产出训练用的观测）：
  A. 朴素串行单环境：纯 Python 规则 + 每步 Python BFS（就是"一次跑一个环境"的写法）
  B. 向量化 N 环境：同一条规则，N 个环境同步推进（一次 numpy 调用服务 N 个环境）

另附：只跑规则不算观测的参考值（说明 numpy 在 N=1 时反而是负担）、
观测各部分的成本归因、以及小 MLP 的前向/反向吞吐。

用法： .venv/bin/python bench_env.py
"""

from __future__ import annotations

import time
import numpy as np

from snake_env import VectorSnake, SnakeRefEnv


def _best(fn, reps):
    return max(fn() for _ in range(reps))


def bench_ref_rule(steps: int = 3000, reps: int = 5) -> float:
    """只有规则、不算观测：纯 Python 有多快（N=1 时 numpy 反而更慢）。"""
    def run():
        env = SnakeRefEnv()
        acts = np.random.default_rng(1).integers(0, 3, size=steps)
        t0 = time.perf_counter()
        for a in acts:
            env.step(int(a))
            if env.dead or env.win:
                env.reset()
        return steps / (time.perf_counter() - t0)
    return _best(run, reps)


def bench_ref_obs(steps: int = 150, reps: int = 3) -> float:
    """朴素串行训练基线：每步都要算 BFS 距离 + 可达面积（纯 Python）。"""
    def run():
        env = SnakeRefEnv()
        acts = np.random.default_rng(1).integers(0, 3, size=steps)
        t0 = time.perf_counter()
        for a in acts:
            env.step(int(a))
            if env.dead or env.win:
                env.reset()
            else:
                env.bfs_dist_to_food()
                env.reachable_area()
        return steps / (time.perf_counter() - t0)
    return _best(run, reps)


def bench_vec(n_envs: int, steps: int = 60, use_patch: bool = True,
              reps: int = 3, no_bfs: bool = False) -> float:
    def run():
        env = VectorSnake(n_envs=n_envs, seed=0, use_patch=use_patch)
        if no_bfs:
            # 仅用于成本归因：把 BFS 换成常数结果（不能用于训练）
            zeros = np.zeros((n_envs, env.rows, env.cols), dtype=np.int16)
            ones = np.ones((n_envs, env.rows, env.cols), dtype=bool)
            env._bfs = lambda: (zeros, ones, np.zeros(n_envs, np.int16), np.zeros(n_envs, np.int16))
        rng = np.random.default_rng(1)
        env.step(rng.integers(0, 3, size=n_envs))
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(rng.integers(0, 3, size=n_envs))
        return steps * n_envs / (time.perf_counter() - t0)
    return _best(run, reps)


def bench_net(obs_dim: int, hidden: int, batch: int, iters: int = 20):
    import torch
    from train import ActorCritic

    model = ActorCritic(obs_dim, 3, hidden)
    x = torch.randn(batch, obs_dim)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        fwd = iters * batch / (time.perf_counter() - t0)
    for _ in range(3):
        logits, v = model(x)
        (logits.pow(2).mean() + v.pow(2).mean()).backward()
        opt.zero_grad()
    t0 = time.perf_counter()
    for _ in range(iters):
        logits, v = model(x)
        loss = logits.pow(2).mean() + v.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    fwd_bwd = iters * batch / (time.perf_counter() - t0)
    return fwd, fwd_bwd


def main():
    print("=== 同口径环境吞吐（都产出训练观测）===")
    serial = bench_ref_obs()
    print(f"{'A. 串行 1 环境 + Python BFS':<34}{serial:>12,.0f} 步/秒{'':>4}(基线 1.0x)")
    rows = []
    for n in (1, 8, 64, 256, 1024, 2048, 4096, 8192):
        sps = bench_vec(n)
        rows.append((n, sps))
        print(f"{'B. 向量化 n_envs=' + str(n):<34}{sps:>12,.0f} 步/秒{sps / serial:>6.1f}x")

    print("\n=== 仅规则、不算观测（说明 N=1 时 numpy 的开销）===")
    rule = bench_ref_rule()
    print(f"串行纯 Python 规则                     {rule:>12,.0f} 步/秒")
    print(f"向量化 n_envs=1（同一套规则 + 观测）    {rows[0][1]:>12,.0f} 步/秒"
          f"   -> numpy 在 N=1 时比纯 Python 慢 {rule / rows[0][1]:.0f}x")

    print("\n=== 观测成本归因（n_envs=2048）===")
    full = bench_vec(2048)
    nobfs = bench_vec(2048, no_bfs=True)
    nopatch = bench_vec(2048, use_patch=False)
    print(f"完整观测（BFS + 23 标量 + 7x7x3 patch）  {full:>12,.0f} 步/秒")
    print(f"去掉 patch                            {nopatch:>12,.0f} 步/秒"
          f"   (patch 占 {100 * (1 - full / nopatch):.0f}%)")
    print(f"去掉 BFS（仅归因，不可用于训练）        {nobfs:>12,.0f} 步/秒"
          f"   (BFS/flood-fill 占 {100 * (1 - full / nobfs):.0f}%)")

    print("\n=== 小 MLP 吞吐（CPU，170->256->256，batch 8192）===")
    fwd, fwd_bwd = bench_net(170, 256, 8192)
    print(f"前向        {fwd:>12,.0f} 样本/秒")
    print(f"前向+反向   {fwd_bwd:>12,.0f} 样本/秒")


if __name__ == "__main__":
    main()
