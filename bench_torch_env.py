"""
环境吞吐对比：numpy(CPU) vs torch(GPU)，含安全层两种开关。

两者语义已由 test_torch_parity.py 证明一致（状态/观测/安全层/奖励/结局全对），
所以这里比的纯粹是速度。

用法： .venv/bin/python bench_torch_env.py
"""

import time

import numpy as np
import torch

from snake_env import VectorSnake
from snake_env_torch import VectorSnakeTorch

CFG = dict(seed=0, shaping_scale=0.25, area_shaping=0.2, doomed_penalty=0.3)


def bench_np(n, steps, mask):
    env = VectorSnake(n_envs=n, safety_mask=mask, **CFG)
    env.observe()
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 3, size=(steps, n))
    t0 = time.time()
    for t in range(steps):
        env.step(acts[t])
    dt = time.time() - t0
    return n * steps / dt


def bench_torch(n, steps, mask, device="cuda"):
    env = VectorSnakeTorch(n_envs=n, safety_mask=mask, device=device, **CFG)
    env.observe()
    acts = torch.randint(0, 3, (steps, n), device=device)
    for _ in range(5):                     # 预热（含 kernel 编译/JIT）
        env.step(acts[0])
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(steps):
        env.step(acts[t])
    torch.cuda.synchronize()
    dt = time.time() - t0
    return n * steps / dt


def main():
    print(f"环境吞吐对比（20x20，含观测构造 / BFS / 自动重开）")
    print(f"{'环境数':>7} {'安全层':>6} {'numpy(CPU)':>14} {'torch(GPU)':>14} {'加速':>7}")
    for n, steps in ((1024, 60), (2048, 40), (4096, 25)):
        for mask in (False, True):
            a = bench_np(n, steps, mask)
            b = bench_torch(n, steps, mask)
            print(f"{n:>7} {str(mask):>6} {a:>13,.0f} {b:>14,.0f} {b/a:>6.1f}x", flush=True)


if __name__ == "__main__":
    main()
