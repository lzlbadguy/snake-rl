"""
GPU vs CPU 基准：回答"到底该不该用显存/GPU"。

本机是 DGX Spark GB10（Grace Blackwell，sm_121），121GB 是 **CPU/GPU 统一的 LPDDR5X 内存池**，
不是独立的显存 —— 所以"没用显存"的真正含义是"没用 GPU 的算力"，而不是"内存没吃满"。

测两件事：
  A. 小 MLP（170->256->256）的前向 / 前向+反向：CPU vs GPU，按 PPO 的实际批大小
     （minibatch 16384、整批 262144）
  B. 环境里占 96% 开销的 BFS/flood-fill：numpy（CPU） vs torch（GPU）
     —— 这是真正可能被 GPU 救回来的部分

用法： .venv/bin/python bench_gpu.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from train import ActorCritic

COLS = ROWS = 20
N = 2048


def timeit(fn, iters=10, warmup=3, gpu=False):
    for _ in range(warmup):
        fn()
    if gpu:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if gpu:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# ---------------- BFS 的两种实现 ----------------

def bfs_numpy(occ, src):
    """occ: (n,R,C) bool；src: (n,) cell（起点须已置为可通行）。返回 vis。"""
    n = occ.shape[0]
    ar = np.arange(n)
    sy, sx = src // COLS, src % COLS
    vis = np.zeros_like(occ)
    fr = np.zeros_like(occ)
    fr[ar, sy, sx] = True
    vis |= fr
    while fr.any():
        nb = np.zeros_like(fr)
        nb[:, 1:, :] |= fr[:, :-1, :]
        nb[:, :-1, :] |= fr[:, 1:, :]
        nb[:, :, 1:] |= fr[:, :, :-1]
        nb[:, :, :-1] |= fr[:, :, 1:]
        nb &= ~occ
        nb &= ~vis
        if not nb.any():
            break
        vis |= nb
        fr = nb
    return vis


def bfs_torch(occ, src, fixed_layers=None, leave_src_free=True):
    """torch 版（可放在 GPU）。fixed_layers 非空时不做逐层 any() 早停（避免每层一次同步）。"""
    n = occ.shape[0]
    ar = torch.arange(n, device=occ.device)
    sy, sx = src // COLS, src % COLS
    vis = torch.zeros_like(occ)
    fr = torch.zeros_like(occ)
    fr[ar, sy, sx] = True
    vis |= fr
    d = 0
    while True:
        nb = torch.zeros_like(fr)
        nb[:, 1:, :] |= fr[:, :-1, :]
        nb[:, :-1, :] |= fr[:, 1:, :]
        nb[:, :, 1:] |= fr[:, :, :-1]
        nb[:, :, :-1] |= fr[:, :, 1:]
        nb &= ~occ
        nb &= ~vis
        d += 1
        if fixed_layers is not None:
            vis |= nb
            fr = nb
            if d >= fixed_layers:
                break
        else:
            if not bool(nb.any()):
                break
            vis |= nb
            fr = nb
    return vis


def make_state(device="cpu"):
    rng = np.random.default_rng(0)
    occ = np.zeros((N, ROWS, COLS), dtype=bool)
    # 造几条随机蛇，制造和真实训练接近的占据率
    for i in range(N):
        y = rng.integers(1, ROWS - 1)
        for k in range(int(rng.integers(5, 60))):
            x = (3 + k) % (COLS - 1)
            occ[i, y, x] = True
    head = np.array([np.flatnonzero(~occ[i].reshape(-1))[0] for i in range(N)], dtype=np.int64)
    return occ, head


def main():
    print("=== 设备 ===")
    print(f"torch {torch.__version__}  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"设备: {p.name}  capability sm_{p.major}{p.minor}  "
              f"总内存 {p.total_memory/2**30:.0f} GiB（统一内存池）")

    # ---------------- A. 网络 ----------------
    print("\n=== A. 小 MLP 170->256->256 ===")
    print(f"{'设备':<6}{'批大小':>9}{'前向 (样本/秒)':>18}{'前向+反向 (样本/秒)':>22}")
    for bs in (16384, 262144):
        for dev, gpu in (("cpu", False), ("cuda", True)):
            if gpu and not torch.cuda.is_available():
                continue
            m = ActorCritic(170, 3, 256).to(dev)
            x = torch.randn(bs, 170, device=dev)
            opt = torch.optim.Adam(m.parameters(), lr=3e-4)

            with torch.no_grad():
                t = timeit(lambda: m(x), iters=10, gpu=gpu)
            fwd = bs / t

            def stepb():
                logits, v = m(x)
                loss = logits.pow(2).mean() + v.pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            t2 = timeit(stepb, iters=10, gpu=gpu)
            print(f"{dev:<6}{bs:>9}{fwd:>18,.0f}{bs/t2:>22,.0f}")

    # ---------------- B. BFS ----------------
    print("\n=== B. BFS/flood-fill（环境 96% 的开销都在这）===")
    occ_np, head = make_state()
    print(f"n_envs={N}, 20x20, 随机占据")
    t_cpu1 = timeit(lambda: bfs_numpy(occ_np, head), iters=5)
    print(f"{'numpy (CPU)':<28}{'1 次 BFS':>16}{1/t_cpu1:>12,.0f} 次/秒")
    if torch.cuda.is_available():
        occ_g = torch.from_numpy(occ_np).cuda()
        src_g = torch.from_numpy(head).cuda()
        t_g_sync = timeit(lambda: bfs_torch(occ_g, src_g), iters=5, gpu=True)
        print(f"{'torch (GPU, 逐层早停)':<28}{'1 次 BFS':>16}{1/t_g_sync:>12,.0f} 次/秒"
              f"   vs CPU {t_cpu1/t_g_sync:5.1f}x")
        for layers in (45, 80):
            t_g_fix = timeit(lambda: bfs_torch(occ_g, src_g, fixed_layers=layers),
                             iters=5, gpu=True)
            print(f"{'torch (GPU, 固定' + str(layers) + ' 层)':<28}{'1 次 BFS':>16}"
                  f"{1/t_g_fix:>12,.0f} 次/秒   vs CPU {t_cpu1/t_g_fix:5.1f}x")
    print(f"\n安全层每步需要 3 次 BFS（直行/左转/右转各一次）：")
    print(f"  CPU: 单步环境开销约 {3*t_cpu1*1000:.1f} ms 仅 BFS 部分"
          f"  → 理论上限 {N/(3*t_cpu1):,.0f} 环境步/秒")


if __name__ == "__main__":
    main()
