"""
对拍：GPU(torch) 环境 vs CPU(numpy) 环境。

两者喂**同一串**均匀随机数（决定食物落点）和**同一串**动作，要求逐步完全一致：
  * 状态：身体序列 / 头指针 / 长度 / 食物 / 分数 / 步数 / 朝向 / 可达面积
  * 观测：170 维（float32 精度内）
  * 安全层判据：三个动作的布尔值（必须**完全相等**，不许有近似）
  * 奖励、结束时机、死因、终局分数

这是把环境搬上 GPU 之前必须过的关：一旦不一致，GPU 训练出的策略在游戏里就会失准。

用法： .venv/bin/python test_torch_parity.py [步数]
"""

import sys

import numpy as np
import torch

from snake_env import VectorSnake
from snake_env_torch import VectorSnakeTorch

N = 64


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    if not torch.cuda.is_available():
        sys.exit("CUDA 不可用")
    rng = np.random.default_rng(1234)
    uniforms = rng.random(3_000_000)
    acts = rng.integers(0, 3, size=(steps, N))

    kw = dict(n_envs=N, seed=11, food_reward=1.0, death_penalty=1.0, step_penalty=0.001,
              shaping_scale=0.25, area_shaping=0.2, doomed_penalty=0.3,
              safety_mask=True, shield_mode="arc")
    a = VectorSnake(**kw, uniforms=uniforms)
    b = VectorSnakeTorch(**kw, uniforms=uniforms, device="cuda")

    m_obs = m_rew = 0.0
    bad_state = bad_safe = bad_fin = 0
    first = None
    fin_a = fin_b = 0

    for t in range(steps):
        diff = (a.safe != b.safe.cpu().numpy())
        if diff.any():
            bad_safe += 1
            if first is None:
                ei, ai = np.unravel_index(int(np.flatnonzero(diff)[0]), diff.shape)
                first = ("安全层", t, int(ei), int(ai),
                         a.safe[ei].tolist(), b.safe[ei].cpu().numpy().tolist(),
                         a.area3[ei].tolist(), b.area3[ei].cpu().numpy().tolist(),
                         int(a.length[ei]), int(a.food[ei]), int(a.dir[ei]),
                         a.body_cells(int(ei))[:8], int(a.head_ptr[ei]))
        oa = a.observe()
        ob = b.observe().cpu().numpy()
        m_obs = max(m_obs, float(np.abs(oa - ob).max()))

        same = (np.array_equal(a.body, b.body.cpu().numpy())
                and np.array_equal(a.head_ptr, b.head_ptr.cpu().numpy())
                and np.array_equal(a.length, b.length.cpu().numpy())
                and np.array_equal(a.food, b.food.cpu().numpy())
                and np.array_equal(a.score, b.score.cpu().numpy())
                and np.array_equal(a.steps, b.steps.cpu().numpy())
                and np.array_equal(a.dir, b.dir.cpu().numpy())
                and np.array_equal(a.area, b.area.cpu().numpy()))
        if not same:
            bad_state += 1
            if first is None:
                first = ("状态", t, -1, "", "")

        _, ra, ta, tra, ia = a.step(acts[t])
        _, rb, tb, trb, ib = b.step(acts[t])
        m_rew = max(m_rew, float(np.abs(ra - rb.cpu().numpy()).max()))
        fin_a += int(ia["finished"].sum())
        fin_b += int(ib["finished"].sum())
        if (not np.array_equal(ta.cpu().numpy() if torch.is_tensor(ta) else ta, tb.cpu().numpy())
                or not np.array_equal(ia["cause"], ib["cause"])
                or not np.array_equal(ia["final_score"], ib["final_score"])
                or not np.array_equal(ia["final_steps"], ib["final_steps"])):
            bad_fin += 1
            if first is None:
                first = ("结局信息", t, -1, "", "")

    print(f"对拍 {steps} 步 × {N} 环境（同一随机流 + 同一动作序列）")
    print(f"[状态]   不一致步数 {bad_state}   （应为 0）")
    print(f"[安全层] 不一致步数 {bad_safe}   （应为 0）")
    print(f"[观测]   最大逐项偏差 {m_obs:.3e}   （float32 精度内）")
    print(f"[奖励]   最大逐项偏差 {m_rew:.3e}")
    print(f"[结局]   不一致步数 {bad_fin}   回合数 numpy {fin_a} / torch {fin_b}")
    if first:
        name, t, ei, ai = first[0], first[1], first[2], first[3]
        print(f"\n❌ 首个不一致: {name} @t={t} env={ei} action={ai}")
        if name == "安全层":
            print(f"   numpy safe {first[4]}   torch safe {first[5]}")
            print(f"   numpy area {first[6]}   torch area {first[7]}")
            print(f"   长度 {first[8]}  食物 {first[9]}  朝向 {first[10]}")
            print(f"   身体(头→尾前8): {first[11]}  head_ptr={first[12]}")
        return 1
    print("✅ 全部一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
