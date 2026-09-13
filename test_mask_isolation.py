"""
隔离测试：开/不开安全掩码，环境行为必须完全一致（掩码只是额外算出来的一份信息，
不允许对状态、观测、奖励产生任何副作用）。

同时输出掩码在不同蛇长下的宽松度，用来判断它会不会在早期就把动作压死。

用法： .venv/bin/python test_mask_isolation.py
"""

import numpy as np

from snake_env import VectorSnake

STEPS = 400
N = 8


def main():
    rng = np.random.default_rng(7)
    acts = rng.integers(0, 3, size=(STEPS, N))

    kwargs = dict(n_envs=N, seed=11, max_steps=10 ** 9, food_reward=1.0,
                  death_penalty=1.0, step_penalty=0.001, shaping_scale=0.25,
                  area_shaping=0.2, doomed_penalty=0.3)
    a = VectorSnake(**kwargs, safety_mask=False)
    b = VectorSnake(**kwargs, safety_mask=True)

    oa, ob = a.observe(), b.observe()
    m_obs = np.abs(oa - ob).max()
    m_rew = 0.0
    m_state = 0
    allowed_rows = []
    for t in range(STEPS):
        mask = b.safe.copy()
        allowed_rows.append((b.length.copy(), mask.sum(1)))
        _, ra, ta, tra, ia = a.step(acts[t])
        _, rb, tb, trb, ib = b.step(acts[t])
        m_obs = max(m_obs, float(np.abs(a.observe() - b.observe()).max()))
        m_rew = max(m_rew, float(np.abs(ra - rb).max()))
        if (not np.array_equal(a.body, b.body) or not np.array_equal(a.length, b.length)
                or not np.array_equal(a.food, b.food) or not np.array_equal(a.score, b.score)
                or not np.array_equal(a.head_ptr, b.head_ptr)
                or not np.array_equal(ta, tb)):
            m_state += 1

    print("=== 隔离测试（相同动作序列，400 步 × 8 环境）===")
    print(f"观测最大偏差      {m_obs:.3e}   （应为 0）")
    print(f"奖励最大偏差      {m_rew:.3e}   （应为 0）")
    print(f"状态不一致步数    {m_state}     （应为 0）")
    print(f"掩码本身是否被算出来: {'是' if b.safe.any() or (~b.safe).any() else '否'}")

    print("\n=== 掩码宽松度 vs 蛇长（随机动作策略）===")
    L = np.concatenate([r[0] for r in allowed_rows])
    A = np.concatenate([r[1] for r in allowed_rows])
    for lo, hi in [(3, 10), (11, 30), (31, 60), (61, 120), (121, 250), (251, 400)]:
        m = (L >= lo) & (L <= hi)
        if m.sum():
            print(f"  长度 {lo:>3}-{hi:<3}: 平均允许 {A[m].mean():.2f} 个动作   "
                  f"样本 {m.sum()}   全禁比例 {100*(A[m]==0).mean():.1f}%   "
                  f"只剩一个比例 {100*(A[m]==1).mean():.1f}%")


if __name__ == "__main__":
    main()
