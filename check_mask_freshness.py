"""
掩码新鲜度检查：env.safe 是不是用"当前状态"算的？

env.safe 在 observe() 里算，而 observe() 在 step() 结尾被调用。如果调用顺序错了
（比如在更新身体/指針之前算），那么下一步读到的掩码其实是**上一步状态**的掩码 ——
这会让盾整体错位一拍，并解释"本该被挡的动作却放行了"。
"""

import numpy as np

from snake_env import VectorSnake


def main():
    for mode in ("cycle", "tail"):
        env = VectorSnake(n_envs=8, seed=0, safety_mask=True, shield_mode=mode,
                          shaping_scale=0.0)
        env.observe()
        rng = np.random.default_rng(0)
        bad = 0
        first = None
        for t in range(600):
            stored = env.safe.copy()
            if mode == "cycle":
                fresh, _ = env._cycle_mask()
            else:
                st, al, _ = env.safety_flags()
                fresh, _ = env._shield_tiers(st, al)
            if not np.array_equal(stored, fresh):
                bad += 1
                if first is None:
                    first = (t, stored.tolist(), fresh.tolist())
            env.step(rng.integers(0, 3, size=env.n))
        print(f"[{mode}] 掩码与当前状态不一致: {bad} / 600 步")
        if first:
            print(f"   首次 @t={first[0]}\n   存着的 {first[1][0]}\n   重算的 {first[2][0]}")
        else:
            print("   ✅ 掩码始终对应当前状态")


if __name__ == "__main__":
    main()
