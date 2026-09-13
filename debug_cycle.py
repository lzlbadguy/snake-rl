"""
定位环盾不变量的第一次破坏：把违规前后的完整状态打出来。

不变量：所有身体格的回路序号（相对尾巴 T 的前向距离）都必须 <= 头的前向距离 dTH。
违规意味着某一步走完之后，身体在回路上"交叉"了 —— 说明可行动作的判据不够强。
"""

import numpy as np

from snake_env import VectorSnake


def dump(env, i):
    head = int(env.body[i, env.head_ptr[i]])
    L = int(env.length[i])
    tail = int(env.body[i, (env.head_ptr[i] - L + 1) % env.max_len])
    H = int(env.cyc_idx[head])
    T = int(env.cyc_idx[tail])
    path = [int(env.body[i, (env.head_ptr[i] - k) % env.max_len]) for k in range(L)]
    ds = [int((int(env.cyc_idx[c]) - T) % env.cyc_n) for c in path]
    return dict(head=head, tail=tail, H=H, T=T, dTH=int((H - T) % env.cyc_n), L=L,
                path=path, ds=ds, dir=int(env.dir[i]), steps=int(env.steps[i]),
                score=int(env.score[i]))


def viols(d):
    return [k for k, v in enumerate(d["ds"]) if v > d["dTH"]]


def main():
    env = VectorSnake(n_envs=64, seed=0, safety_mask=True, shield_mode="cycle",
                      shaping_scale=0.0)
    env.observe()
    rng = np.random.default_rng(0)
    n = env.n
    for t in range(20000):
        prev = [dump(env, i) for i in range(n)]
        prev_mask = env.safe.copy()
        act = np.array([rng.choice(np.flatnonzero(env.safe[i])) if env.safe[i].any() else 0
                        for i in range(n)])
        env.step(act)
        for i in range(n):
            d = dump(env, i)
            v = viols(d)
            if v and d["steps"] > 1:          # steps<=1 是刚重开的回合，跳过
                print(f"❌ 首个破坏: t={t} env={i}")
                p = prev[i]
                print(f"  走之前: 头={p['head']} 尾={p['tail']} H={p['H']} T={p['T']} "
                      f"dTH={p['dTH']} L={p['L']} dir={p['dir']} 分={p['score']}")
                print(f"          身体路径 {p['path']}  序号距离 {p['ds']}")
                print(f"          盾 {prev_mask[i].tolist()}  选的动作 {int(act[i])}")
                print(f"  走之后: 头={d['head']} 尾={d['tail']} H={d['H']} T={d['T']} "
                      f"dTH={d['dTH']} L={d['L']} dir={d['dir']}")
                print(f"          身体路径 {d['path']}  序号距离 {d['ds']}")
                print(f"          违规位置 {v}（这些格子的前向距离 > 头的）")
                return 0
    print("✅ 20000 步没有破坏")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
