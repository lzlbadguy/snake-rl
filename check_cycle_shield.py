"""
验证"环盾"的构造性保证：从可行动作里随便选，蛇到底会不会死？

理论断言（见 snake_env.py::_cycle_mask 的证明）：
  ① 初始状态满足不变量：所有身体格的回路序号都在 [T, H] 这段前向弧上；
  ② 该不变量被"可行动作"保持；
  ③ 可行动作集**永不为空**（沿回路前进一步永远可行）。
  ⇒ 从可行动作里选，蛇结构上不可能死（不撞墙、不撞自己）。

本脚本做三件事：
  A. 逐步检查不变量（不只看死没死，而是每一步验证"所有身体格仍在弧上"）
  B. 用**均匀随机**策略跑很久 —— 如果它会死，理论就是错的
  C. 用"贪心缩短到食物的回路距离"策略跑，看**吃满需要多少步**（对比纯 Hamilton 的 39,802）

用法： .venv/bin/python check_cycle_shield.py [步数] [环境数]
"""

import sys
import time

import numpy as np

from snake_env import VectorSnake


def invariant_violations(env):
    """违反不变量的身体格数：身体格相对**弧起点 T** 的前向距离必须 <= 头的前向距离。"""
    ar = env._ar
    head = env.body[ar, env.head_ptr]
    H = env.cyc_idx[head]
    T = env.arc_T
    dTH = (H - T) % env.cyc_n
    bad = 0
    for k in range(int(env.length.max())):
        seg = env.body[ar, (env.head_ptr - k) % env.max_len]
        valid = k < env.length
        d = (env.cyc_idx[seg] - T) % env.cyc_n
        bad += int(((d > dTH) & valid).sum())
    return bad


def run(steps, n_envs, greedy, seed=0, verbose_every=0, mode="arc", cap=80000,
        check_inv=True):
    env = VectorSnake(n_envs=n_envs, seed=seed, safety_mask=True, shield_mode=mode,
                      max_steps=cap, food_reward=1.0, death_penalty=1.0, step_penalty=0.001,
                      shaping_scale=0.0)
    obs = env.observe()
    rng = np.random.default_rng(seed)
    ar = env._ar
    t0 = time.time()
    deaths = 0
    inv_bad = 0
    tier_hist = np.zeros(3, dtype=np.int64)
    filled = 0
    for t in range(steps):
        mask = env.safe
        tier_hist += np.bincount(env.tier, minlength=3)[:3]
        if greedy:
            head = env.body[ar, env.head_ptr]
            T = env.cyc_idx[env.body[ar, (env.head_ptr - env.length + 1) % env.max_len]]
            hy, hx = head // env.cols, head % env.cols
            keys = np.zeros((n_envs, 3), dtype=np.float64)
            for a in range(3):
                from snake_env import TURN, DY, DX
                d = ((env.dir.astype(np.int16) + TURN[a]) % 4).astype(np.int16)
                ny = np.clip(hy + DY[d], 0, env.rows - 1)
                nx = np.clip(hx + DX[d], 0, env.cols - 1)
                cell = (ny * env.cols + nx).astype(np.int32)
                dd = (env.cyc_idx[cell] - env.cyc_idx[env.food]) % env.cyc_n
                keys[:, a] = np.where(mask[:, a], dd, -1e9)   # 最大化"候选格到食物的前向距离"= 尽量靠近食物
            act = keys.argmax(axis=1)
        else:
            act = np.array([rng.choice(np.flatnonzero(mask[i])) if mask[i].any() else 0
                            for i in range(n_envs)])
        obs, rew, term, trunc, info = env.step(act)
        bad = invariant_violations(env) if check_inv else 0
        inv_bad += bad
        fin = int(info["finished"].sum())
        if fin:
            deaths += fin
            filled += int((info["final_score"][np.flatnonzero(info["finished"])] >= 397).sum())
            if verbose_every and deaths % verbose_every == 0:
                print(f"    t={t} 死亡/结束 {deaths} 次，最高分 {env.score.max()}", flush=True)
        if bad:
            print(f"❌ 不变量在第 {t} 步被破坏（{bad} 处）—— 理论有误！")
            return dict(ok=False, t=t)
    dt = time.time() - t0
    return dict(ok=True, steps=steps, deaths=deaths, inv_bad=inv_bad, filled=filled,
                tier=tier_hist.tolist(), score_max=int(env.score.max()),
                score_mean=float(env.score.mean()), sps=n_envs * steps / dt)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    mode = sys.argv[3] if len(sys.argv) > 3 else "arc"
    phase = sys.argv[4] if len(sys.argv) > 4 else "both"

    if phase != "greedy_only":
        print(f"=== A/B: 均匀随机策略（模式 {mode}，只要不死就说明不变量确实被保持）===")
        r = run(steps, n_envs, greedy=False, mode=mode)
        if not r["ok"]:
            return 1
        tt = max(1, sum(r["tier"]))
        print(f"  步数 {r['steps'] * n_envs:,} | 结束回合 {r['deaths']} 次 | 不变量违反 {r['inv_bad']} 处")
        print(f"  盾级别分布: 一级 {r['tier'][0]} / 二级 {r['tier'][1]} / 死局 {r['tier'][2]}"
              f"  → 可行动作集为空的比例 {100*r['tier'][2]/tt:.4f}%")
        print(f"  随机游走的分数: 均 {r['score_mean']:.1f} 最高 {r['score_max']}  （吃满 {r['filled']} 次）")
        print(f"  吞吐 {r['sps']:,.0f} 步/秒")

    print("\n=== C: 贪心（在可行动作里选最靠近食物的）—— 测吃满 ===")
    r2 = run(steps, n_envs, greedy=True, mode=mode,
             check_inv=(phase != "greedy_only"))
    print(f"  步数 {r2['steps'] * n_envs:,} | 结束回合 {r2['deaths']} 次 | 不变量违反 {r2['inv_bad']} 处")
    print(f"  分数: 均 {r2['score_mean']:.1f} 最高 {r2['score_max']}  吃满(397) {r2['filled']} 次")
    print(f"  吞吐 {r2['sps']:,.0f} 步/秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
