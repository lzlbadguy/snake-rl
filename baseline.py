"""
启发式基线：BFS 追食物 + 尾巴可达性安全检查 + 可达面积兜底（完全不需要训练）。

目的：回答"20x20 到底能不能吃满"—— 先看非学习方法的成功率与效率，
再判断"纯 RL 吃满"这个目标是否现实。

每步决策（对 3 个候选动作各做一次 BFS）：
  1. 剔除会撞墙 / 撞身体的动作
  2. 对存活动作，模拟走完后的占据图（尾巴在非进食时会让位），从新头位置做一趟 BFS：
     - dist  → 到食物的最短距离
     - vis   → 可达区域（面积），以及"头能否到达尾巴"（安全检查）
  3. 排序：安全的优先（尾巴可达 或 可达面积 ≥ 身体长度）；
     安全动作里挑离食物最近的；并列或都不安全时挑可达面积最大的

用法： .venv/bin/python baseline.py [局数] [并行环境数]
"""

import sys

import numpy as np

from snake_env import VectorSnake, TURN, DY, DX, CAUSE_WALL, CAUSE_SELF, CAUSE_CAP, CAUSE_WIN

# 吃满一盘需要上千步，回合上限必须放开（训练时的 1600 会从构造上封死上限）
MAX_STEPS = 20000


def bfs_field(occ, src, cols, rows):
    """从 src(每环境一个 cell) 在 occ 为障碍的网格上做 BFS。
    返回 (dist (n,R,C) int16, vis (n,R,C) bool)。occ 中起点需为可通行。"""
    n = occ.shape[0]
    ar = np.arange(n)
    sy, sx = src // cols, src % cols
    vis = np.zeros_like(occ)
    fr = np.zeros_like(occ)
    fr[ar, sy, sx] = True
    vis |= fr
    dist = np.zeros((n, rows, cols), dtype=np.int16)
    d = 0
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
        d += 1
        dist[nb] = d
        vis |= nb
        fr = nb
    return dist, vis


def heuristic_actions(env):
    n, cols, rows = env.n, env.cols, env.rows
    ar = env._ar
    head = env.body[ar, env.head_ptr]
    hy, hx = head // cols, head % cols
    tail = env.body[ar, (env.head_ptr - (env.length - 1)) % env.max_len]
    ty, tx = tail // cols, tail % cols
    occ_now = env.occ.reshape(n, rows, cols)
    hasf = env.food >= 0
    fy = np.where(hasf, env.food // cols, 0)
    fx = np.where(hasf, env.food % cols, 0)

    keys = np.full((n, 3), -1e9, dtype=np.float64)
    # 身体线段：0 = 头，L-1 = 尾
    def seg(k):
        return env.body[ar, (env.head_ptr - k) % env.max_len]

    tail_now = seg(env.length - 1)                       # 当前尾巴
    tail_after_move = seg(np.maximum(env.length - 2, 0))  # 不吃食物时，新尾巴是倒数第二节
    for a in range(3):
        d = ((env.dir.astype(np.int16) + TURN[a]) % 4).astype(np.int16)
        ny = hy + DY[d]
        nx = hx + DX[d]
        out = (ny < 0) | (ny >= rows) | (nx < 0) | (nx >= cols)
        nyc = np.clip(ny, 0, rows - 1)
        nxc = np.clip(nx, 0, cols - 1)
        ncell = (nyc * cols + nxc).astype(np.int32)
        ate = ncell == env.food
        vac = ~ate
        hit_self = env.occ[ar, ncell] & ~((ncell == tail_now) & vac)
        alive = ~(out | hit_self)

        # 关键：吃食物那一步蛇变长、尾巴**不让位**，安全检查必须按这个正确模拟
        target_tail = np.where(ate, tail_now, tail_after_move)
        length_after = env.length + ate.astype(np.int32)

        occ = occ_now.copy()
        occ[ar[vac], tail_now[vac] // cols, tail_now[vac] % cols] = False  # 尾巴让位
        occ[ar, nyc, nxc] = False                                          # 新头所在格可站
        dist, vis = bfs_field(occ, ncell, cols, rows)

        area = vis.reshape(n, -1).sum(1).astype(np.float64)
        reached = np.zeros(n, dtype=bool)
        dd = np.full(n, 10 ** 4, dtype=np.float64)
        if hasf.any():
            idx = np.flatnonzero(hasf)
            reached[idx] = vis[idx, fy[idx], fx[idx]]
            dd[idx] = np.where(vis[idx, fy[idx], fx[idx]], dist[idx, fy[idx], fx[idx]], 10 ** 4)
        t2y, t2x = target_tail // cols, target_tail % cols
        tail_ok = vis[ar, t2y, t2x]
        tail_dist = np.where(tail_ok, dist[ar, t2y, t2x].astype(np.float64), 10 ** 4)

        # 字典序优先级：
        #   1) 能安全地去吃食物（活着 + 尾巴仍可达 + 食物可达）→ 挑最近的
        #   2) 其余安全动作 → 跟随尾巴（离尾巴越近越稳，经典 tail-chase）
        #   3) 都不安全 → 挑可达面积最大的
        safe = alive & (tail_ok | (area >= length_after))
        food_ok = safe & reached
        keys[:, a] = np.where(
            food_ok, 3e6 - dd,
            np.where(safe, 2e6 - tail_dist,
                     np.where(alive, 1e6 + area, area * 1e-3)))

    return keys.argmax(axis=1).astype(np.int64)


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_env = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    env = VectorSnake(n_envs=n_env, seed=2024, max_steps=MAX_STEPS,
                      food_reward=0.0, death_penalty=0.0, step_penalty=0.0, shaping_scale=0.0)
    scores, causes, lens, steps_used = [], [], [], []
    while len(scores) < episodes:
        act = heuristic_actions(env)
        _, _, _, _, info = env.step(act)
        for i in np.flatnonzero(info["finished"]):
            scores.append(int(info["final_score"][i]))
            causes.append(int(info["cause"][i]))
            lens.append(int(info["final_len"][i]))
            steps_used.append(int(info["final_steps"][i]))

    s = np.array(scores[:episodes])
    c = np.array(causes[:episodes])
    st = np.array(steps_used[:episodes])
    FULL = 397
    print(f"启发式基线（BFS 追食物 + 尾巴可达性安全检查），{episodes} 局，回合上限 {MAX_STEPS} 步")
    print(f"  均分 {s.mean():.1f}   中位 {np.median(s):.0f}   最高 {s.max()}   (满分 {FULL})")
    print(f"  分数分位: p10 {np.percentile(s,10):.0f}  p50 {np.median(s):.0f} "
          f"p90 {np.percentile(s,90):.0f}  p99 {np.percentile(s,99):.0f}")
    print(f"  吃满率：==397: {100*(s==FULL).mean():.1f}%   ≥380: {100*(s>=380).mean():.1f}%   "
          f"≥300: {100*(s>=300).mean():.1f}%   ≥150: {100*(s>=150).mean():.1f}%")
    print(f"  死因: 撞墙 {100*(c==CAUSE_WALL).mean():.1f}%  自撞 {100*(c==CAUSE_SELF).mean():.1f}%  "
          f"撞步数上限 {100*(c==CAUSE_CAP).mean():.1f}%")
    print(f"  每局步数: 均值 {st.mean():.0f}  中位 {np.median(st):.0f}  最大 {st.max()}")
    print(f"  最长的 5 局分数: {sorted(s, reverse=True)[:5]}")


if __name__ == "__main__":
    main()
