"""
A 部分的核心证据：Hamilton 回路跟随器 —— 保证把 20x20 棋盘吃满。

回路构造（偶数网格，2 行条带 + 第 0 列回收通道）：
    对每个条带 k（行 2k, 2k+1）：沿列 1..C-1 从左到右走行 2k，
                                再从右到左走行 2k+1，末端落在 (1, 2k+1)
    条带之间 (1,2k+1) → (1,2k+2) 相邻；最后一条带结束后
    从 (1,R-1) 拐进第 0 列，沿第 0 列上行回到 (0,0)，闭合到起点 (1,0)

为什么用这个构造：它的偶数行是"从左到右"遍历的，所以行 10 上天然存在
(3,10) → (4,10) → (5,10) 这个顺序三元组 —— 而游戏初始身体恰好是
头 (5,10)、身 (4,10)、(3,10)。也就是说蛇**一开局就是回路的对齐状态**，
不需要额外的对齐阶段（这是 Hamilton 类策略最容易踩的坑）。

跟随者永不死亡：身体始终占据回路上"头之后的一段弧"，下一步永远是弧外格。

用法： .venv/bin/python hamilton.py [局数]
"""

import sys

import numpy as np

from snake_env import VectorSnake, TURN, DY, DX, CAUSE_CAP, CAUSE_WIN


def build_cycle(rows=20, cols=20):
    """返回回路顺序的 (x, y) 列表（长度 = rows*cols）。"""
    assert rows % 2 == 0, "需要偶数行"
    cyc = []
    for k in range(rows // 2):
        y0, y1 = 2 * k, 2 * k + 1
        for x in range(1, cols):
            cyc.append((x, y0))
        for x in range(cols - 1, 0, -1):
            cyc.append((x, y1))
    for y in range(rows - 1, -1, -1):
        cyc.append((0, y))
    return cyc


def verify_cycle(cyc, rows, cols):
    """校验：恰好覆盖所有格子，且相邻（首尾也相邻）。"""
    cells = set(cyc)
    assert len(cyc) == rows * cols, f"长度 {len(cyc)} != {rows*cols}"
    assert len(cells) == len(cyc), "有重复格子"
    assert cells == {(x, y) for x in range(cols) for y in range(rows)}, "没覆盖全"
    for i in range(len(cyc)):
        (x1, y1) = cyc[i]
        (x2, y2) = cyc[(i + 1) % len(cyc)]
        assert abs(x1 - x2) + abs(y1 - y2) == 1, f"第 {i} 步不相邻: {(x1,y1)} -> {(x2,y2)}"


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    rows = cols = 20
    cyc = build_cycle(rows, cols)
    verify_cycle(cyc, rows, cols)
    print(f"Hamilton 回路校验通过：{len(cyc)} 格全覆盖且首尾相邻")

    # 回路顺序 -> 下一格 cell 索引
    cells = rows * cols
    nxt = np.zeros(cells, dtype=np.int32)
    for i, (x, y) in enumerate(cyc):
        nx, ny = cyc[(i + 1) % len(cyc)]
        nxt[y * cols + x] = ny * cols + nx

    # 动作查表：当前朝向 d、目标朝向 t -> 相对动作
    action_table = np.zeros((4, 4), dtype=np.int64)
    for d in range(4):
        for t in range(4):
            r = (t - d) % 4
            action_table[d, t] = {0: 0, 3: 1, 1: 2}.get(r, 0)

    env = VectorSnake(n_envs=64, seed=3, rows=rows, cols=cols, max_steps=400000,
                      food_reward=0.0, death_penalty=0.0, step_penalty=0.0, shaping_scale=0.0)
    n = env.n
    ar = env._ar

    # ---- 关键校验：初始身体必须是回路上的连续弧（顺序为 头 -> 尾） ----
    pos = {c: i for i, c in enumerate(cyc)}
    head_cell = int(env.body[0, env.head_ptr[0]])
    hx, hy = head_cell % cols, head_cell // cols
    body_order = [cyc[(pos[(hx, hy)] - k) % len(cyc)] for k in range(int(env.length[0]))]
    expect = [(int(env.body[0, (env.head_ptr[0] - k) % env.max_len]) % cols,
               int(env.body[0, (env.head_ptr[0] - k) % env.max_len]) // cols)
              for k in range(int(env.length[0]))]
    aligned = body_order == expect
    print(f"初始身体与回路对齐: {aligned}   身体 {expect}")
    if not aligned:
        print("  ⚠ 未对齐 —— 跟随者可能早期就撞到自己")

    scores, causes, steps_used, lens = [], [], [], []
    max_loop = 400000
    for _ in range(max_loop):
        if len(scores) >= episodes:
            break
        head = env.body[ar, env.head_ptr]
        tgt = nxt[head]
        hy_ = head // cols
        hx_ = head % cols
        ty_ = tgt // cols
        tx_ = tgt % cols
        # 目标朝向
        tdir = np.where(ty_ < hy_, 0, np.where(tx_ > hx_, 1, np.where(ty_ > hy_, 2, 3)))
        act = action_table[env.dir, tdir.astype(np.int64)]
        _, _, _, _, info = env.step(act)
        for i in np.flatnonzero(info["finished"]):
            scores.append(int(info["final_score"][i]))
            causes.append(int(info["cause"][i]))
            steps_used.append(int(info["final_steps"][i]))
            lens.append(int(info["final_len"][i]))

    s = np.array(scores)
    c = np.array(causes)
    st = np.array(steps_used)
    FULL = rows * cols - 3
    print(f"\nHamilton 跟随器：{len(s)} 局，满分 {FULL}")
    print(f"  吃满率: {100*(s==FULL).mean():.1f}%   均分 {s.mean():.1f}   最低 {s.min()}")
    print(f"  吃满步数: 均值 {st.mean():.0f}  中位 {np.median(st):.0f}  最少 {st.min()}  "
          f"最多 {st.max()}")
    print(f"  死因: 撞墙 {100*(c==0).mean():.1f}%  自撞 {100*(c==1).mean():.1f}%  "
          f"步数上限 {100*(c==CAUSE_CAP).mean():.1f}%  通关 {100*(c==CAUSE_WIN).mean():.1f}%")
    print(f"  效率: 每颗食物 {st.mean()/max(s.mean(),1):.1f} 步（理论下界 1.0，"
          f"沿固定回路绕行 ≈ 盘面一半 = {0.5*rows*cols:.0f} 步）")


if __name__ == "__main__":
    main()
