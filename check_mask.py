"""
校验安全掩码：拿"可证明安全"的 Hamilton 回路动作去撞掩码。

Hamilton 跟随者的每一步在数学上都是安全的（身体始终占回路上头之后的一段弧），
所以掩码**绝不应该**挡掉它的动作。一旦挡掉，就说明 safety_flags() 有 bug。

顺带输出掩码的统计（每步允许几个动作），用来判断掩码是不是过严。

用法： .venv/bin/python check_mask.py
"""

import numpy as np

from snake_env import VectorSnake
from hamilton import build_cycle, verify_cycle


def main():
    rows = cols = 20
    cyc = build_cycle(rows, cols)
    verify_cycle(cyc, rows, cols)
    nxt = np.zeros(rows * cols, dtype=np.int32)
    for i, (x, y) in enumerate(cyc):
        nx, ny = cyc[(i + 1) % len(cyc)]
        nxt[y * cols + x] = ny * cols + nx
    action_table = np.zeros((4, 4), dtype=np.int64)
    for d in range(4):
        for t in range(4):
            action_table[d, t] = {0: 0, 3: 1, 1: 2}.get((t - d) % 4, 0)

    env = VectorSnake(n_envs=64, rows=rows, cols=cols, safety_mask=True,
                      max_steps=400000, food_reward=0.0, death_penalty=0.0,
                      step_penalty=0.0, shaping_scale=0.0)
    n, ar = env.n, env._ar
    env.observe()                       # 先算一次掩码

    blocked = tot = 0
    blocked_strict = 0
    hist = np.zeros(4, dtype=np.int64)
    tier_hist = np.zeros(3, dtype=np.int64)
    first_block = None
    wins = 0
    for t in range(60000):
        mask = env.safe                            # 实际生效的盾（两级）
        strict, alive, _ = env.safety_flags()      # 严格一级（诊断用）
        tier_hist += np.bincount(env.tier, minlength=3)[:3]
        head = env.body[ar, env.head_ptr]
        hy, hx = head // cols, head % cols
        tgt = nxt[head]
        ty, tx = tgt // cols, tgt % cols
        tdir = np.where(ty < hy, 0, np.where(tx > hx, 1, np.where(ty > hy, 2, 3)))
        act = action_table[env.dir, tdir.astype(np.int64)]

        cnt = mask.sum(1)
        hist += np.bincount(cnt, minlength=4)[:4]
        bad = ~mask[ar, act]
        blocked += int(bad.sum())
        blocked_strict += int((~strict[ar, act]).sum())
        tot += n
        if bad.any() and first_block is None:
            i = int(np.flatnonzero(bad)[0])
            first_block = (t, i, int(env.length[i]), int(env.score[i]), mask[i].tolist())
        _, _, term, trunc, info = env.step(act)
        wins += int(info["finished"].sum())
        if t > 30000 and wins >= n:
            break

    print(f"Hamilton 动作被盾挡掉: 两级盾 {blocked} / {tot} 步 = {100*blocked/max(1,tot):.4f}%"
          f"   （严格一级 {blocked_strict} / {tot} = {100*blocked_strict/max(1,tot):.4f}%）")
    tt = max(1, int(tier_hist.sum()))
    print(f"盾级别分布: 严格 {tier_hist[0]} 步 / 降级 {tier_hist[1]} / 死局 {tier_hist[2]}"
          f"  → 需要降级或死局的比例 {100*(tier_hist[1]+tier_hist[2])/tt:.2f}%")
    if first_block:
        print(f"  首次被挡: 步 {first_block[0]} env {first_block[1]} 长度 {first_block[2]} "
              f"分数 {first_block[3]} 掩码 {first_block[4]}")
    print(f"每步允许的动作数分布: 0个 {hist[0]} 次, 1个 {hist[1]} 次, "
          f"2个 {hist[2]} 次, 3个 {hist[3]} 次")
    print(f"  平均允许动作数 {np.dot(hist, np.arange(4))/max(1,hist.sum()):.2f} "
          f"（3.00 表示掩码从不生效，1.00 表示只剩一个选择）")


if __name__ == "__main__":
    main()
