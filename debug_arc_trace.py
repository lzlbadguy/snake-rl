"""打印弧盾下贪心策略的一小段轨迹，看它卡在哪。"""

import numpy as np

from snake_env import VectorSnake, TURN, DY, DX

env = VectorSnake(n_envs=1, seed=3, safety_mask=True, shield_mode="arc", arc_slack=8,
                  max_steps=100000, shaping_scale=0.0)
env.observe()
rng = np.random.default_rng(3)
n = env.n
ar = env._ar

print(f"{'步':>4} {'头':>4} {'T':>4} {'H':>4} {'dTH':>4} {'食物':>4} {'盾':>15} {'动作':>4} {'分':>3}")
for t in range(400):
    head = env.body[ar, env.head_ptr]
    T = env.arc_T
    H = env.cyc_idx[head]
    dTH = (H - T) % env.cyc_n
    mask = env.safe
    hy, hx = head // env.cols, head % env.cols
    keys = np.zeros((n, 3))
    for a in range(3):
        d = ((env.dir.astype(np.int16) + TURN[a]) % 4).astype(np.int16)
        ny = np.clip(hy + DY[d], 0, env.rows - 1)
        nx = np.clip(hx + DX[d], 0, env.cols - 1)
        cell = (ny * env.cols + nx).astype(np.int32)
        dd = (env.cyc_idx[cell] - env.cyc_idx[env.food]) % env.cyc_n
        keys[:, a] = np.where(mask[:, a], dd, -1e9)
    act = keys.argmax(axis=1)
    if t < 40 or t % 25 == 0:
        print(f"{t:>4} {int(head[0]):>4} {int(T[0]):>4} {int(H[0]):>4} {int(dTH[0]):>4} "
              f"{int(env.food[0]):>4} {str(mask[0].tolist()):>15} {int(act[0]):>4} "
              f"{int(env.score[0]):>3}")
    env.step(act)
