"""
诊断 2：给"自撞死亡"找真正有前瞻性的安全信号。

要设计奖励，先要回答：死亡前那一刻，盘面到底长什么样？
本脚本在当前策略的贪心轨迹上，逐局记录：

  A. area/length（可达格数 / 身体长度）在死前 1/5/10/20/40 步的值
     — 若死亡时该比值已经接近 1，说明"空间不足"类奖励有用；若仍很大，说明没用
  B. tail_reachable：走完这一步后，头还能不能到达尾巴那一格（把尾巴当可通行，因为它下一步会让开）
     — 这是 BFS 类贪吃蛇 AI 的标准安全检查
  C. 上述信号"首次触发 → 死亡"的提前量，以及"触发后还活了很久"的误报率

用法： .venv/bin/python check_safety.py [局数]
"""

import sys
import numpy as np
import torch

from snake_env import VectorSnake, N_ACTIONS, CAUSE_SELF, DY, DX
from train import ActorCritic

WINDOW = 64          # 保存每局最近 WINDOW 步的指标


def tail_reachable(env):
    """走完当前状态后，头能否到达尾巴格（尾巴格视为可通行）。向量化 BFS。"""
    n = env.n
    ar = env._ar
    head = env.body[ar, env.head_ptr]
    tail = env.body[ar, (env.head_ptr - (env.length - 1)) % env.max_len]
    hy, hx = head // env.cols, head % env.cols
    free = (~env.occ).reshape(n, env.rows, env.cols).copy()
    free[ar, hy, hx] = True
    ty, tx = tail // env.cols, tail % env.cols
    free[ar, ty, tx] = True                     # 尾巴下一步会腾空
    vis = np.zeros_like(free)
    fr = np.zeros_like(free)
    fr[ar, hy, hx] = True
    vis |= fr
    while fr.any():
        nb = np.zeros_like(fr)
        nb[:, 1:, :] |= fr[:, :-1, :]
        nb[:, :-1, :] |= fr[:, 1:, :]
        nb[:, :, 1:] |= fr[:, :, :-1]
        nb[:, :, :-1] |= fr[:, :, 1:]
        nb &= free
        nb &= ~vis
        if not nb.any():
            break
        vis |= nb
        fr = nb
    return vis[ar, ty, tx]


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ck = torch.load("runs/ppo20/ckpt.pt", map_location="cpu", weights_only=True)
    args = ck.get("args", {})
    n_env = 64
    env = VectorSnake(n_envs=n_env, seed=321, use_patch=not args.get("no_patch", False),
                      food_reward=0.0, death_penalty=0.0, step_penalty=0.0, shaping_scale=0.0)
    model = ActorCritic(env.obs_dim, N_ACTIONS, args.get("hidden", 256))
    model.load_state_dict(ck["model"])
    model.eval()

    obs = env.observe()
    tstep = np.zeros(n_env, dtype=np.int64)
    ratio = np.zeros((n_env, WINDOW), dtype=np.float32)   # 环形缓冲：area/length
    tailok = np.zeros((n_env, WINDOW), dtype=bool)
    fire = np.full(n_env, -1, dtype=np.int64)             # 本局 tail 检查首次报"不可达"的步号

    done = 0
    at_death = {k: [] for k in (1, 5, 10, 20, 40)}
    tail_fire_lead, tail_fire_survived = [], []
    self_deaths = 0
    while done < n_target:
        with torch.no_grad():
            act = model(torch.from_numpy(obs))[0].argmax(-1).numpy()
        obs, _, _, _, info = env.step(act)
        tstep += 1
        i = env._ar
        r = (env.area / np.maximum(env.length, 1)).astype(np.float32)
        ok = tail_reachable(env)
        ratio[i, tstep % WINDOW] = r
        tailok[i, tstep % WINDOW] = ok
        newly = (~ok) & (fire < 0)
        fire[newly] = tstep[newly]

        for e in np.flatnonzero(info["finished"]):
            done += 1
            t = int(tstep[e])
            if int(info["cause"][e]) == CAUSE_SELF:
                self_deaths += 1
                for k in at_death:
                    if t - k >= 0:
                        v = ratio[e, (t - k) % WINDOW]
                        if v > 0:
                            at_death[k].append(float(v))
            if fire[e] >= 0:
                lead = t - int(fire[e])
                tail_fire_lead.append(lead)
                tail_fire_survived.append(lead)   # 触发后还能活多少步 = 误报代价
            tstep[e] = 0
            fire[e] = -1

    print(f"政策 ckpt update={ck.get('update')}  共 {done} 局，其中自撞死亡 {self_deaths} 局")
    print("\n[A] 自撞死亡前 area/length 的取值（越大说明死亡时空间越充裕 → 空间类奖励越无用）")
    for k in (1, 5, 10, 20, 40):
        v = np.array(at_death[k])
        if len(v):
            print(f"   死前 {k:2d} 步: 均值 {v.mean():5.2f}  中位 {np.median(v):5.2f}  "
                  f"10%分位 {np.percentile(v,10):5.2f}  最小 {v.min():.2f}")

    print("\n[B][C] tail_reachable（走完后头是否还能到达尾巴）")
    if tail_fire_lead:
        lead = np.array(tail_fire_lead)
        print(f"   触发『不可达』的回合: {len(lead)}/{done} = {100*len(lead)/done:.1f}%")
        print(f"   触发 → 死亡 的提前量: 中位 {np.median(lead):.0f} 步  均值 {lead.mean():.1f} 步")
        for lo, hi in [(0, 2), (3, 10), (11, 25), (26, 60), (61, 10**9)]:
            m = (lead >= lo) & (lead <= hi)
            label = f"{lo}-{hi}" if hi < 10**9 else f">{lo}"
            print(f"      {label:<8} {100*m.mean():5.1f}%")
        long_alive = (lead > 25).mean()
        print(f"   误报（触发后还活过 25 步以上）: {100*long_alive:.1f}%")
    else:
        print("   从未触发 —— 头一直能到达尾巴")


if __name__ == "__main__":
    main()
