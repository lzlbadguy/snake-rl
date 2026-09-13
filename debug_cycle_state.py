"""
手工重建 debug_cycle 里那条"破坏不变量"的状态，逐动作核对盾到底算了什么。

状态：身体路径（头→尾）= [182, 202, 201, 181]，朝向 0（上）
  182 = (x=2,y=9) 头
  202 = (x=2,y=10)
  201 = (x=1,y=10)
  181 = (x=1,y=9) 尾
"""

import numpy as np

from snake_env import VectorSnake, TURN, DY, DX

path = [182, 202, 201, 181]
env = VectorSnake(n_envs=1, seed=0, safety_mask=True, shield_mode="cycle", shaping_scale=0.0)
env.length[:] = len(path)
env.head_ptr[:] = 0
for k, c in enumerate(path):
    env.body[0, (-k) % env.max_len] = c
env.occ[:] = False
for c in path:
    env.occ[0, c] = True
env.dir[:] = 0
env.food[:] = 100

print(f"max_len={env.max_len}  cyc_n={env.cyc_n}")
print(f"cyc_idx[182]={env.cyc_idx[182]}  cyc_idx[183]={env.cyc_idx[183]}  "
      f"cyc_idx[181]={env.cyc_idx[181]}  cyc_idx[201]={env.cyc_idx[201]}  "
      f"cyc_idx[202]={env.cyc_idx[202]}")

head_v = int(env.body[0, env.head_ptr[0]])
tail_v = int(env.body[0, (env.head_ptr[0] - env.length[0] + 1) % env.max_len])
H = int(env.cyc_idx[head_v])
T = int(env.cyc_idx[tail_v])
dTH = (H - T) % env.cyc_n
print(f"\n盾读到的: 头格={head_v}(序号{H}) 尾格={tail_v}(序号{T})  dTH={dTH}")
print("预期:     头格=182(序号188) 尾格=181(序号189)  dTH=399")

m, tier = env._cycle_mask()
print(f"\n盾输出 mask={m[0].tolist()} tier={int(tier[0])}")

hy, hx = 9, 2
print("\n逐动作核对:")
for a in range(3):
    d = int((0 + TURN[a]) % 4)
    ny = hy + int(DY[d])
    nx = hx + int(DX[d])
    if not (0 <= ny < 20 and 0 <= nx < 20):
        print(f"  a={a} (转成方向{d}) → 出界")
        continue
    cell = ny * 20 + nx
    idx = int(env.cyc_idx[cell])
    dTq = (idx - T) % env.cyc_n
    alive = not bool(env.occ[0, cell])
    print(f"  a={a} (转成方向{d}) → 格{cell} 序号{idx} dTq={dTq} 活着={alive} "
          f"→ 判据 dTq>=dTH: {dTq >= dTH}   (盾里={bool(m[0, a])})")
