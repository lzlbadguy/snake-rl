"""
JS ↔ Python 对拍：验证浏览器里那份策略实现（观测构造 + MLP 前向）与 Python 训练时
用的是同一个东西。

流程：读 game/js_trace.json（由无头火狐跑游戏产生），用**同一串**食物随机数、
**同一串**共享动作序列在 Python 环境里重放，然后逐步比对：

  * 游戏状态：身体序列 / 朝向 / 食物 / 分数 / 死亡时机
  * 观测向量：170 维逐项比对（这是最容易出错的地方 —— 布局、通道顺序、旋转方向）
  * 贪心动作：JS 前向 vs Python 前向（float64 vs float32 的舍入可能让个别接近的动作翻转，需看比例）

用法： .venv/bin/python game/check_js_parity.py [game/js_trace.json]
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snake_env import VectorSnake, N_ACTIONS
from export_policy import f32_b64, np_forward

GROUPS = [
    (0, 2, "危险(直/左/右)"), (3, 6, "到墙距离"), (7, 10, "朝向one-hot"),
    (11, 12, "食物位移"), (13, 16, "食物方向"), (17, 18, "同行同列"),
    (19, 19, "BFS距离"), (20, 20, "食物可达"), (21, 21, "可达面积"),
    (22, 22, "长度占比"),
]


def group_of(i):
    for lo, hi, name in GROUPS:
        if lo <= i <= hi:
            return name
    return "patch"


def lcg(seed):
    s = seed & 0xFFFFFFFF
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s / 4294967296


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "game/js_trace.json"
    data = json.load(open(path))
    pol = json.load(open("runs/ppo20/policy.json"))
    layers = [{"W": f32_b64(L["w"]).reshape(L["out"], L["in"]),
               "b": f32_b64(L["b"]), "act": L["act"]} for L in pol["layers"]]

    uni = lcg(12345)
    act_rng = lcg(999)
    uniforms = np.array([next(uni) for _ in range(300000)], dtype=np.float64)
    env = VectorSnake(n_envs=1, seed=0, uniforms=uniforms, max_steps=10 ** 9, safety_mask=True)

    trace = data["trace"]
    n_live = n_dead_js = 0
    n_dead_py = 0
    maxdiff = 0.0
    worst = []
    act_agree = act_tot = 0
    safe_agree = safe_tot = 0
    first_safe_bad = None
    st_bad = 0
    first_st_bad = None
    next_expect_dead = False
    dead_misalign = 0

    for entry in trace:
        if entry.get("dead"):
            if not next_expect_dead:
                dead_misalign += 1
            next_expect_dead = False
            n_dead_js += 1
            continue

        # ---- Python 侧：当前(动作前)状态 ----
        p_obs = env.observe()[0]
        p_cells = env.body_cells(0)                      # 尾→头
        p_dir = int(env.dir[0])
        p_food = int(env.food[0])
        p_score = int(env.score[0])

        st = entry["st"]
        js_cells = [y * env.cols + x for (x, y) in st["snake"]][::-1]
        if (js_cells != p_cells or st["dir"] != p_dir or st["score"] != p_score
                or (st["food"][1] * env.cols + st["food"][0] if st["food"] else -1) != p_food):
            st_bad += 1
            if first_st_bad is None:
                first_st_bad = (entry["t"], st, {"cells": p_cells, "dir": p_dir,
                                                 "food": p_food, "score": p_score})

        # ---- 观测比对 ----
        j_obs = np.asarray(entry["obs"], dtype=np.float32)
        d = np.abs(j_obs - p_obs)
        if d.max() > maxdiff:
            maxdiff = float(d.max())
        if d.max() > 1e-4:
            idx = int(d.argmax())
            worst.append((entry["t"], idx, group_of(idx), float(d[idx]),
                          float(j_obs[idx]), float(p_obs[idx])))

        # ---- 贪心动作比对 ----
        p_act = int(np_forward(p_obs[None, :], layers).argmax(axis=1)[0])
        act_tot += 1
        if p_act == entry["aiAct"]:
            act_agree += 1

        # ---- 安全盾判据比对（这才是"游戏里的 AI 不会自陷"的依据）----
        # 两边都取**两级盾**的结果：JS 的 safety() 返回值就是盾，Python 侧对应 env.safe
        py_safe = env.safe.copy()
        js_safe = np.asarray(entry["safe"], dtype=bool)
        safe_tot += 1
        if np.array_equal(py_safe[0], js_safe):
            safe_agree += 1
        elif first_safe_bad is None:
            strict, alive, _ = env.safety_flags()
            first_safe_bad = (entry["t"], py_safe[0].tolist(), js_safe.tolist(),
                              strict[0].tolist(), alive[0].tolist())

        # ---- 用共享动作推进 Python ----
        assert int(entry["act"]) == int(np.floor(next(act_rng) * 3)), "动作流错位"
        _, _, term, trunc, info = env.step(np.array([entry["act"]]))
        if bool(info["finished"][0]):
            n_dead_py += 1
            next_expect_dead = True
        n_live += 1

    print(f"对拍条目 {n_live} 步（其中死亡 {n_dead_py} 次）")
    print(f"[状态] 不一致步数: {st_bad}" + (f"  首次 @t={first_st_bad[0]}" if first_st_bad else ""))
    if first_st_bad:
        print(f"        JS: {first_st_bad[1]}\n        PY: {first_st_bad[2]}")
    print(f"[观测] 最大逐项偏差 {maxdiff:.3e}")
    if worst:
        print(f"       偏差 >1e-4 的次数 {len(worst)}，最差几处：")
        for t, idx, g, dv, jv, pv in sorted(worst, key=lambda z: -z[3])[:5]:
            print(f"         t={t:<4} idx={idx:<4} {g:<12} JS={jv:.6f} PY={pv:.6f} Δ={dv:.2e}")
    else:
        print("       全部 170 维在 1e-4 内一致")
    print(f"[动作] 贪心动作一致率 {act_agree}/{act_tot} = {100*act_agree/max(1,act_tot):.3f}%")
    print(f"[安全] 安全层判据一致率 {safe_agree}/{safe_tot} = {100*safe_agree/max(1,safe_tot):.3f}%")
    if first_safe_bad:
        print(f"       首次不一致 @t={first_safe_bad[0]}  PY={first_safe_bad[1]}  JS={first_safe_bad[2]}")
    print(f"[死亡] JS {n_dead_js} 次 / PY {n_dead_py} 次，错位 {dead_misalign}")


if __name__ == "__main__":
    main()
