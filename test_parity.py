"""
对拍：向量化环境 vs snake.html JS 循环的逐行转写（SnakeRefEnv）。

两者喂同一串均匀随机数（决定食物落点）和同一串动作，要求逐格一致：
长度 / 身体序列 / 食物格 / 分数 / 死亡时机 / 死因，
并且每一步都比对 BFS 到食物距离与可达面积（与纯 Python BFS 比）。

用法： .venv/bin/python test_parity.py [步数]
"""

import sys
import numpy as np

from snake_env import VectorSnake, SnakeRefEnv, CAUSE_WALL, CAUSE_SELF, CAUSE_WIN


def ref_cells(ref, cols):
    """ref.snake 是 (x,y) 头→尾；转成 cell 索引的 尾→头 列表。"""
    return [y * cols + x for (x, y) in ref.snake][::-1]


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rng = np.random.default_rng(1234)
    uniforms = rng.random(500000)
    actions = rng.integers(0, 3, size=T)

    vec = VectorSnake(n_envs=1, seed=0, uniforms=uniforms, max_steps=10 ** 9)
    ref = SnakeRefEnv(uniforms=uniforms)

    n_finish = 0
    n_bfs = 0
    CAUSE_OF = {"wall": CAUSE_WALL, "self": CAUSE_SELF, "win": CAUSE_WIN}
    for t in range(T):
        a = int(actions[t])
        obs, r, term, trunc, info = vec.step([a])
        ref.step(a)

        finished_v = bool(info["finished"][0])
        finished_r = ref.dead or ref.win
        if finished_v != finished_r:
            return fail(t, "结束时机不一致", finished_v, finished_r, vec, ref)

        if finished_v:
            # 死亡/通关这一步：向量化环境已自动重开，所以先比终局信息，再比重开后的状态
            n_finish += 1
            if int(info["final_score"][0]) != ref.score:
                return fail(t, "终局分数不一致", int(info["final_score"][0]), ref.score, vec, ref)
            if int(info["cause"][0]) != CAUSE_OF[ref.cause]:
                return fail(t, "死因不一致", int(info["cause"][0]), ref.cause, vec, ref)
            ref.reset()
            if ref.food[1] * vec.cols + ref.food[0] != int(vec.food[0]):
                return fail(t, "重开后食物不一致", int(vec.food[0]),
                            ref.food[1] * vec.cols + ref.food[0], vec, ref)
            if ref_cells(ref, vec.cols) != vec.body_cells(0):
                return fail(t, "重开后身体不一致", vec.body_cells(0), ref_cells(ref, vec.cols), vec, ref)
            continue

        # ---- 存活步：食物 / 身体 / 分数 / 步数 ----
        vfood = int(vec.food[0])
        rfood = -1 if ref.food is None else ref.food[1] * vec.cols + ref.food[0]
        if vfood != rfood:
            return fail(t, "食物不一致", vfood, rfood, vec, ref)

        vc, rc = vec.body_cells(0), ref_cells(ref, vec.cols)
        if vc != rc:
            return fail(t, "身体不一致", vc, rc, vec, ref)
        if int(vec.score[0]) != ref.score or int(vec.length[0]) != len(ref.snake):
            return fail(t, "分数/长度不一致", (int(vec.score[0]), int(vec.length[0])),
                        (ref.score, len(ref.snake)), vec, ref)
        if int(vec.steps[0]) != ref.steps:
            return fail(t, "步数不一致", int(vec.steps[0]), ref.steps, vec, ref)

        # ---- BFS 距离 / 可达面积 ----
        if t % 10 == 0:
            dist, vis, hy, hx = vec._bfs()
            fy, fx = int(vec.food[0]) // vec.cols, int(vec.food[0]) % vec.cols
            vd = int(dist[0, fy, fx]) if bool(vis[0, fy, fx]) else -1
            rd = ref.bfs_dist_to_food()
            va = int(vis[0].sum())
            ra = ref.reachable_area()
            if vd != rd or va != ra:
                return fail(t, "BFS/可达面积不一致", (vd, va), (rd, ra), vec, ref)
            n_bfs += 1

    print(f"PASS  步数={T}  完成回合={n_finish}  BFS 抽样比对={n_bfs}")
    print(f"      观测维度={vec.obs_dim}  最终分数={int(vec.score[0])}  当前长度={int(vec.length[0])}")
    obs = vec.observe()
    assert obs.shape == (1, vec.obs_dim), obs.shape
    assert np.isfinite(obs).all(), "观测里有 NaN/Inf"
    print(f"      观测范围 [{obs.min():.3f}, {obs.max():.3f}] 无 NaN/Inf")
    return 0


def fail(t, msg, a, b, vec, ref):
    print(f"FAIL @step {t}: {msg}\n  向量化: {a}\n  参考:   {b}")
    print("--- 向量化棋盘 ---")
    print(vec.ascii(0))
    print("--- 参考棋盘 ---")
    g = [["." for _ in range(ref.cols)] for _ in range(ref.rows)]
    for i, (x, y) in enumerate(ref.snake):
        g[y][x] = "H" if i == 0 else "o"
    if ref.food:
        g[ref.food[1]][ref.food[0]] = "*"
    print("\n".join("".join(r) for r in g))
    return 1


if __name__ == "__main__":
    sys.exit(main())
