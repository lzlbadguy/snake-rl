"""
向量化贪吃蛇环境 —— 规则与浏览器游戏 /home/lzl/snake.html 完全一致。

游戏规则（对齐 snake.html）：
  * 20x20 网格，不穿墙；初始长度 3，位置 (5,10),(4,10),(3,10)，朝右
  * 撞墙 / 撞到自己身体死亡；**尾巴正在离开的那一格不算碰撞**
  * 食物在空闲格里均匀随机，采样顺序为 y 主序（逐行扫描），与 JS 的
    `free[(Math.random()*free.length)|0]` 完全一致
  * 不加速、无时限：速度机制只影响观感，不影响 MDP

设计要点：
  * N 个环境全部存在扁平的 numpy 数组里，一次 step() 只是十几个向量化算子，
    每步开销 ~O(N) 且没有 Python 层循环 —— 多环境加速就来自这里
  * 身体用环形缓冲：body[env, (head_ptr - k) % MAXLEN] 是第 k 段（k=0 是头）。
    变长 = 前移 head_ptr 但不覆盖尾巴那一格
  * 观测是"自车坐标系"：7x7x3 局部 patch（身体/食物/墙，按朝向旋转，前方恒为上）
    + 23 维标量特征，其中含 BFS 到食物的距离和 flood-fill 可达面积（自陷信号）
  * 自动 reset：死亡的 env 在 step() 内部直接重开，terminated / truncated 分开返回，
    训练器据此对 bootstrap 做掩码
"""

from __future__ import annotations

import numpy as np

# 方向索引：0=上 1=右 2=下 3=左（y 向下，与 JS 一致，顺时针）
DY = np.array([-1, 0, 1, 0], dtype=np.int16)
DX = np.array([0, 1, 0, -1], dtype=np.int16)
# 动作：0=直行 1=左转 2=右转（相对动作，天然禁掉 180° 掉头）
TURN = np.array([0, -1, 1], dtype=np.int8)
N_ACTIONS = 3

CAUSE_WALL, CAUSE_SELF, CAUSE_CAP, CAUSE_WIN = 0, 1, 2, 3

PATCH_R = 3                      # patch 半径 → 7x7
PATCH_K = 2 * PATCH_R + 1
WALL_CH, FOOD_CH, BODY_CH = 0, 1, 2   # patch 通道顺序
N_SCALAR = 23


def _build_patch_offsets() -> np.ndarray:
    """PATCH_OFF[d, i, j] = (dy, dx)：朝向 d 下，patch 第 i 行第 j 列相对头部的偏移。

    行索引沿前进方向增大（patch[0] = 前方 3 格），列索引沿蛇的右手方向增大。
    """
    off = np.zeros((4, PATCH_K, PATCH_K, 2), dtype=np.int16)
    for d in range(4):
        fy, fx = DY[d], DX[d]
        ry, rx = DY[(d + 1) % 4], DX[(d + 1) % 4]
        for i in range(PATCH_K):
            for j in range(PATCH_K):
                f, s = i - PATCH_R, j - PATCH_R
                off[d, i, j, 0] = f * fy + s * ry
                off[d, i, j, 1] = f * fx + s * rx
    return off


PATCH_OFF = _build_patch_offsets()


def bfs_from(occ, src, cols, rows):
    """从 src(每环境一个 cell 索引) 在 occ 为障碍的网格上做 BFS。
    occ 中起点需已被置为可通行。返回 (dist (n,R,C) int16, vis (n,R,C) bool)。"""
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


class VectorSnake:
    """向量化贪吃蛇。N 个环境同步推进。"""

    def __init__(self, n_envs=1024, rows=20, cols=20, max_steps=None, seed=0,
                 use_patch=True, gamma=0.99, food_reward=1.0, death_penalty=1.0,
                 step_penalty=0.001, shaping_scale=0.05, area_shaping=0.0,
                 doomed_penalty=0.0, safety_mask=False, tier_shield=True, shield_mode="tail",
                 arc_slack=8, uniforms=None):
        self.n = int(n_envs)
        self.rows, self.cols = int(rows), int(cols)
        self.cells = self.rows * self.cols
        self.max_len = self.cells
        self.max_steps = int(max_steps) if max_steps else 4 * self.cells
        self.use_patch = bool(use_patch)
        self.gamma = float(gamma)
        self.food_reward = float(food_reward)
        self.death_penalty = float(death_penalty)
        self.step_penalty = float(step_penalty)
        self.shaping_scale = float(shaping_scale)
        self.area_shaping = float(area_shaping)
        self.doomed_penalty = float(doomed_penalty)
        self.safety_mask = bool(safety_mask)
        self.tier_shield = bool(tier_shield)        # True=两级盾（一级严格/二级降级）；False=旧行为
        self.shield_mode = str(shield_mode)         # "tail"=尾巴可达性 / "arc"=回路连续弧（可证明不死）
        if self.shield_mode in ("cycle", "arc"):
            from hamilton import build_cycle
            cyc = build_cycle(self.rows, self.cols)
            idx = np.zeros(self.cells, dtype=np.int32)
            cel = np.zeros(len(cyc), dtype=np.int32)
            for i, (x, y) in enumerate(cyc):
                idx[y * self.cols + x] = i
                cel[i] = y * self.cols + x
            self.cyc_idx = idx
            self.cyc_cell = cel
            self.cyc_n = len(cyc)
        else:
            self.cyc_idx = None
            self.cyc_cell = None
            self.cyc_n = 0
        self.arc_T = np.zeros(self.n, dtype=np.int32)   # 弧起点（回路序号），只前进
        self.arc_slack = int(arc_slack)                # 一次最多能超前蛇头多少格（限制弧的膨胀）
        self.tier = np.zeros(self.n, dtype=np.int8) # 本步生效的盾级别：0 严格 / 1 降级 / 2 死局
        self.max_dim = max(self.rows, self.cols)
        # 势函数归一化尺度：2*max_dim 之外的 BFS 距离一律当"很远"
        self.phi_scale = 2.0 * self.max_dim

        self.rng = np.random.default_rng(seed)
        self._uniforms = uniforms          # 可选：外部注定的均匀随机流（对拍用）
        self._u = 0

        self.obs_dim = N_SCALAR + (3 * PATCH_K * PATCH_K if self.use_patch else 0)

        n = self.n
        self._ar = np.arange(n)
        self.body = np.zeros((n, self.max_len), dtype=np.int32)   # 环形缓冲
        self.head_ptr = np.zeros(n, dtype=np.int32)
        self.length = np.zeros(n, dtype=np.int32)
        self.occ = np.zeros((n, self.cells), dtype=bool)
        self.dir = np.zeros(n, dtype=np.int8)
        self.food = np.full(n, -1, dtype=np.int32)
        self.steps = np.zeros(n, dtype=np.int32)
        self.score = np.zeros(n, dtype=np.int32)
        self.phi = np.zeros(n, dtype=np.float32)   # 上一时刻势函数 Φ(s)
        self.area = np.zeros(n, dtype=np.int32)    # 蛇头可达空闲格数（含头）
        self.doomed = np.zeros(n, dtype=bool)      # 是否已"注定死亡"
        self.safe = np.ones((n, 3), dtype=bool)    # 三个相对动作的安全掩码
        self.area3 = np.zeros((n, 3), dtype=np.int32)
        self.reset()

    # ---------------- 基础 ----------------

    def _segment(self, k):
        """第 k 段（k=0 为头）所在的 cell 索引，形状 (n,)。"""
        return self.body[self._ar, (self.head_ptr - k) % self.max_len]

    def _draw_uniforms(self, m):
        if m <= 0:
            return np.zeros(0, dtype=np.float64)
        if self._uniforms is not None:
            u = np.asarray(self._uniforms[self._u:self._u + m], dtype=np.float64)
            self._u += m
            return u
        return self.rng.random(m)

    def reset(self, ids=None):
        ids = self._ar if ids is None else np.asarray(ids)
        if ids.size == 0:
            return
        cy = self.rows // 2
        # JS: snake = [(5,cy),(4,cy),(3,cy)]，头在最前
        head = cy * self.cols + 5
        for k, cell in enumerate((head, head - 1, head - 2)):
            self.body[ids, (-k) % self.max_len] = cell
        if self.shield_mode == "arc":
            # 弧起点 = 初始身体在回路上的最小序号（初始三节正好是回路上连续的弧）
            cells3 = np.array([head, head - 1, head - 2], dtype=np.int32)
            self.arc_T[ids] = int(self.cyc_idx[cells3].min())
        self.head_ptr[ids] = 0
        self.length[ids] = 3
        self.dir[ids] = 1                                   # 朝右
        self.steps[ids] = 0
        self.score[ids] = 0
        self.occ[ids] = False
        for k in range(3):
            self.occ[ids, head - k] = True
        self.food[ids] = -1
        self._place_food(ids)

    def _place_food(self, ids, u=None):
        """在空闲格里按均匀分布放食物（y 主序），返回"无处可放"（填满棋盘）的掩码。"""
        ids = np.atleast_1d(np.asarray(ids))
        m = ids.size
        if m == 0:
            return np.zeros(0, dtype=bool)
        u = self._draw_uniforms(m) if u is None else np.asarray(u, dtype=np.float64)
        free = ~self.occ[ids]                        # (m, cells)
        cnt = free.sum(axis=1)
        cum = np.cumsum(free, axis=1)
        # 等价于 free[floor(u*len(free))]：第一个累计数 > floor(u*cnt) 的空闲格
        thr = np.floor(u * cnt).astype(np.int32)
        pick = np.argmax(cum > thr[:, None], axis=1).astype(np.int32)
        no_space = cnt == 0
        self.food[ids] = np.where(no_space, -1, pick)
        return no_space

    # ---------------- 观测 ----------------

    def _bfs(self):
        """从头在"身体=障碍"的网格上做 BFS，返回 (dist(n,R,C), reachable(n,R,C))。

        这一趟同时给出：到食物的最短路长度 + 蛇头可达的空闲面积。
        """
        n = self.n
        ar = self._ar
        head_cell = self.body[ar, self.head_ptr]
        hy, hx = head_cell // self.cols, head_cell % self.cols
        free3 = (~self.occ).reshape(n, self.rows, self.cols)
        free3[ar, hy, hx] = True              # 头自己当然算"可站立"
        vis = np.zeros_like(free3)
        fr = np.zeros_like(free3)
        fr[ar, hy, hx] = True
        vis |= fr
        dist = np.zeros((n, self.rows, self.cols), dtype=np.int16)
        d = 0
        while fr.any():
            nb = np.zeros_like(fr)
            nb[:, 1:, :] |= fr[:, :-1, :]
            nb[:, :-1, :] |= fr[:, 1:, :]
            nb[:, :, 1:] |= fr[:, :, :-1]
            nb[:, :, :-1] |= fr[:, :, 1:]
            nb &= free3
            nb &= ~vis
            if not nb.any():
                break
            d += 1
            dist[nb] = d
            vis |= nb
            fr = nb
        return dist, vis, hy, hx

    def observe(self):
        n, cells = self.n, self.cells
        ar = self._ar
        if self.safety_mask:
            if self.shield_mode == "arc":
                self._arc_advance()                     # 先释放空洞、推弧起点
                self.safe, self.tier = self._arc_mask()
            elif self.shield_mode == "cycle":
                self.safe, self.tier = self._cycle_mask()
            else:
                strict, alive, self.area3 = self.safety_flags()
                self.safe, self.tier = self._shield_tiers(strict, alive)
        head_cell = self.body[ar, self.head_ptr]
        hy = (head_cell // self.cols).astype(np.int16)
        hx = (head_cell % self.cols).astype(np.int16)

        dist, vis, _, _ = self._bfs()

        # --- 到食物的 BFS 距离 ---
        has_food = self.food >= 0
        fy = np.where(has_food, self.food // self.cols, 0).astype(np.int16)
        fx = np.where(has_food, self.food % self.cols, 0).astype(np.int16)
        reached = vis[ar, fy, fx] & has_food
        bfs_d = np.where(reached, dist[ar, fy, fx], -1).astype(np.float32)
        bfs_norm = np.where(bfs_d < 0, 1.0, np.clip(bfs_d / self.phi_scale, 0.0, 1.0)).astype(np.float32)
        self.area = vis.sum(axis=(1, 2)).astype(np.int32)
        area_frac = (self.area / cells).astype(np.float32)

        # --- 势函数（用于 potential-based shaping，策略不变）---
        # 第二项把"可达空间"也放进势函数：空间被自己压缩时 Φ 变差，给出
        # "别把自己困死"的密集梯度。两项都只是状态的函数，不改变最优策略。
        self.phi = (-self.shaping_scale * bfs_norm
                    - self.area_shaping * (1.0 - area_frac)).astype(np.float32)

        # --- 绝对方向上的撞墙距离 ---
        md = float(self.max_dim)
        wall_up = (hy.astype(np.float32) / md)
        wall_dn = ((self.rows - 1 - hy).astype(np.float32) / md)
        wall_lf = (hx.astype(np.float32) / md)
        wall_rt = ((self.cols - 1 - hx).astype(np.float32) / md)

        # --- 相对方向（前/左/右）的即时危险 ---
        occ_flat = self.occ
        danger = np.zeros((n, 3), dtype=np.float32)
        danger[:, 0] = self._danger_mask(0, hy, hx, occ_flat)   # 直行
        danger[:, 1] = self._danger_mask(1, hy, hx, occ_flat)   # 左转
        danger[:, 2] = self._danger_mask(2, hy, hx, occ_flat)   # 右转

        # --- 朝向 one-hot ---
        dir_oh = np.zeros((n, 4), dtype=np.float32)
        dir_oh[ar, self.dir] = 1.0

        # --- 食物相对位置 ---
        ddx = np.where(has_food, fx.astype(np.float32) - hx.astype(np.float32), 0.0) / (self.cols - 1)
        ddy = np.where(has_food, fy.astype(np.float32) - hy.astype(np.float32), 0.0) / (self.rows - 1)
        fdir = np.stack([
            (fy < hy).astype(np.float32), (fy > hy).astype(np.float32),
            (fx < hx).astype(np.float32), (fx > hx).astype(np.float32),
        ], axis=1)
        aligned = np.stack([(fy == hy).astype(np.float32), (fx == hx).astype(np.float32)], axis=1)
        if not has_food.any():
            fdir = np.zeros_like(fdir)

        scalars = np.concatenate([
            danger,                                    # 3
            np.stack([wall_up, wall_dn, wall_lf, wall_rt], axis=1),   # 4
            dir_oh,                                    # 4
            np.stack([ddx, ddy], axis=1),              # 2
            fdir,                                      # 4
            aligned,                                   # 2
            bfs_norm[:, None],                         # 1  到食物最短路
            reached.astype(np.float32)[:, None],       # 1  食物是否可达
            area_frac[:, None],                        # 1  可达空闲面积占比
            (self.length / cells).astype(np.float32)[:, None],        # 1
        ], axis=1).astype(np.float32)

        if not self.use_patch:
            return scalars

        # --- 自车坐标系 7x7x3 局部 patch ---
        off = PATCH_OFF[self.dir]                       # (n,K,K,2)
        oy = hy[:, None, None] + off[..., 0]
        ox = hx[:, None, None] + off[..., 1]
        valid = (oy >= 0) & (oy < self.rows) & (ox >= 0) & (ox < self.cols)
        oyc = np.clip(oy, 0, self.rows - 1).astype(np.int32)
        oxc = np.clip(ox, 0, self.cols - 1).astype(np.int32)
        pc = (oyc * self.cols + oxc).astype(np.int32)
        body_p = (occ_flat[ar[:, None, None], pc] & valid).astype(np.float32)
        food_p = ((pc == self.food[:, None, None]) & valid).astype(np.float32)
        wall_p = (~valid).astype(np.float32)
        patch = np.concatenate([wall_p, food_p, body_p], axis=1)   # (n,3,K,K)
        return np.concatenate([scalars, patch.reshape(n, -1)], axis=1).astype(np.float32)

    def _danger_mask(self, action, hy, hx, occ_flat):
        """相对动作 action 之后，头会落到的格子是否危险（出界或撞身体）。"""
        d = ((self.dir.astype(np.int16) + TURN[action]) % 4).astype(np.int16)
        ny = hy.astype(np.int16) + DY[d]
        nx = hx.astype(np.int16) + DX[d]
        out = (ny < 0) | (ny >= self.rows) | (nx < 0) | (nx >= self.cols)
        nyc = np.clip(ny, 0, self.rows - 1).astype(np.int32)
        nxc = np.clip(nx, 0, self.cols - 1).astype(np.int32)
        c = (nyc * self.cols + nxc).astype(np.int32)
        hit = occ_flat[self._ar, c]
        return (out | hit).astype(np.float32)

    # ---------------- 安全层（动作屏蔽） ----------------

    def safety_flags(self):
        """(safe (n,3) bool, areas (n,3))：三个相对动作各自是否安全。

        安全 = 走完这一步后还活着，且（尾巴仍可达 或 可达面积 ≥ 新身体长度）。
        关键细节：**吃食物那一步蛇会变长、尾巴不让位**，必须按这个语义模拟，
        否则会在最关键的时刻（吃进死胡同）判断过于乐观 —— 这是这类检查最常踩的坑。
        """
        n, cols, rows = self.n, self.cols, self.rows
        ar = self._ar
        head = self.body[ar, self.head_ptr]
        hy, hx = head // cols, head % cols

        def seg(k):
            return self.body[ar, (self.head_ptr - k) % self.max_len]

        tail_now = seg(self.length - 1)
        occ_now = self.occ.reshape(n, rows, cols)

        strict = np.zeros((n, 3), dtype=bool)
        alive_flags = np.zeros((n, 3), dtype=bool)
        areas = np.zeros((n, 3), dtype=np.int32)
        for a in range(3):
            d = ((self.dir.astype(np.int16) + TURN[a]) % 4).astype(np.int16)
            ny = hy + DY[d]
            nx = hx + DX[d]
            out = (ny < 0) | (ny >= rows) | (nx < 0) | (nx >= cols)
            nyc = np.clip(ny, 0, rows - 1)
            nxc = np.clip(nx, 0, cols - 1)
            ncell = (nyc * cols + nxc).astype(np.int32)
            ate = ncell == self.food
            vac = ~ate
            hit = self.occ[ar, ncell] & ~((ncell == tail_now) & vac)
            alive = ~(out | hit)
            length_after = self.length + ate.astype(np.int32)

            # 安全判据（两项）：
            #   1) tail_ok —— 走完之后，蛇头还能到达"当前尾巴那一格"（该格下一步会让开，
            #      所以必须把它当作可通行；**不能**拿新尾巴当目标，那一格此刻还被身体占着，
            #      BFS 永远进不去，判据会退化成恒 False）
            #   2) 可达面积 ≥ 走完之后的身体长度 —— 空间还容得下自己（管吃食物变长的情况）
            occ = occ_now.copy()
            occ[ar[vac], tail_now[vac] // cols, tail_now[vac] % cols] = False  # 尾巴让位
            occ[ar, tail_now // cols, tail_now % cols] = False                 # 目标格本身
            occ[ar, nyc, nxc] = False                                          # 新头所在格
            _, vis = bfs_from(occ, ncell, cols, rows)
            area = vis.reshape(n, -1).sum(1).astype(np.int32)
            tail_ok = vis[ar, tail_now // cols, tail_now % cols]
            alive_flags[:, a] = alive
            strict[:, a] = alive & (tail_ok | (area >= length_after))
            areas[:, a] = area
        return strict, alive_flags, areas

    def _shield_tiers(self, strict, alive):
        """两级安全盾（+ 死局兜底）。

        一级（严格）：尾巴仍可达 或 可达面积 ≥ 走完后的身体长度 —— 维持"回得到自己尾巴"
        这个不变量，是保命主力。
        二级（降级）：一级全空时，退而只禁「这一步就会死」的动作（出界 / 撞身体），
        让策略在"可能进死胡同、但这一步不会死"的动作里自己挑。
        三级（死局）：连二级都空（三面全堵）才放开全部 —— 那时已经无解。

        为什么必须有二级：旧实现一级全空时直接放开全部，于是**最危险的那一步反而完全没保护**。
        实测（回合上限 4000）72% 的死亡是撞墙，而出界动作在一级和二级里都是被禁的。
        """
        any_strict = strict.any(axis=1, keepdims=True)
        if not self.tier_shield:
            m = np.where(any_strict, strict, True)
            tier = np.where(any_strict[:, 0], 0, 2).astype(np.int8)
            return m, tier
        any_alive = alive.any(axis=1, keepdims=True)
        m = np.where(any_strict, strict, np.where(any_alive, alive, True))
        tier = np.where(any_strict[:, 0], 0, np.where(any_alive[:, 0], 1, 2)).astype(np.int8)
        return m, tier

    # ---------------- 弧盾（可证明不死） ----------------

    def _arc_advance(self):
        """推进弧起点 T：只要 T 指的格子没被身体占着，就释放它并前进一格。

        这条释放规则是弧盾成立的关键（上一版 cycle 模式正是栽在这里）：T **只前进**，
        每步只做 O(1) 的占用查询，均摊 O(1)（每格每圈最多释放一次）。它保证
        [T, H] 始终是**包含身体的最小连续弧** ⇒ 度量基准稳定、序关系不会翻转。
        """
        n = self.n
        for _ in range(self.cyc_n + 2):
            occ = self.occ[self._ar, self.cyc_cell[self.arc_T]]
            if not bool((~occ).any()):
                break
            self.arc_T = np.where(~occ, (self.arc_T + 1) % self.cyc_n, self.arc_T).astype(np.int32)

    def _arc_mask(self):
        """弧盾：**可证明不死**的动作屏蔽（O(1) 判据，不需要 BFS）。

        不变量 INV：身体 ⊆ [T, H] —— 回路上从弧起点 T 到蛇头 H 的前向连续弧，蛇头在弧末端。
        弧内非身体的格子即 **reserved（保留格：游戏里空闲、逻辑上不可进入）**，它是派生量
        （弧减去身体），不需要额外记账。

        为什么可证明：可行动作 q 满足 `alive ∧ d(T,q) ≥ d(T,H) ∧ (d(T,q)-d(T,H)) ≤ arc_slack`。
          * 头只前进（相对 T）⇒ 旧身体 ⊆ [T,H] ⊆ [T,H']，弧依然连续；
          * 超前量封顶 ⇒ 弧不会无限膨胀（不封顶时实测贪心 384 万步一口没吃：食物被困在
            保留格里）；slack ≥ 1 保证"沿回路前进一步"永远可行动作 ⇒ 可行动作集恒不为空；
          * 尾巴让位只会往弧里"加"保留格，不会让弧断裂；弧起点由 _arc_advance 从前面释放。
        ⇒ 只要策略从可行动作里选，蛇**不撞墙、不撞自己、可无限存活**（已验证：1.28M 步逐
        步校验不变量 0 违反；3.84M 步 0 死亡；可行动作集为空的比例 0.0000%）。

        ⚠️ **但"不死"不等于"会前进"**：判据的度量是 d(T,·)，而 T 每步前进，所以头可以
        "保持在相对偏移上漂移"（实测轨迹：H 每步 +1，同时 T 也 +1，头并不扫整条回路），
        食物只能碰巧遇上（贪心实测 6 分 / 2.5 万步，而纯 Hamilton 约 250 分 / 2.5 万步）。
        经典 shortcut 算法还缺一块：**身体的回路序（body 必须按回路顺序从头顶到尾排开）
        + 被跳过格的显式记账**，靠它堵住"反向漂移"。本模式因此仍标注为实验性：
        安全性可用，进度/效率不可用。
        """
        ar, n = self._ar, self.n
        head = self.body[ar, self.head_ptr]
        tail = self.body[ar, (self.head_ptr - self.length + 1) % self.max_len]
        H = self.cyc_idx[head]
        T = self.arc_T
        dTH = (H - T) % self.cyc_n
        hy = (head // self.cols).astype(np.int16)
        hx = (head % self.cols).astype(np.int16)
        ok = np.zeros((n, 3), dtype=bool)
        alive_arr = np.zeros((n, 3), dtype=bool)
        for a in range(3):
            d = ((self.dir.astype(np.int16) + TURN[a]) % 4).astype(np.int16)
            ny = hy + DY[d]
            nx = hx + DX[d]
            out = (ny < 0) | (ny >= self.rows) | (nx < 0) | (nx >= self.cols)
            nyc = np.clip(ny, 0, self.rows - 1)
            nxc = np.clip(nx, 0, self.cols - 1)
            cell = (nyc * self.cols + nxc).astype(np.int32)
            ate = cell == self.food
            vac = ~ate
            hit = self.occ[ar, cell] & ~((cell == tail) & vac)
            alive = ~(out | hit)
            dTq = (self.cyc_idx[cell] - T) % self.cyc_n
            alive_arr[:, a] = alive
            # 允许的超前量必须封顶：否则弧越长空洞越多，食物会被困在保留格里吃不到
            # （实测：不封顶时贪心 384 万步一口没吃）。slack=1 就退化成纯 Hamilton 跟随。
            ok[:, a] = alive & (dTq >= dTH) & ((dTq - dTH) <= self.arc_slack)
        any_ok = ok.any(axis=1, keepdims=True)
        any_alive = alive_arr.any(axis=1, keepdims=True)
        m = np.where(any_ok, ok, np.where(any_alive, alive_arr, True))
        tier = np.where(any_ok[:, 0], 0, np.where(any_alive[:, 0], 1, 2)).astype(np.int8)
        return m, tier

    def _cycle_mask(self):
        """⚠️ 实验性 —— **理论已被证伪，不可用于生产**（2026-09-13）。

        设计意图：不变量"所有身体格的回路序号落在 [T,H] 前向弧上"，可行动作
        = `活着 ∧ d(T,q) >= d(T,H)`，并声称"沿回路前进一步永远可行" ⇒ 结构上不死。

        实测证伪（`check_cycle_shield.py` + `debug_cycle.py`）：均匀随机策略跑到
        **第 3378 步**破坏不变量。反例状态：头格 182(序号 188)、尾格 181(序号 189)
        —— dTH = 399 = N-1 的退化态，此时头的前向弧覆盖整个回路，"跟随时空转"，
        严格判据对三个动作全部为假（364 / 0 / 398 均 < 399），于是走降级兜底放行了
        "沿回路后退一步"（dTq=398），身体在回路上交叉。

        根因：判据的度量基准是**当前尾巴 T**，而尾巴每步会移动 ⇒ 度量基准漂移，
        顺序可以在 `(v-u) mod N` 处翻转；"头的回路序号必须 >= 身体的"这个序关系
        **不被前向跳跃保持**（头会越过自己身体在回路上的顺序）。

        正确做法（经典 shortcut 算法）：记 **reserved（被跳过、暂不可进入）格集合**，
        身体 + reserved 必须是回路上的连续弧，尾巴离开某格时才释放它。这是 O(L) 的
        显式记账，不是 O(1) 判据。保留本模式仅作研究记录，别在训练/部署里打开。
        """
        ar = self._ar
        n = self.n
        head = self.body[ar, self.head_ptr]
        tail = self.body[ar, (self.head_ptr - self.length + 1) % self.max_len]
        H = self.cyc_idx[head]
        T = self.cyc_idx[tail]
        dTH = (H - T) % self.cyc_n

        hy = (head // self.cols).astype(np.int16)
        hx = (head % self.cols).astype(np.int16)
        ok = np.zeros((n, 3), dtype=bool)
        alive_arr = np.zeros((n, 3), dtype=bool)
        for a in range(3):
            d = ((self.dir.astype(np.int16) + TURN[a]) % 4).astype(np.int16)
            ny = hy + DY[d]
            nx = hx + DX[d]
            out = (ny < 0) | (ny >= self.rows) | (nx < 0) | (nx >= self.cols)
            nyc = np.clip(ny, 0, self.rows - 1)
            nxc = np.clip(nx, 0, self.cols - 1)
            cell = (nyc * self.cols + nxc).astype(np.int32)
            ate = cell == self.food
            vac = ~ate
            hit = self.occ[ar, cell] & ~((cell == tail) & vac)
            alive = ~(out | hit)
            dTq = (self.cyc_idx[cell] - T) % self.cyc_n
            alive_arr[:, a] = alive
            ok[:, a] = alive & (dTq >= dTH)

        any_ok = ok.any(axis=1, keepdims=True)
        any_alive = alive_arr.any(axis=1, keepdims=True)
        m = np.where(any_ok, ok, np.where(any_alive, alive_arr, True))
        tier = np.where(any_ok[:, 0], 0, np.where(any_alive[:, 0], 1, 2)).astype(np.int8)
        return m, tier

    # ---------------- 推进 ----------------

    def step(self, actions):
        """推进一步。返回 (obs, reward, terminated, truncated, info)，自动 reset。

        terminated = 撞墙/撞自己/填满棋盘；truncated = 到达步数上限。
        训练器应对两者都掩掉 bootstrap（截断不 bootstrap 会带来极小偏差，
        但本环境绝大多数回合以死亡结束，影响可忽略）。
        """
        n = self.n
        ar = self._ar
        a = np.asarray(actions, dtype=np.int8)

        self.dir = ((self.dir.astype(np.int16) + TURN[a]) % 4).astype(np.int8)

        head_cell = self.body[ar, self.head_ptr]
        hy = (head_cell // self.cols).astype(np.int16)
        hx = (head_cell % self.cols).astype(np.int16)
        ny = hy + DY[self.dir]
        nx = hx + DX[self.dir]
        out = (ny < 0) | (ny >= self.rows) | (nx < 0) | (nx >= self.cols)
        nyc = np.clip(ny, 0, self.rows - 1).astype(np.int32)
        nxc = np.clip(nx, 0, self.cols - 1).astype(np.int32)
        new_cell = (nyc * self.cols + nxc).astype(np.int32)

        tail_cell = self._segment(self.length - 1)
        ate_raw = new_cell == self.food
        # 尾巴当前那一格在"不吃食物"的情况下会腾空，所以不算碰撞（与 JS 的
        # `for (i = 0; i < len-1; i++)` 等价）
        hit_self = self.occ[ar, new_cell] & ~((new_cell == tail_cell) & ~ate_raw)
        dead = out | hit_self
        ate = ate_raw & ~dead

        # --- 奖励 ---
        r = ate.astype(np.float32) * self.food_reward
        r -= self.step_penalty
        r -= dead.astype(np.float32) * self.death_penalty

        # --- 状态推进（死亡的 env 稍后重置，这里写成无害）---
        rm = ~dead & ~ate
        self.occ[ar[rm], tail_cell[rm]] = False
        self.occ[ar, new_cell] = True
        new_ptr = (self.head_ptr + 1) % self.max_len
        self.body[ar, new_ptr] = new_cell
        self.head_ptr = new_ptr
        self.length = self.length + ate
        self.score = self.score + ate
        self.steps = self.steps + 1

        # --- 食物重采样 ---
        win = np.zeros(n, dtype=bool)
        if ate.any():
            win_ate = self._place_food(np.flatnonzero(ate))
            win[np.flatnonzero(ate)] = win_ate

        truncated = self.steps >= self.max_steps
        terminated = dead | win
        finished = terminated | truncated

        info = {
            "finished": finished,
            "final_score": self.score.copy(),
            "final_len": self.length.copy(),
            "final_steps": self.steps.copy(),
            "cause": np.where(dead & out, CAUSE_WALL,
                     np.where(dead & hit_self, CAUSE_SELF,
                     np.where(win, CAUSE_WIN, CAUSE_CAP))).astype(np.int8),
            "ate": ate,
        }

        # 势函数塑形：r += γΦ(s') - Φ(s)。终局 Φ=0。
        keep = ~finished
        phi_prev = self.phi.copy()

        if finished.any():
            self.reset(np.flatnonzero(finished))

        obs = self.observe()
        r += (self.gamma * np.where(keep, self.phi, 0.0) - phi_prev).astype(np.float32)

        # "注定死亡"提前惩罚：可达格数 < 身体长度 ⇒ 蛇已无处可躲，必然会在
        # 若干步后撞死（可达区域最多只能容纳 body 长度那么多格）。把本来要到
        # 几十步之后才拿到的 -1 提前成当下的密集惩罚，信用分配容易得多。
        # 注意：self.doomed 始终计算（诊断与日志都要用），惩罚项只在权重 >0 时生效
        self.doomed = (self.area < self.length) & keep
        if self.doomed_penalty > 0.0:
            r -= (self.doomed_penalty * self.doomed).astype(np.float32)
        info["doomed"] = self.doomed
        return obs, r.astype(np.float32), terminated, truncated, info

    # ---------------- 调试辅助 ----------------

    def body_cells(self, env):
        """返回某个环境从尾到头的身体 cell 列表（调试/对拍用）。"""
        L = int(self.length[env])
        return [int(self.body[env, (self.head_ptr[env] - k) % self.max_len]) for k in range(L - 1, -1, -1)]

    def ascii(self, env):
        g = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for k in range(int(self.length[env])):
            c = int(self.body[env, (self.head_ptr[env] - k) % self.max_len])
            g[c // self.cols][c % self.cols] = "o" if k else "H"
        if self.food[env] >= 0:
            f = int(self.food[env])
            g[f // self.cols][f % self.cols] = "*"
        return "\n".join("".join(row) for row in g)


# --------------------------------------------------------------------------
# 参考实现：snake.html 里 JS 循环的逐行转写（(x,y) 坐标，纯 Python）
# 用途：向量化实现的正确性对拍 + 串行吞吐基线
# --------------------------------------------------------------------------

DIRS_XY = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class SnakeRefEnv:
    def __init__(self, rows=20, cols=20, uniforms=None, seed=0):
        self.rows, self.cols = rows, cols
        self.max_len = rows * cols
        self.uniforms = uniforms
        self._u = 0
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _next_u(self):
        if self.uniforms is not None:
            v = float(self.uniforms[self._u])
            self._u += 1
            return v
        return float(self.rng.random())

    def reset(self):
        cy = self.rows // 2
        self.snake = [(5, cy), (4, cy), (3, cy)]   # 头在最前
        self.dir = (1, 0)
        self.score = 0
        self.steps = 0
        self.dead = False
        self.win = False
        self.cause = None
        self.food = self._place_food()
        return self

    def _place_food(self):
        body = set(self.snake)
        free = [(x, y) for y in range(self.rows) for x in range(self.cols)
                if (x, y) not in body]
        if not free:
            return None
        return free[int(self._next_u() * len(free))]

    def step(self, action):
        if self.dead or self.win:
            return
        turn = (0, -1, 1)[action]
        idx = DIRS_XY.index(self.dir)
        self.dir = DIRS_XY[(idx + turn) % 4]
        hx, hy = self.snake[0]
        nx, ny = hx + self.dir[0], hy + self.dir[1]
        self.steps += 1
        # JS: 先判撞墙，再判撞自己（身体除尾巴）
        if nx < 0 or ny < 0 or nx >= self.cols or ny >= self.rows:
            self.dead = True
            self.cause = "wall"
            return
        for i in range(len(self.snake) - 1):
            if self.snake[i] == (nx, ny):
                self.dead = True
                self.cause = "self"
                return
        self.snake.insert(0, (nx, ny))
        if self.food is not None and (nx, ny) == self.food:
            self.score += 1
            self.food = self._place_food()
            if self.food is None:
                self.win = True
                self.cause = "win"
        else:
            self.snake.pop()

    def bfs_dist_to_food(self):
        """纯 Python BFS（对拍用），返回从头到食物的最短路长度，不可达返回 -1。"""
        from collections import deque
        body = set(self.snake[1:])
        start = self.snake[0]
        if self.food is None:
            return -1
        q = deque([(start, 0)])
        seen = {start}
        while q:
            (x, y), d = q.popleft()
            if (x, y) == self.food:
                return d
            for dx, dy in DIRS_XY:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.cols and 0 <= ny < self.rows):
                    continue
                if (nx, ny) in body or (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                q.append(((nx, ny), d + 1))
        return -1

    def reachable_area(self):
        from collections import deque
        body = set(self.snake[1:])
        start = self.snake[0]
        q = deque([start])
        seen = {start}
        while q:
            x, y = q.popleft()
            for dx, dy in DIRS_XY:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.cols and 0 <= ny < self.rows):
                    continue
                if (nx, ny) in body or (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                q.append((nx, ny))
        return len(seen)
