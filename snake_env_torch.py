"""
GPU 版向量化环境（torch），规则 / 观测 / 奖励 / 安全层与 snake_env.py 的 numpy 版逐项一致。

为什么要写：numpy 版里 BFS/flood-fill 占环境开销 96%，而它是**单线程**的
（1024 环境约 17.5k 步/秒封顶）。同规模的 BFS 搬到 GPU 上单项快 15 倍
（见 bench_gpu.py：26 → 389 次/秒）。

设计要点：
  * 全部状态常驻显存：body 环形缓冲 / occ / 食物 / 计数器都是 torch 张量，
    训练时观测不用再 H2D 拷贝（obs 本来就在卡上）
  * BFS 用"四向位移 + 按位与"逐层扩散；**每 8 层才做一次 .any() 检查**
    （实测：固定层数不做逐层同步比逐层早停快 1.7 倍，块状检查兼顾两者）
  * 支持注入均匀随机流（uniforms），用于与 numpy 版 / 浏览器 JS 版做逐位对拍
  * 小数组信息（分数、死因等）打包成**一个**张量再 .cpu()，把每步同步压到 1 次

用法（对拍）：
    .venv/bin/python test_torch_parity.py
"""

from __future__ import annotations

import numpy as np
import torch

DY = torch.tensor([-1, 0, 1, 0], dtype=torch.int16)
DX = torch.tensor([0, 1, 0, -1], dtype=torch.int16)
TURN = torch.tensor([0, -1, 1], dtype=torch.int16)   # 0=直行 1=左转 2=右转
N_ACTIONS = 3

CAUSE_WALL, CAUSE_SELF, CAUSE_CAP, CAUSE_WIN = 0, 1, 2, 3

PATCH_R = 3
PATCH_K = 2 * PATCH_R + 1
N_SCALAR = 23


def build_patch_offsets(device="cpu"):
    """PATCH_OFF[d, i, j] = (dy, dx)：朝向 d 下 patch 第 i 行第 j 列相对头部的偏移。
    行沿前进方向增大（patch[0] = 前方 3 格），列沿蛇的右手方向增大。"""
    off = torch.zeros((4, PATCH_K, PATCH_K, 2), dtype=torch.int16, device=device)
    for d in range(4):
        fy, fx = int(DY[d]), int(DX[d])
        ry, rx = int(DY[(d + 1) % 4]), int(DX[(d + 1) % 4])
        for i in range(PATCH_K):
            for j in range(PATCH_K):
                f, s = i - PATCH_R, j - PATCH_R
                off[d, i, j, 0] = f * fy + s * ry
                off[d, i, j, 1] = f * fx + s * rx
    return off


class VectorSnakeTorch:
    def __init__(self, n_envs=1024, rows=20, cols=20, max_steps=None, seed=0,
                 use_patch=True, gamma=0.99, food_reward=1.0, death_penalty=1.0,
                 step_penalty=0.001, shaping_scale=0.05, area_shaping=0.0,
                 doomed_penalty=0.0, safety_mask=False, tier_shield=True, shield_mode="tail",
                 device="cuda", uniforms=None,
                 bfs_block=8, bfs_max_layers=420):
        self.dev = torch.device(device)
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
        self.tier_shield = bool(tier_shield)
        self.shield_mode = str(shield_mode)
        if self.shield_mode in ("cycle", "arc"):
            from hamilton import build_cycle
            cyc = build_cycle(self.rows, self.cols)
            idx = np.zeros(self.cells, dtype=np.int64)
            cel = np.zeros(len(cyc), dtype=np.int64)
            for i, (x, y) in enumerate(cyc):
                idx[y * self.cols + x] = i
                cel[i] = y * self.cols + x
            self.cyc_idx = torch.as_tensor(idx, device=self.dev)
            self.cyc_cell = torch.as_tensor(cel, device=self.dev)
            self.cyc_n = len(cyc)
        else:
            self.cyc_idx = None
            self.cyc_cell = None
            self.cyc_n = 0
        self.arc_T = torch.zeros(self.n, dtype=torch.int32, device=self.dev)
        self.tier = torch.zeros(self.n, dtype=torch.int8, device=self.dev)
        self.max_dim = max(self.rows, self.cols)
        self.phi_scale = 2.0 * self.max_dim
        self.bfs_block = int(bfs_block)
        self.bfs_max_layers = int(bfs_max_layers)
        self.obs_dim = N_SCALAR + (3 * PATCH_K * PATCH_K if self.use_patch else 0)

        n = self.n
        d = self.dev
        self.arange = torch.arange(n, device=d)
        self.DYr = DY.to(d)
        self.DXr = DX.to(d)
        self.TURNr = TURN.to(d)
        self.PATCH_OFF = build_patch_offsets(d)

        self.body = torch.zeros((n, self.max_len), dtype=torch.int32, device=d)
        self.head_ptr = torch.zeros(n, dtype=torch.int32, device=d)
        self.length = torch.zeros(n, dtype=torch.int32, device=d)
        self.occ = torch.zeros((n, self.cells), dtype=torch.bool, device=d)
        self.dir = torch.zeros(n, dtype=torch.int16, device=d)
        self.food = torch.full((n,), -1, dtype=torch.int32, device=d)
        self.steps = torch.zeros(n, dtype=torch.int32, device=d)
        self.score = torch.zeros(n, dtype=torch.int32, device=d)
        self.phi = torch.zeros(n, dtype=torch.float32, device=d)
        self.area = torch.zeros(n, dtype=torch.int32, device=d)
        self.doomed = torch.zeros(n, dtype=torch.bool, device=d)
        self.safe = torch.ones((n, 3), dtype=torch.bool, device=d)
        self.area3 = torch.zeros((n, 3), dtype=torch.int32, device=d)

        self.generator = torch.Generator(device=d)
        self.generator.manual_seed(int(seed))
        self.uniforms = None
        if uniforms is not None:
            self.uniforms = torch.as_tensor(np.asarray(uniforms), dtype=torch.float64, device=d)
        self.uni_ptr = 0
        self.reset()

    # ---------------- 基础 ----------------

    def _segment(self, k):
        return self.body.gather(1, ((self.head_ptr - k) % self.max_len).unsqueeze(1)).squeeze(1)

    def _draw_uniforms(self, m):
        if m <= 0:
            return torch.zeros(0, dtype=torch.float64, device=self.dev)
        if self.uniforms is not None:
            u = self.uniforms[self.uni_ptr:self.uni_ptr + m]
            self.uni_ptr += m
            return u
        return torch.rand(m, dtype=torch.float64, device=self.dev, generator=self.generator)

    def reset(self, ids=None):
        dev = self.dev
        ids = self.arange if ids is None else ids
        if ids.numel() == 0:
            return
        cy = self.rows // 2
        head = cy * self.cols + 5
        for k, cell in enumerate((head, head - 1, head - 2)):
            self.body[ids, (-k) % self.max_len] = cell
        if self.shield_mode == "arc":
            cells3 = torch.tensor([head, head - 1, head - 2], dtype=torch.long, device=dev)
            self.arc_T[ids] = self.cyc_idx[cells3].min().to(torch.int32)
        self.head_ptr[ids] = 0
        self.length[ids] = 3
        self.dir[ids] = 1
        self.steps[ids] = 0
        self.score[ids] = 0
        self.occ[ids] = False
        for k in range(3):
            self.occ[ids, head - k] = True
        self.food[ids] = -1
        self._place_food(ids)

    def _place_food(self, ids):
        """空闲格均匀采样（y 主序），返回"无处可放"（填满棋盘）的掩码。"""
        ids = ids.reshape(-1)
        m = int(ids.numel())
        if m == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.dev)
        u = self._draw_uniforms(m)
        free = ~self.occ[ids]                                   # (m, cells)
        cnt = free.sum(dim=1).long()
        cum = torch.cumsum(free.to(torch.int32), dim=1)          # (m, cells)
        thr = torch.floor(u * cnt.double()).to(torch.int32)
        pick = torch.argmax((cum > thr.unsqueeze(1)).to(torch.int8), dim=1).to(torch.int32)
        no_space = cnt == 0
        self.food[ids] = torch.where(no_space, torch.full_like(pick, -1), pick)
        return no_space

    # ---------------- BFS ----------------

    def _bfs(self, free, src):
        """free: (n,R,C) bool（起点须可通行）；src: (n,) cell。
        返回 (vis (n,R,C) bool, dist (n,R,C) int16)。块状检查以减少同步。"""
        n, R, C = free.shape
        ar = self.arange
        sy, sx = src // self.cols, src % self.cols
        vis = torch.zeros_like(free)
        fr = torch.zeros_like(free)
        fr[ar, sy, sx] = True
        vis |= fr
        dist = torch.zeros_like(free, dtype=torch.int16)
        d = 0
        while True:
            for _ in range(self.bfs_block):
                nb = torch.zeros_like(fr)
                nb[:, 1:, :] |= fr[:, :-1, :]
                nb[:, :-1, :] |= fr[:, 1:, :]
                nb[:, :, 1:] |= fr[:, :, :-1]
                nb[:, :, :-1] |= fr[:, :, 1:]
                nb &= free & ~vis
                d += 1
                dist[nb] = d
                vis |= nb
                fr = nb
            if (not bool(fr.any())) or d >= self.bfs_max_layers:
                break
        return vis, dist

    # ---------------- 观测 ----------------

    def observe(self):
        dev = self.dev
        n, cells = self.n, self.cells
        ar = self.arange
        if self.safety_mask:
            if self.shield_mode == "arc":
                self._arc_advance()
                self.safe, self.tier = self._arc_mask()
            elif self.shield_mode == "cycle":
                self.safe, self.tier = self._cycle_mask()
            else:
                strict, alive, self.area3 = self.safety_flags()
                self.safe, self.tier = self._shield_tiers(strict, alive)

        head_cell = self.body.gather(1, self.head_ptr.unsqueeze(1)).squeeze(1)
        hy = (head_cell // self.cols).to(torch.int16)
        hx = (head_cell % self.cols).to(torch.int16)
        occ3 = self.occ.view(n, self.rows, self.cols)

        free = ~occ3
        free[ar, hy.long(), hx.long()] = True
        vis, dist = self._bfs(free, head_cell)
        self.area = vis.reshape(n, -1).sum(dim=1).to(torch.int32)
        area_frac = (self.area.float() / cells)

        has_food = self.food >= 0
        fy = torch.where(has_food, self.food // self.cols, torch.zeros_like(self.food)).to(torch.int16)
        fx = torch.where(has_food, self.food % self.cols, torch.zeros_like(self.food)).to(torch.int16)
        fcell = torch.where(has_food, self.food, torch.full_like(self.food, -1)).long()
        reached = torch.zeros(n, dtype=torch.bool, device=dev)
        if bool(has_food.any()):
            fy_c = fy.clamp(0, self.rows - 1).long()
            fx_c = fx.clamp(0, self.cols - 1).long()
            reached = vis[ar, fy_c, fx_c] & has_food
        bfs_d = torch.where(reached, dist[ar, fy.clamp(0, self.rows - 1).long(),
                                             fx.clamp(0, self.cols - 1).long()].float(),
                            torch.full((n,), -1.0, device=dev))
        bfs_norm = torch.where(bfs_d < 0, torch.ones_like(bfs_d),
                              (bfs_d / self.phi_scale).clamp(0.0, 1.0))

        # 势函数（potential-based：策略不变）
        self.phi = (-self.shaping_scale * bfs_norm - self.area_shaping * (1.0 - area_frac)).to(torch.float32)

        md = float(self.max_dim)
        hyf, hxf = hy.float(), hx.float()
        wall = torch.stack([hyf / md, (self.rows - 1 - hyf) / md,
                            hxf / md, (self.cols - 1 - hxf) / md], dim=1)

        # 三个相对动作的即时危险（用 occ，含头；与 numpy 版一致）
        danger = torch.zeros((n, 3), dtype=torch.float32, device=dev)
        for a in range(3):
            nd = ((self.dir + self.TURNr[a]) % 4).long()
            ny = hy + self.DYr[nd]
            nx = hx + self.DXr[nd]
            out = (ny < 0) | (ny >= self.rows) | (nx < 0) | (nx >= self.cols)
            nyc = ny.clamp(0, self.rows - 1).long()
            nxc = nx.clamp(0, self.cols - 1).long()
            c = (nyc * self.cols + nxc)
            hit = self.occ.gather(1, c.unsqueeze(1)).squeeze(1)
            danger[:, a] = (out | hit).float()

        dir_oh = torch.zeros((n, 4), dtype=torch.float32, device=dev)
        dir_oh[ar, self.dir.long()] = 1.0

        ddx = torch.where(has_food, (fx - hx).float(), torch.zeros(n, device=dev)) / (self.cols - 1)
        ddy = torch.where(has_food, (fy - hy).float(), torch.zeros(n, device=dev)) / (self.rows - 1)
        fdir = torch.stack([(fy < hy).float(), (fy > hy).float(),
                            (fx < hx).float(), (fx > hx).float()], dim=1)
        aligned = torch.stack([(fy == hy).float(), (fx == hx).float()], dim=1)

        scalars = torch.cat([danger, wall, dir_oh,
                             torch.stack([ddx, ddy], dim=1), fdir, aligned,
                             bfs_norm.unsqueeze(1),
                             reached.float().unsqueeze(1),
                             area_frac.unsqueeze(1),
                             (self.length.float() / cells).unsqueeze(1)], dim=1).to(torch.float32)
        if not self.use_patch:
            return scalars

        off = self.PATCH_OFF[self.dir.long()]                     # (n,K,K,2)
        oy = hy.unsqueeze(1).unsqueeze(1) + off[..., 0]
        ox = hx.unsqueeze(1).unsqueeze(1) + off[..., 1]
        valid = (oy >= 0) & (oy < self.rows) & (ox >= 0) & (ox < self.cols)
        oyc = oy.clamp(0, self.rows - 1).long()
        oxc = ox.clamp(0, self.cols - 1).long()
        pc = (oyc * self.cols + oxc).view(n, -1)                  # (n,K*K)
        body_p = self.occ.gather(1, pc).view(n, PATCH_K, PATCH_K) & valid
        food_p = (pc == fcell.unsqueeze(1)) & valid.view(n, -1)
        wall_p = ~valid
        # 通道顺序 wall, food, body —— 与 numpy 版 concatenate(axis=1)+reshape 的展平顺序一致
        patch = torch.stack([wall_p, food_p.view(n, PATCH_K, PATCH_K),
                             body_p], dim=1)                      # (n,3,K,K)
        return torch.cat([scalars, patch.reshape(n, -1).float()], dim=1).to(torch.float32)

    # ---------------- 安全层 ----------------

    def safety_flags(self):
        """(safe (n,3) bool, areas (n,3))：与 numpy 版 safety_flags() 逐项一致。
        判据：活着 且（**当前尾巴那一格**仍可达 或 可达面积 ≥ 走完后的长度）。"""
        n = self.n
        ar = self.arange
        head = self.body.gather(1, self.head_ptr.unsqueeze(1)).squeeze(1)
        hy = (head // self.cols).to(torch.int16)
        hx = (head % self.cols).to(torch.int16)
        tail_now = self._segment(self.length - 1)
        ty = (tail_now // self.cols).long()
        tx = (tail_now % self.cols).long()
        occ3 = self.occ.view(n, self.rows, self.cols)

        strict = torch.zeros((n, 3), dtype=torch.bool, device=self.dev)
        alive_flags = torch.zeros((n, 3), dtype=torch.bool, device=self.dev)
        areas = torch.zeros((n, 3), dtype=torch.int32, device=self.dev)
        for a in range(3):
            nd = ((self.dir + self.TURNr[a]) % 4).long()
            ny = hy + self.DYr[nd]
            nx = hx + self.DXr[nd]
            out = (ny < 0) | (ny >= self.rows) | (nx < 0) | (nx >= self.cols)
            nyc = ny.clamp(0, self.rows - 1).long()
            nxc = nx.clamp(0, self.cols - 1).long()
            ncell = (nyc * self.cols + nxc)
            ate = ncell == self.food.long()
            vac = ~ate
            hit = self.occ.gather(1, ncell.unsqueeze(1)).squeeze(1)
            hit = hit & ~((ncell == tail_now.long()) & vac)
            alive = ~out & ~hit
            length_after = self.length + ate.to(torch.int32)

            occ = occ3.clone()
            occ[ar, ty, tx] = False        # 目标尾巴格（下一步会让开）
            occ[ar, nyc, nxc] = False      # 新头所在格（BFS 起点）
            vis, _ = self._bfs(~occ, ncell)   # 注意：_bfs 收的是"可通行"掩码，occ 是障碍
            area = vis.reshape(n, -1).sum(dim=1).to(torch.int32)
            tail_ok = vis[ar, ty, tx]
            alive_flags[:, a] = alive
            strict[:, a] = alive & (tail_ok | (area >= length_after))
            areas[:, a] = area
        return strict, alive_flags, areas

    def _shield_tiers(self, strict, alive):
        """两级安全盾 + 死局兜底（与 numpy 版逐项一致，切勿只改一边）。

        一级严格：尾巴可达 或 面积 ≥ 走完后的长度；
        二级降级：一级全空时只禁「这一步就会死」（出界/撞身体）的动作；
        三级死局：连二级都空才放开全部。
        """
        any_strict = strict.any(dim=1, keepdim=True)
        zero = torch.zeros(self.n, dtype=torch.int8, device=self.dev)
        two = torch.full((self.n,), 2, dtype=torch.int8, device=self.dev)
        if not self.tier_shield:
            m = torch.where(any_strict, strict, torch.ones_like(strict))
            return m, torch.where(any_strict[:, 0], zero, two)
        any_alive = alive.any(dim=1, keepdim=True)
        m = torch.where(any_strict, strict, torch.where(any_alive, alive, torch.ones_like(strict)))
        tier = torch.where(any_alive[:, 0], torch.ones_like(two), two)
        tier = torch.where(any_strict[:, 0], zero, tier)
        return m, tier

    # ---------------- 弧盾（可证明不死；与 numpy 版逐项一致，改一边必须改另一边） ----------------

    def _arc_advance(self):
        """推进弧起点 T：T 指的格子空闲就释放并前进一格（T 只前进，均摊 O(1)）。"""
        for _ in range(self.cyc_n + 2):
            cell = self.cyc_cell[self.arc_T.long()]
            free = ~self.occ[self.arange, cell]
            if not bool(free.any()):
                break
            self.arc_T = torch.where(free, (self.arc_T + 1) % self.cyc_n,
                                     self.arc_T).to(torch.int32)

    def _arc_mask(self):
        """弧盾：身体 ⊆ [T, H] 的回路连续弧；可行动作 = alive ∧ d(T,q) ≥ d(T,H) ∧ d(T,q) ≤ N-2。

        证明与用法见 numpy 版 `VectorSnake._arc_mask` 的注释（含"沿回路前进一步恒可用"
        的论证）。判定 O(1)，不需要 BFS。
        """
        ar = self.arange
        n = self.n
        head = self.body.gather(1, self.head_ptr.unsqueeze(1)).squeeze(1)
        tp = ((self.head_ptr - self.length + 1) % self.max_len).unsqueeze(1)
        tail = self.body.gather(1, tp).squeeze(1)
        H = self.cyc_idx[head.long()]
        T = self.arc_T
        dTH = (H - T) % self.cyc_n
        hy = (head // self.cols).to(torch.int16)
        hx = (head % self.cols).to(torch.int16)
        ok = torch.zeros((n, 3), dtype=torch.bool, device=self.dev)
        alive_arr = torch.zeros((n, 3), dtype=torch.bool, device=self.dev)
        for a in range(3):
            nd = ((self.dir + self.TURNr[a]) % 4).long()
            ny = hy + self.DYr[nd]
            nx = hx + self.DXr[nd]
            out = (ny < 0) | (ny >= self.rows) | (nx < 0) | (nx >= self.cols)
            nyc = ny.clamp(0, self.rows - 1)
            nxc = nx.clamp(0, self.cols - 1)
            cell = (nyc * self.cols + nxc).long()
            ate = cell == self.food
            vac = ~ate
            hit = self.occ.gather(1, cell.unsqueeze(1)).squeeze(1) & ~((cell == tail) & vac)
            alive = ~(out | hit)
            dTq = (self.cyc_idx[cell] - T) % self.cyc_n
            alive_arr[:, a] = alive
            ok[:, a] = alive & (dTq >= dTH) & (dTq <= self.cyc_n - 2)
        any_ok = ok.any(dim=1, keepdim=True)
        any_alive = alive_arr.any(dim=1, keepdim=True)
        m = torch.where(any_ok, ok, torch.where(any_alive, alive_arr, torch.ones_like(ok)))
        zero = torch.zeros(n, dtype=torch.int8, device=self.dev)
        two = torch.full((n,), 2, dtype=torch.int8, device=self.dev)
        tier = torch.where(any_alive[:, 0], torch.ones_like(two), two)
        tier = torch.where(any_ok[:, 0], zero, tier)
        return m, tier

    # ---------------- 推进 ----------------

    def step(self, actions):
        """actions: (n,) 张量或 numpy 数组（相对动作 0/1/2）。
        返回 (obs, reward, terminated, truncated, info)，自动 reset。"""
        dev = self.dev
        n = self.n
        ar = self.arange
        a = torch.as_tensor(actions, device=dev).to(torch.int16).reshape(-1)

        self.dir = ((self.dir + self.TURNr[a.long()]) % 4).to(torch.int16)

        head_cell = self.body.gather(1, self.head_ptr.unsqueeze(1)).squeeze(1)
        hy = (head_cell // self.cols).to(torch.int16)
        hx = (head_cell % self.cols).to(torch.int16)
        ny = hy + self.DYr[self.dir.long()]
        nx = hx + self.DXr[self.dir.long()]
        out = (ny < 0) | (ny >= self.rows) | (nx < 0) | (nx >= self.cols)
        nyc = ny.clamp(0, self.rows - 1).long()
        nxc = nx.clamp(0, self.cols - 1).long()
        new_cell = (nyc * self.cols + nxc)

        tail_cell = self._segment(self.length - 1)
        ate_raw = new_cell == self.food.long()
        hit_self = self.occ.gather(1, new_cell.unsqueeze(1)).squeeze(1) \
            & ~((new_cell == tail_cell.long()) & ~ate_raw)
        dead = out | hit_self
        ate = ate_raw & ~dead

        r = ate.float() * self.food_reward
        r = r - self.step_penalty
        r = r - dead.float() * self.death_penalty

        # 状态推进
        rm = ~dead & ~ate
        self.occ[ar[rm], tail_cell[rm].long()] = False
        self.occ[ar, new_cell] = True
        new_ptr = (self.head_ptr + 1) % self.max_len
        self.body[ar, new_ptr] = new_cell.to(torch.int32)
        self.head_ptr = new_ptr
        self.length = self.length + ate.to(torch.int32)
        self.score = self.score + ate.to(torch.int32)
        self.steps = self.steps + 1

        win = torch.zeros(n, dtype=torch.bool, device=dev)
        if bool(ate.any()):
            idx = torch.nonzero(ate, as_tuple=False).squeeze(1)
            win[idx] = self._place_food(idx)

        truncated = self.steps >= self.max_steps
        terminated = dead | win
        finished = terminated | truncated

        cause = torch.where(dead & out, torch.full_like(self.steps, CAUSE_WALL),
                torch.where(dead & hit_self, torch.full_like(self.steps, CAUSE_SELF),
                torch.where(win, torch.full_like(self.steps, CAUSE_WIN),
                            torch.full_like(self.steps, CAUSE_CAP)))).to(torch.int32)

        # 打包成**一个**张量再回 CPU，把同步压到每步 1 次
        packed = torch.stack([finished.to(torch.int32), self.score, self.length, self.steps,
                              cause, self.doomed.to(torch.int32), ate.to(torch.int32)], dim=1)
        p = packed.cpu().numpy()

        keep = ~finished
        phi_prev = self.phi
        if bool(finished.any()):
            self.reset(torch.nonzero(finished, as_tuple=False).squeeze(1))

        obs = self.observe()
        r = r + (self.gamma * torch.where(keep, self.phi, torch.zeros_like(self.phi)) - phi_prev)

        self.doomed = (self.area < self.length) & keep
        if self.doomed_penalty > 0.0:
            r = r - self.doomed_penalty * self.doomed.float()

        info = {"finished": p[:, 0].astype(bool), "final_score": p[:, 1], "final_len": p[:, 2],
                "final_steps": p[:, 3], "cause": p[:, 4].astype(np.int8),
                "doomed": p[:, 5].astype(bool), "ate": p[:, 6].astype(bool)}
        return obs, r, terminated, truncated, info

    # ---------------- 调试 ----------------

    def body_cells(self, env):
        L = int(self.length[env])
        return [int(self.body[env, (self.head_ptr[env] - k) % self.max_len]) for k in range(L - 1, -1, -1)]

    def ascii(self, env):
        g = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for k in range(int(self.length[env])):
            c = int(self.body[env, (self.head_ptr[env] - k) % self.max_len])
            g[c // self.cols][c % self.cols] = "o" if k else "H"
        if int(self.food[env]) >= 0:
            f = int(self.food[env])
            g[f // self.cols][f % self.cols] = "*"
        return "\n".join("".join(row) for row in g)
