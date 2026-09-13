# Snake RL · 项目状态（2026-09-13 收工）

> 一句话现状：**最佳策略 `runs/ppo20/ckpt.pt`（裸 62.47 / 加载安全盾 105.66），已部署进 `/home/lzl/snake_ai.html`；原始版 `/home/lzl/snake.html` 未改动。**
> 所有训练/评估/服务进程已停止，机器无残留。详细踩坑记录见 skill **`snake-rl`**（`~/.hermes/skills/software-development/snake-rl/`）。

---

## 1. 目标回顾

| 阶段 | 问题 | 结论 |
|---|---|---|
| 原始需求 | 写一个方向键可玩的贪吃蛇 | ✅ `/home/lzl/snake.html`（20×20、不穿墙、localStorage 最高分） |
| 训练模型玩它 | PPO 能不能学会 | ✅ 55.3M 步 → 均分 62.79（比 30 行启发式的 64~73 略低） |
| 能不能吃满（397） | 分三层看 | ① 图论能保证 ✅ ② 纯 RL 不能 ✅ ③ 现实是混合方案 |

**两条路线定名**：
- **A（图论）**：Hamilton 回路跟随 → 保证吃满，但不看食物、绕远路
- **B（RL + 安全层）**：学出来的策略负责效率，安全层负责不死

---

## 2. 结果账本

### A. 图论上界（`hamilton.py`）

| 指标 | 值 |
|---|---|
| 吃满率（满分 397） | **100%**（64/64 局，零死亡） |
| 吃满步数 | 均值 **39,802** / 中位 39,786 / 最少 36,836 / 最多 42,720 |
| 每颗食物步数 | 100.3 |
| 构造 | 偶数行 2 行条带 + 第 0 列回收通道；**行 10 天然含 (3,10)→(4,10)→(5,10)，与游戏初始身体对齐**（代码里有断言） |

对照：PPO 最优 62.8、启发式 64~73 → 都吃不满。**吃满靠图论，效率靠 RL。**

### B. RL + 安全盾（20×20，200 局，seed=777，贪心）

| 训练方式 | 步数 | 裸策略 | 旧盾(cap1600/4000) | **两级盾(cap4000)** |
|---|---:|---:|---:|---:|
| **ppo20（基线奖励）** | 55.3M | **62.47** | 105.20 | **105.66** 👑 |
| ppo20_both（A/B 胜出奖励） | 60.0M | 27.71 | 84.57 | 90.58 |
| cur_g20（掩码课程，8→20 迁移） | 12.1M | **3.09** | 84.11 | 89.03 |
| ab_both（20M 步） | 20.2M | 55.85 | 73.84 | 89.69 |
| mask_cap4000_warm | 26.2M | — | — | 84.69（自评） |

最佳 ckpt 的细节：
- 结构 `170 → 256 → 256 → 3`（tanh），110,596 参数，3 个相对动作
- 观测 = 23 标量（危险3/到墙4/朝向4/食物位移+方向6/同行列2/BFS距离/可达性/可达面积/长度）+ 7×7×3 patch（wall/food/body，按朝向旋转）
- 带盾表现：中位 104 / 最高 147 / p10 80 / 均步数 2627；死因 墙 58.5% 己 27%

### C. 游戏内实测（`snake_ai.html`，无回合上限）

| 配置 | 局数 | 均分 | 最高 | 每局步数 |
|---|---:|---:|---:|---:|
| 安全盾 **关** | 19 | 66.16 | 101 | 1228 |
| 安全盾 **开** | 3 | **104** | 125 | 2604 |

### D. GPU 加速（环境 + 网络都在 GPU）

| 环境数 | 安全层 | numpy(CPU) | torch(GPU) | 加速 |
|---:|---|---:|---:|---:|
| 1024 | 关 | 64,031 | 252,857 | 3.9x |
| 1024 | 开 | 16,238 | 87,597 | 5.4x |
| 2048 | 关 | 65,047 | 459,576 | 7.1x |
| 4096 | 关 | 66,586 | **653,049** | 9.8x |
| 4096 | 开 | 16,709 | 196,418 | **11.8x** |

- 网络（batch 16384）前向+反向：CPU 246k → GPU **5.95M** 样本/秒（**24x**）
- **端到端**：10~15k 步/秒（纯 CPU）→ **48~139k 步/秒**（GPU）
- 一轮 30M 步实验：~35 分钟 → **~5 分钟**

### E. 弧盾（一项做了但没做成的探索）

| 目标 | 结果 |
|---|---|
| **保证不死** | ✅ 1.28M 步逐步校验不变量 0 违反；3.84M 步零死亡；可行动作集为空 0.0000%；判据 O(1) **不需要 BFS** |
| 保证吃满 | ❌ 吃满率 **0%**（200 局，上限 6 万步） |
| 保住策略效率 | ❌ 105 → **5.7**（每颗食物 24.6 → ~10,500 步，慢 400 倍） |
| 训练时进弧盾 | ❌ 13.1M 步训练态 49.45，贪心评估 27.54（中位 6，99% 撞上限） |

**判据（已实现，`shield_mode="arc"`）**：`alive ∧ d(T,q) ≥ d(T,H) ∧ (d(T,q)-d(T,H)) ≤ arc_slack`，T 为弧起点、只前进。缺的最后一块是**经典 shortcut 算法的 body 回路序 + reserved 格显式记账**（O(L) 而非 O(1)），这正是防止"弧膨胀 ⇒ 可行集被压死"的东西。

---

## 3. 三条负面结果（价值不低于正面）

1. **A/B 胜出的奖励配置，训到 60M 步反而训坏**：`area 0.2 + doomed 0.3` 在 20M 步时 +4.64 分（t≈7），60M 步下裸策略 27.71 vs 基线 62.47（自撞率 100%）⇒ **A/B 结论只在相同训练预算内成立**。
2. **弧盾是口"安全棺材"**：不死 ✅ 但不可用 ❌（见上表 E）。
3. **我自己提的 O(1)"可证明不死"理论被证伪**（`check_cycle_shield.py` 第 3378 步，随机策略压力测试）⇒ 根因：判据的度量基准不能是**会移动的量**（那一版用当前尾巴 T 作基准，尾巴每步移动 ⇒ 序关系在 `(v-u) mod N` 处翻转）。

---

## 4. 文件地图

| 文件 | 作用 |
|---|---|
| `snake_env.py` | numpy 向量化环境（规则/观测/奖励/盾）；`shield_mode="tail"`（默认）或 `"arc"`（实验） |
| `snake_env_torch.py` | **GPU 常驻环境**，与 numpy 版逐位一致；BFS 用位移+按位与，每 8 层一次同步 |
| `train.py` | PPO；`--env-backend torch --device cuda`、`--mask-safety`、`--shield-mode`、`--init-ckpt` |
| `bench_env.py` / `bench_torch_env.py` / `bench_gpu.py` | 吞吐基准 |
| `test_parity.py` / `test_torch_parity.py` / `test_mask_isolation.py` | 三层对拍：env↔参考实现 / numpy↔torch / 盾零副作用 |
| `check_mask.py` | 盾的对抗测试（拿可证明安全的 Hamilton 动作撞盾，必须 0 挡） |
| `check_cycle_shield.py` | 弧盾的**随机策略压力测试 + 逐步不变量校验** |
| `check_mask_freshness.py` | 盾是否对应当前状态 |
| `hamilton.py` | Hamilton 回路构造 + 跟随器（A 的证据） |
| `baseline.py` | 启发式基线（BFS 追食物 + 尾巴安全检查 + tail-chase） |
| `export_policy.py` → `build_game.py` | 导出权重（base64 f32 + numpy 自检）→ 生成自包含游戏 |
| `eval_any_ckpt.py` / `eval_fill.py` / `eval_ckpts.py` | 多盾对照评估 / 吃满评估 / 多 ckpt 同种子评估 |
| `game/snake_template.html` | 游戏模板（AI 托管 + 速度 + 安全层开关） |
| `game/{parity_server.js,check_js_parity.py,make_test_page.py,...}` | 浏览器↔Python 逐位对拍链 |
| `game/js_keys_ai_inject.js` | 键盘路径实测（派发真实 KeyboardEvent） |

**占位符**：`runs/ppo20/`（最佳，含 `policy.json` = 游戏内嵌权重）、`runs/ab_*/`、`runs/cur_g*/`、`runs/arc_rl_warm/`、`runs/ppo20_both/`（60M 步，无 summary，已被否）。

---

## 5. 复现命令

```bash
cd /home/lzl/snake-rl

# 环境对拍（numpy vs 参考转写 / numpy vs torch / 盾零副作用）
.venv/bin/python test_parity.py 5000
.venv/bin/python test_torch_parity.py 3000          # 含弧盾，必须"全部一致"
.venv/bin/python test_mask_isolation.py

# 盾的正确性
.venv/bin/python check_mask.py                      # 对抗测试：必须 0% 挡
.venv/bin/python check_cycle_shield.py 20000 64     # 弧盾：逐步校验不变量

# 图论上界
.venv/bin/python hamilton.py 64                     # 100% 吃满，平均 ~39,802 步

# 训练（GPU，约 4~10 分钟一轮 30M 步）
.venv/bin/python train.py --device cuda --env-backend torch --grid 20 --n-envs 2048 \
  --rollout 128 --train-steps 30000000 --minutes 25 --max-steps 4000 \
  --shaping 0.25 --eval-every 10 --eval-episodes 200 --out runs/xxx

# 评估任意 ckpt（无盾 / 旧盾 / 两级盾，同种子）
.venv/bin/python eval_any_ckpt.py runs/ppo20/ckpt.pt 200 20

# 吃满评估
.venv/bin/python eval_fill.py runs/ppo20/ckpt.pt 200 60000 arc

# 导出权重 → 重建游戏（会重写 /home/lzl/snake_ai.html）
.venv/bin/python export_policy.py runs/ppo20/ckpt.pt
.venv/bin/python build_game.py --out /home/lzl/snake_ai.html --no-backup

# 浏览器↔Python 逐位对拍
node game/parity_server.js 8733 &
.venv/bin/python game/make_test_page.py /home/lzl/snake_ai.html game/_js_test.html
MOZ_HEADLESS=1 ~/.cache/ms-playwright/firefox-1532/firefox/firefox --headless \
  --new-instance http://127.0.0.1:8733/_js_test.html
.venv/bin/python game/check_js_parity.py
```

---

## 6. 关键常量（训练/游戏必须一致）

- 20×20、不穿墙、初始 3 节 `(5,10)(4,10)(3,10)` 朝右、**尾巴正在让位的那格不算碰撞**
- 食物在空闲格按 y 主序均匀随机；满分 397
- 观测 170 维；动作 3 个相对动作（直行/左转/右转，天然禁 180°）
- 死因编码：`CAUSE_WALL, CAUSE_SELF, CAUSE_CAP, CAUSE_WIN = 0, 1, 2, 3`（**脚本里最容易写反的一处**）
- 超参：n_envs 2048、rollout 128、hidden 256、γ0.99 / λ0.95 / clip0.2 / lr3e-4、entropy 退火、`--shaping 0.25`、`--area-shaping 0`、`--doomed-penalty 0`

---

## 7. 下一步（按性价比排序）

1. **训练回合上限 1600 → 4000 + 保持基线奖励 + GPU**（唯一被数据支持的方向）
   现有证据：放开上限让 ppo20 从 78.63 → 105.19（+34%），而奖励塑形的改动已被 60M 步证伪。
2. **真正"保证吃满"**：补经典 shortcut 算法的 **reserved-cells 记账**（body 回路序 + 被跳过格），规格已精确，验证台现成（`check_cycle_shield.py`）。
3. **治本"注定态"**：加前瞻观测特征（三个候选动作各自走完后的可达面积/尾巴可达性），或 action mask 化的安全集合；诊断数据：77.5% 的回合进入过注定态，死前 5~10 步才收口。
4. 把两级盾也做进游戏 JS（目前游戏里是最初版安全层，`safetyFlags()` 已是两级，但游戏内的"最佳组合"尚未重跑对照）。

---

## 8. 别踩的坑（详见 skill `snake-rl`）

1. 塑形梯度被步惩罚抵消（shaping 0.05 → 0.25 是最大单点杠杆）
2. GAE 对**截断**回合也把 bootstrap 归零（掩码训练永远学不动）
3. numpy 小数组开销：n=1 的向量化比纯 Python 慢 436 倍 ⇒ 并行是前提
4. 掩码会改变回合结构（回合数少 80 倍 ⇒ 学不动）⇒ 需要棋盘尺寸课程
5. **安全检查的目标格必须是"当前尾巴"那格**（写成新尾巴会让判据恒 False）
6. 把"死亡原因"当 bug 修，可能只是推迟死亡（撞墙是"位置已输"的症状）
7. 别假设"小网络 GPU 更慢"——GB10 实测快 24 倍
8. O(1) 的"可证明安全"不变量要有**随机策略压力测试**才敢信
9. "不变量被保持" ≠ "系统会前进"：两项必须分别设指标

**方法论**：任何一个"我证出来了 / 我优化好了"的结论，先写测试让它去撞；今天 4 次靠这个习惯抓住了真 bug。
