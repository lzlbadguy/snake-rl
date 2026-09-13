"""
PPO 训练向量化贪吃蛇（20x20，规则与 snake.html 一致）。

  观测: 23 维标量 + 7x7x3 自车坐标 patch（默认，--no-patch 可关）
  动作: 3 个相对动作（直行/左转/右转）
  奖励: 吃食物 +1、死亡 -1、每步 -0.001，外加 potential-based shaping
        r += γΦ(s') - Φ(s)，Φ = -0.05 * (BFS 距离/40)，策略不变
  算法: PPO，GAE(0.95)，多环境并行 rollout
  已掩码项: terminated 与 truncated 都不 bootstrap

用法示例：
  .venv/bin/python -u train.py --n-envs 2048 --rollout 128 --minutes 40 --out runs/ppo20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from snake_env import (VectorSnake, N_ACTIONS, CAUSE_WALL, CAUSE_SELF, CAUSE_CAP, CAUSE_WIN)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2.0))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=2048)
    p.add_argument("--rollout", type=int, default=128, help="每个环境每次更新采集的步数")
    p.add_argument("--updates", type=int, default=10 ** 9)
    p.add_argument("--minutes", type=float, default=40.0, help="墙钟预算（分钟）")
    p.add_argument("--train-steps", type=int, default=0,
                   help="按环境步数停止（>0 时优先于 --minutes，且 lr 调度按步数进度走）")
    p.add_argument("--grid", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=1600)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--no-patch", action="store_true")
    p.add_argument("--env-backend", type=str, default="numpy", choices=["numpy", "torch"],
                   help="numpy=环境在 CPU（原实现）；torch=环境常驻显存（BFS 也走 GPU，快 4~12 倍）")
    p.add_argument("--mask-safety", action="store_true",
                   help="启用安全层：把不安全动作屏蔽掉，PPO 只在安全动作里学")
    p.add_argument("--shield-mode", type=str, default="tail", choices=["tail", "arc", "cycle"],
                   help="tail=尾巴可达性检查（旧）；arc=回路连续弧（可证明不死，O(1) 判据）；cycle=已证伪的实验版")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--shaping", type=float, default=0.05)
    p.add_argument("--area-shaping", type=float, default=0.0,
                   help="势函数里『可达空间』项的权重（potential-based，不改变最优策略）")
    p.add_argument("--doomed-penalty", type=float, default=0.0,
                   help="进入『注定死亡』状态（可达格数<长度）时的每步惩罚")
    p.add_argument("--step-penalty", type=float, default=0.001)
    p.add_argument("--death-penalty", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-ckpt", type=str, default="",
                   help="从这里加载模型权重再继续训练（课程学习：棋盘从小到大迁移）")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="runs/ppo20")
    return p.parse_args()


def apply_mask(logits, mask):
    """把不安全动作的 logit 压到 -1e9。某个 env 三个动作都不安全时放开全部，避免 NaN。"""
    all_bad = ~mask.any(dim=-1, keepdim=True)
    m = mask | all_bad
    return logits.masked_fill(~m, -1e9)


def t2(x, dev):
    """numpy / 张量 -> 目标设备张量（本机 GPU 与 CPU 走统一内存，搬运成本可忽略）。"""
    if torch.is_tensor(x):
        return x.to(dev, non_blocking=True)
    return torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True)


def t2n(x):
    """张量 / numpy -> numpy（torch 后端把观测留在显存上，需要落回 CPU 时用）。"""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def _zero_reward(env):
    """评估只看策略行为，不需要奖励：把奖励项清零，少算一点 shaping。"""
    env.food_reward = 0.0
    env.death_penalty = 0.0
    env.step_penalty = 0.0
    env.shaping_scale = 0.0
    env.area_shaping = 0.0
    env.doomed_penalty = 0.0
    return env


def make_env(args, n_envs, seed):
    """按 --env-backend 构造环境：numpy 版常驻内存，torch 版常驻显存。"""
    kw = dict(n_envs=n_envs, rows=args.grid, cols=args.grid, max_steps=args.max_steps,
              seed=seed, use_patch=not args.no_patch, gamma=args.gamma,
              shaping_scale=args.shaping, area_shaping=args.area_shaping,
              doomed_penalty=args.doomed_penalty, safety_mask=args.mask_safety,
              shield_mode=args.shield_mode,
              step_penalty=args.step_penalty, death_penalty=args.death_penalty)
    if args.env_backend == "torch":
        from snake_env_torch import VectorSnakeTorch
        return VectorSnakeTorch(**kw, device=args.device)
    return VectorSnake(**kw)


def evaluate(model, args, n_episodes):
    """贪心策略评估（固定种子、无探索噪声），返回 (分数列表, 死因列表)。"""
    dev = next(model.parameters()).device
    env = make_env(args, 64, 10_000)
    env = _zero_reward(env)
    obs = env.observe()
    scores, causes = [], []
    while len(scores) < n_episodes:
        with torch.no_grad():
            logits, _ = model(t2(obs, dev))
            if args.mask_safety:
                logits = apply_mask(logits, t2(env.safe, dev))
            act = logits.argmax(dim=-1).cpu().numpy()
        obs, _, _, _, info = env.step(act)
        for i in np.flatnonzero(info["finished"]):
            scores.append(int(info["final_score"][i]))
            causes.append(int(info["cause"][i]))
    return scores[:n_episodes], causes[:n_episodes]


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count())
    dev = torch.device(args.device)

    env = make_env(args, args.n_envs, args.seed)
    obs_dim, n = env.obs_dim, args.n_envs
    model = ActorCritic(obs_dim, N_ACTIONS, args.hidden).to(dev)
    if args.init_ckpt:
        from_ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(from_ck["model"])
        print(f"[init] 已加载 {args.init_ckpt} (update {from_ck.get('update')})", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    T = args.rollout
    obs_buf = np.zeros((T, n, obs_dim), dtype=np.float32)
    act_buf = np.zeros((T, n), dtype=np.int64)
    logp_buf = np.zeros((T, n), dtype=np.float32)
    val_buf = np.zeros((T, n), dtype=np.float32)
    rew_buf = np.zeros((T, n), dtype=np.float32)
    term_buf = np.zeros((T, n), dtype=np.float32)
    trunc_buf = np.zeros((T, n), dtype=np.float32)
    mask_buf = np.ones((T, n, 3), dtype=bool)

    metrics_path = os.path.join(args.out, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["update", "env_steps", "wall_s", "sps", "ep_mean100",
                                "ep_max", "len_mean", "wall%", "self%", "cap%", "win%",
                                "pi_loss", "v_loss", "entropy", "clipfrac", "explvar", "lr",
                                "doomed%", "ep_doomed%"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[config] n_envs={n} rollout={T} grid={args.grid} obs_dim={obs_dim} "
          f"params={n_params} device={args.device} env={args.env_backend}", flush=True)
    print(f"[config] rollout 每次更新样本数 = {T * n:,}", flush=True)

    obs = env.observe()
    ep_scores, ep_lens = [], []
    cause_count = np.zeros(4, dtype=np.int64)
    doomed_steps = 0
    ep_finished = 0
    ep_doomed = 0
    was_doomed = np.zeros(n, dtype=bool)
    total_steps = 0
    t_start = time.time()
    update = 0

    while True:
        update += 1
        if (time.time() - t_start > args.minutes * 60 or update > args.updates
                or (args.train_steps and total_steps >= args.train_steps)):
            break
        t_upd = time.time()

        # ---------------- 采集 rollout ----------------
        last_info = None
        for t in range(T):
            obs_t = t2(obs, dev)
            with torch.no_grad():
                logits, value = model(obs_t)
                if args.mask_safety:
                    logits = apply_mask(logits, t2(env.safe, dev))
                dist = Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
            obs_buf[t] = t2n(obs)
            if args.mask_safety:
                mask_buf[t] = t2n(env.safe)     # 与 obs_buf[t] 对应的那一份掩码
            act_buf[t] = act.cpu().numpy()
            logp_buf[t] = logp.cpu().numpy()
            val_buf[t] = value.cpu().numpy()

            a_in = act if args.env_backend == "torch" else act.cpu().numpy()
            obs, rew, term, trunc, info = env.step(a_in)
            rew_buf[t] = t2n(rew) * args.reward_scale
            term_buf[t] = t2n(term).astype(np.float32)
            trunc_buf[t] = t2n(trunc).astype(np.float32)

            doomed_steps += int(info["doomed"].sum())
            was_doomed |= info["doomed"]
            fin = np.flatnonzero(info["finished"])
            if fin.size:
                ep_scores.extend(info["final_score"][fin].tolist())
                ep_lens.extend(info["final_len"][fin].tolist())
                for c in info["cause"][fin]:
                    cause_count[int(c)] += 1
                ep_finished += int(fin.size)
                ep_doomed += int(was_doomed[fin].sum())
                was_doomed[fin] = False
            last_info = info

        total_steps += T * n
        ep_scores_100 = ep_scores[-100:]

        # ---------------- GAE ----------------
        with torch.no_grad():
            _, last_v = model(t2(obs, dev))
        last_v = last_v.cpu().numpy()
        adv_buf = np.zeros_like(rew_buf)
        gae = np.zeros(n, dtype=np.float32)
        for t in range(T - 1, -1, -1):
            # 只有"真终局"（死亡/填满棋盘）才把 bootstrap 归零；被步数上限截断的回合
            # 必须用 V(s) 续上，否则价值函数会把每个回合的未来都当成 0 —— 掩码启用后
            # 所有回合都靠截断结束，这个差别是致命的（ev 会一直为负、策略学不动）
            nonterm = 1.0 - term_buf[t]
            next_v = last_v if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * next_v * nonterm - val_buf[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv_buf[t] = gae
        ret_buf = adv_buf + val_buf

        # ---------------- PPO 更新 ----------------
        b_obs = t2(obs_buf.reshape(-1, obs_dim), dev)
        b_mask = (t2(mask_buf.reshape(-1, 3), dev) if args.mask_safety else None)
        b_act = t2(act_buf.reshape(-1), dev)
        b_logp = t2(logp_buf.reshape(-1), dev)
        b_adv = t2(adv_buf.reshape(-1), dev)
        b_ret = t2(ret_buf.reshape(-1), dev)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n_samples = T * n

        if args.train_steps:
            # 按步数进度退火，保证不同配置在"同一步数"上可比
            progress = min(1.0, total_steps / args.train_steps)
        else:
            progress = min(1.0, (time.time() - t_start) / (args.minutes * 60))
        lr_now = max(args.lr * (1.0 - progress), args.lr * 0.05)
        for g in opt.param_groups:
            g["lr"] = lr_now
        ent_now = args.ent_coef * max(0.1, 1.0 - progress)

        pi_losses, v_losses, ents, clips = [], [], [], []
        for _ in range(args.epochs):
            perm = torch.randperm(n_samples, device=dev)
            for s in range(0, n_samples, args.minibatch):
                mb = perm[s:s + args.minibatch]
                logits, value = model(b_obs[mb])
                if b_mask is not None:
                    logits = apply_mask(logits, b_mask[mb])   # 更新时用同一份掩码
                dist = Categorical(logits=logits)
                logp = dist.log_prob(b_act[mb])
                ratio = torch.exp(logp - b_logp[mb])
                a = b_adv[mb]
                pg1 = -a * ratio
                pg2 = -a * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                pi_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * (value - b_ret[mb]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pi_loss + args.vf_coef * v_loss - ent_now * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())
                ents.append(ent.item())
                with torch.no_grad():
                    clips.append(((ratio - 1).abs() > args.clip).float().mean().item())

        with torch.no_grad():
            var_ret = b_ret.var()
            explvar = 1.0 - (b_ret - t2(val_buf.reshape(-1), dev)).var() / (var_ret + 1e-8)
            explvar = float(explvar)

        dt = time.time() - t_upd
        sps = (T * n) / dt
        tot = cause_count.sum()
        row = [update, total_steps, round(time.time() - t_start, 1), round(sps),
               round(float(np.mean(ep_scores_100)) if ep_scores_100 else 0.0, 2),
               float(np.max(ep_scores_100)) if ep_scores_100 else 0.0,
               round(float(np.mean(ep_lens[-100:])) if ep_lens else 0.0, 2),
               round(100 * cause_count[CAUSE_WALL] / tot, 1) if tot else 0.0,
               round(100 * cause_count[CAUSE_SELF] / tot, 1) if tot else 0.0,
               round(100 * cause_count[CAUSE_CAP] / tot, 1) if tot else 0.0,
               round(100 * cause_count[CAUSE_WIN] / tot, 1) if tot else 0.0,
               round(float(np.mean(pi_losses)), 4), round(float(np.mean(v_losses)), 3),
               round(float(np.mean(ents)), 4), round(float(np.mean(clips)), 4),
               round(explvar, 3), round(lr_now, 6),
               round(100.0 * doomed_steps / max(1, total_steps), 2),
               round(100.0 * ep_doomed / max(1, ep_finished), 1)]
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"upd {update:5d} | 步数 {total_steps/1e6:7.2f}M | {sps/1000:6.1f}k 步/秒 | "
              f"近100局均分 {row[4]:7.2f} 最高 {row[5]:4.0f} | 均长 {row[6]:6.1f} | "
              f"死因 墙{row[7]:4.1f}% 己{row[8]:4.1f}% | 注定态 {row[17]:4.1f}% 步 | "
              f"ent {row[13]:.3f} clip {row[14]:.3f} ev {row[15]:.2f}", flush=True)

        if args.eval_every and update % args.eval_every == 0:
            scores, causes = evaluate(model, args, args.eval_episodes)
            scores = np.array(scores)
            causes = np.array(causes)
            print(f"  [eval] {len(scores)} 局贪心: 均分 {scores.mean():.2f} 中位 "
                  f"{np.median(scores):.0f} 最高 {scores.max()} | "
                  f"墙 {(causes == CAUSE_WALL).mean() * 100:.0f}% "
                  f"己 {(causes == CAUSE_SELF).mean() * 100:.0f}% "
                  f"上限 {(causes == CAUSE_CAP).mean() * 100:.0f}%", flush=True)
            with open(os.path.join(args.out, "eval.csv"), "a", newline="") as f:
                csv.writer(f).writerow([update, total_steps, round(float(scores.mean()), 3),
                                        float(np.median(scores)), int(scores.max()),
                                        round(float((causes == CAUSE_WALL).mean()), 4),
                                        round(float((causes == CAUSE_SELF).mean()), 4),
                                        round(float((causes == CAUSE_CAP).mean()), 4)])
            torch.save({"model": model.state_dict(), "args": vars(args), "update": update},
                       os.path.join(args.out, "ckpt.pt"))

    # ---------------- 收尾 ----------------
    torch.save({"model": model.state_dict(), "args": vars(args), "update": update},
               os.path.join(args.out, "ckpt.pt"))
    scores, causes = evaluate(model, args, 500)
    scores = np.array(scores)
    causes = np.array(causes)
    summary = {
        "updates": update, "env_steps": total_steps,
        "wall_min": round((time.time() - t_start) / 60, 1),
        "eval_mean": round(float(scores.mean()), 2),
        "eval_median": float(np.median(scores)),
        "eval_max": int(scores.max()),
        "eval_p10": float(np.percentile(scores, 10)),
        "cause_wall": round(float((causes == CAUSE_WALL).mean()), 3),
        "cause_self": round(float((causes == CAUSE_SELF).mean()), 3),
        "cause_cap": round(float((causes == CAUSE_CAP).mean()), 3),
        "cause_win": round(float((causes == CAUSE_WIN).mean()), 3),
        "doomed_steps_pct": round(100.0 * doomed_steps / max(1, total_steps), 2),
        "doomed_episodes_pct": round(100.0 * ep_doomed / max(1, ep_finished), 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
