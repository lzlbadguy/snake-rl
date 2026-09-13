"""
把训练好的策略导出成浏览器可用的 base64 float32 权重，并做一次无损性自检。

自检分两步（都很关键）：
  1. 把写出的 base64 反解回数组，用**纯 numpy** 重算一遍前向，与 torch 模型的贪心动作
     在整个状态分布上比对 —— 一致率 100% 才说明导出无损、且前向的层序/转置正确
  2. 用这份 numpy 前向在环境里跑贪心评估，分数应与 torch 版本一致

用法：
  .venv/bin/python export_policy.py --ckpt runs/ppo20/ckpt.pt --out runs/ppo20/policy.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np
import torch

from snake_env import VectorSnake, N_ACTIONS


def b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode()


def f32_b64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def np_forward(obs: np.ndarray, layers) -> np.ndarray:
    """纯 numpy 前向：y = act(x @ W.T + b)，W 为 (out, in) 行主序。"""
    x = np.asarray(obs, dtype=np.float32)
    for k, L in enumerate(layers):
        y = x @ L["W"].T + L["b"]
        if L["act"] == "tanh":
            y = np.tanh(y)
        x = y
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ppo20/ckpt.pt")
    ap.add_argument("--out", default="runs/ppo20/policy.json")
    ap.add_argument("--eval-episodes", type=int, default=300)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = ck["model"]
    print(f"checkpoint: {args.ckpt} (update {ck.get('update', '?')})")

    spec = [("trunk.0", "tanh"), ("trunk.2", "tanh"), ("pi", "linear")]
    layers = []
    for name, act in spec:
        w = sd[name + ".weight"].detach().cpu().numpy().astype(np.float32)
        b = sd[name + ".bias"].detach().cpu().numpy().astype(np.float32)
        layers.append({"name": name, "in": int(w.shape[1]), "out": int(w.shape[0]),
                       "act": act, "w": b64_f32(w), "b": b64_f32(b)})
        print(f"  {name:<9} {w.shape} act={act:<6} 参数量 {w.size + b.size}")

    obs_dim = layers[0]["in"]
    policy = {"obs_dim": obs_dim, "hidden": layers[0]["out"], "n_actions": N_ACTIONS,
              "layers": layers,
              "obs_layout": {
                  "scalars": ["danger_front", "danger_left", "danger_right",
                              "wall_up", "wall_down", "wall_left", "wall_right",
                              "dir_onehot(up,right,down,left)",
                              "food_dx_norm", "food_dy_norm",
                              "food_dir_bits(up,down,left,right)",
                              "food_same_row", "food_same_col",
                              "bfs_dist_norm", "food_reachable",
                              "reachable_area_frac", "length_frac"],
                  "patch": "3 channels x 7x7, channel-major: wall, food, body; "
                           "row index increases forward, col index increases to the snake's right",
                  "actions": "0=straight, 1=turn left, 2=turn right"},
              "meta": {"chkpt_update": ck.get("update"), "trained_with": ck.get("args", {})}}
    with open(args.out, "w") as f:
        json.dump(policy, f)
    print(f"已写出 {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")

    # ---------- 自检 1：base64 -> numpy 前向 vs torch 前向 ----------
    layers_np = [{"W": f32_b64(L["w"]).reshape(L["out"], L["in"]),
                  "b": f32_b64(L["b"]), "act": L["act"]} for L in policy["layers"]]
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(obs_dim, layers[0]["out"]), nn.Tanh(),
                                       nn.Linear(layers[0]["out"], layers[1]["out"]), nn.Tanh())
            self.pi = nn.Linear(layers[1]["out"], N_ACTIONS)

        def forward(self, x):
            return self.pi(self.trunk(x))

    net = Net()
    # 只要 trunk/pi（价值头 v.* 是推理时不需要的）
    net.load_state_dict({k: v for k, v in sd.items() if k.startswith(("trunk.", "pi."))})
    net.eval()

    env = VectorSnake(n_envs=256, seed=7, use_patch=True)
    obs = env.observe()
    agree = tot = 0
    maxdiff = 0.0
    for _ in range(40):
        a_np = np_forward(obs, layers_np).argmax(axis=1)
        with torch.no_grad():
            logits_t = net(torch.from_numpy(obs)).numpy()
        maxdiff = max(maxdiff, float(np.abs(logits_t - np_forward(obs, layers_np)).max()))
        agree += int((a_np == logits_t.argmax(axis=1)).sum())
        tot += obs.shape[0]
        obs, _, _, _, _ = env.step(a_np.astype(np.int64))
    print(f"自检1  numpy(导出权重) vs torch 贪心动作一致率: {agree}/{tot} = "
          f"{100 * agree / tot:.3f}%   最大 logit 差 {maxdiff:.2e}")

    # ---------- 自检 2：导出权重在环境里的贪心评估 ----------
    n = args.eval_episodes
    ev = VectorSnake(n_envs=64, seed=99, use_patch=True, food_reward=0.0,
                     death_penalty=0.0, step_penalty=0.0, shaping_scale=0.0)
    obs = ev.observe()
    scores, causes = [], []
    while len(scores) < n:
        a = np_forward(obs, layers_np).argmax(axis=1)
        obs, _, _, _, info = ev.step(a.astype(np.int64))
        for i in np.flatnonzero(info["finished"]):
            scores.append(int(info["final_score"][i]))
            causes.append(int(info["cause"][i]))
    s = np.array(scores[:n])
    c = np.array(causes[:n])
    print(f"自检2  导出权重贪心评估 {n} 局: 均分 {s.mean():.2f} 中位 {np.median(s):.0f} "
          f"最高 {s.max()} | 墙 {(c==0).mean()*100:.0f}% 己 {(c==1).mean()*100:.0f}% "
          f"上限 {(c==2).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
