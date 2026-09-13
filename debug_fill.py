"""
定位矛盾："通关 100%" 但均分只有 5.7。
逐局打印 分数/长度/占据格数/食物/步数/结束标志/结束原因。
占据格数（occ 求和）如果接近 400，就说明占据图被写坏了 —— 那会让 _place_food 误判"无空格" ⇒ 假通关。
"""

import numpy as np
import torch

from snake_env import VectorSnake
from train import ActorCritic, N_ACTIONS, apply_mask

ck = torch.load("runs/ppo20/ckpt.pt", map_location="cpu", weights_only=True)
a = ck.get("args", {})
env = VectorSnake(n_envs=4, rows=20, cols=20, max_steps=60000, seed=12345,
                  use_patch=not a.get("no_patch", False), safety_mask=True,
                  shield_mode="arc")
net = ActorCritic(env.obs_dim, N_ACTIONS, a.get("hidden", 256))
net.load_state_dict(ck["model"])
net.eval()

obs = env.observe()
for t in range(400):
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(obs))
        logits = apply_mask(logits, torch.from_numpy(env.safe))
        act = logits.argmax(dim=-1).numpy()
    obs, _, _, _, info = env.step(act)
    fin = info["finished"]
    if t < 5 or t % 25 == 0 or fin.any():
        print(f"t={t:>4} 分{env.score.tolist()} 长{env.length.tolist()} "
              f"占据{env.occ.sum(1).tolist()} 食{env.food.tolist()} "
              f"步{env.steps.tolist()} 结束{fin.astype(int).tolist()} "
              f"因{info['cause'].tolist()}")
    if fin.any():
        break
