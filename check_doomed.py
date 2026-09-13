"""
诊断：当前策略"什么时候"进入注定死亡状态（可达格数 < 身体长度）。

这决定了"提前惩罚"这个奖励改动值不值得做：
  * 若多数回合在死前几十步就进入注定态 → 提前惩罚能把 -1 的信用分配窗口大幅缩短，很值
  * 若只在死前 1~2 步才进入 → 等于没提前，白改

用法： .venv/bin/python check_doomed.py [局数]
"""

import sys
import numpy as np
import torch

from snake_env import VectorSnake, N_ACTIONS, CAUSE_WALL, CAUSE_SELF
from train import ActorCritic


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ck = torch.load("runs/ppo20/ckpt.pt", map_location="cpu", weights_only=True)
    args = ck.get("args", {})
    n_env = 64

    env = VectorSnake(n_envs=n_env, seed=123, use_patch=not args.get("no_patch", False),
                      food_reward=0.0, death_penalty=0.0, step_penalty=0.0,
                      shaping_scale=0.0)
    model = ActorCritic(env.obs_dim, N_ACTIONS, args.get("hidden", 256))
    model.load_state_dict(ck["model"])
    model.eval()

    obs = env.observe()
    cur = np.zeros(n_env, dtype=bool)          # 本回合是否已进入过注定态
    first = np.full(n_env, -1, dtype=np.int64)
    tstep = np.zeros(n_env, dtype=np.int64)

    done = 0
    n_doomed = 0
    lead = []            # 首次注定态 -> 死亡的步数
    scores, self_deaths_doomed, self_deaths = [], 0, 0

    while done < n_target:
        with torch.no_grad():
            act = model(torch.from_numpy(obs))[0].argmax(-1).numpy()
        obs, _, _, _, info = env.step(act)
        tstep += 1
        dm = info["doomed"]
        newly = dm & ~cur
        first[newly] = tstep[newly]
        cur |= dm

        fin = info["finished"]
        for i in np.flatnonzero(fin):
            done += 1
            scores.append(int(info["final_score"][i]))
            cause = int(info["cause"][i])
            if cause == CAUSE_SELF:
                self_deaths += 1
                if first[i] >= 0:
                    self_deaths_doomed += 1
            if first[i] >= 0:
                n_doomed += 1
                lead.append(int(tstep[i] - first[i]))
            cur[i] = False
            first[i] = -1
            tstep[i] = 0

    s = np.array(scores)
    lead = np.array(lead)
    print(f"政策: ckpt update={ck.get('update')}  评估 {done} 局  均分 {s.mean():.1f} 中位 {np.median(s):.0f}")
    print(f"进入过『注定死亡』态的回合: {n_doomed}/{done} = {100*n_doomed/done:.1f}%")
    if len(lead):
        print(f"提前量（首次注定态 → 死亡）: 均值 {lead.mean():.1f} 步  中位 {np.median(lead):.0f} 步")
        buckets = [(0, 2), (3, 10), (11, 25), (26, 50), (51, 10**9)]
        for lo, hi in buckets:
            m = (lead >= lo) & (lead <= hi)
            label = f"{lo}-{hi} 步" if hi < 10**9 else f">{lo} 步"
            print(f"   {label:<12} {100*m.mean():5.1f}%")
    print(f"自撞死亡中，事前进入过注定态的占比: {self_deaths_doomed}/{self_deaths} = "
          f"{100*self_deaths_doomed/max(1,self_deaths):.1f}%")


if __name__ == "__main__":
    main()
