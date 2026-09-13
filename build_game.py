"""
把导出好的策略权重内联进游戏模板，生成自包含的 snake.html。

用法： .venv/bin/python build_game.py
       （默认读 game/snake_template.html + runs/ppo20/policy.json → 写 /home/lzl/snake.html）
"""

import argparse
import json
import os
import shutil
import sys

PLACEHOLDER = "/*__POLICY_DATA__*/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="game/snake_template.html")
    ap.add_argument("--policy", default="runs/ppo20/policy.json")
    ap.add_argument("--out", default="/home/lzl/snake.html")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    tpl = open(args.template, encoding="utf-8").read()
    pol_txt = open(args.policy, encoding="utf-8").read()
    pol = json.loads(pol_txt)                      # 顺带校验 JSON 合法
    if PLACEHOLDER not in tpl:
        sys.exit(f"模板里找不到占位符 {PLACEHOLDER}")
    html = tpl.replace(PLACEHOLDER, pol_txt)

    if not args.no_backup and os.path.exists(args.out):
        bak = args.out + ".bak"
        if not os.path.exists(bak):                # 只在第一次留备份，别覆盖真正的原版
            shutil.copy2(args.out, bak)
            print(f"已备份原游戏到 {bak}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成 {args.out}  {os.path.getsize(args.out)/1024:.0f} KB")
    print(f"  策略 {args.policy}  ckpt_update={pol['meta']['chkpt_update']}  obs_dim={pol['obs_dim']}")
    for L in pol["layers"]:
        print(f"    {L['name']:<9} {L['in']:>4} -> {L['out']:<4} act={L['act']}")


if __name__ == "__main__":
    main()
