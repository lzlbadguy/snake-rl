"""
生成对拍用的测试页：把 game/js_test_inject.js 注入到 AI 版游戏里。

用法： .venv/bin/python game/make_test_page.py [源文件] [输出]
      默认 /home/lzl/snake_ai.html -> game/_js_test.html
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/home/lzl/snake_ai.html"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "_js_test.html")
    inj = sys.argv[3] if len(sys.argv) > 3 else "js_test_inject.js"
    inject = open(os.path.join(HERE, inj), encoding="utf-8").read()
    html = open(src, encoding="utf-8").read()
    if "</body>" not in html:
        sys.exit("源文件里没有 </body>")
    html = html.replace("</body>", inject + "\n</body>", 1)
    # 测试页用相对路径的 fetch，由本地服务提供
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"测试页已生成: {out}  ({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
