"""
截图专用的慢速本地服务：/slow.png 故意延迟 DELAY 秒才返回。

用途：把页面的 load 事件压住几秒，让游戏在**真实主循环**里先跑出一段中局，
Firefox --screenshot 就抓得到"蛇很长、遮罩已消失"的真实状态，
而不是绕过状态机（用 debug API 的 apply()）留下的陈旧"准备开始"。

用法： .venv/bin/python game/slow_server.py [端口] [延迟秒]
"""

import http.server
import socketserver
import sys
import time

ROOT = "/home/lzl/snake-rl/game"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8740
DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_GET(self):
        if self.path.startswith("/slow.png"):
            time.sleep(DELAY)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        return super().do_GET()

    def log_message(self, *a):
        pass


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as srv:
    print(f"slow server on {PORT}（/slow.png 延迟 {DELAY}s）", flush=True)
    srv.serve_forever()
