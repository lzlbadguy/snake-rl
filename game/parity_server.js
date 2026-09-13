/*
 * 对拍用的本地服务：把游戏页面发给浏览器，并接住浏览器 POST 回来的轨迹。
 * 用法： node game/parity_server.js [port]
 */
const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const PORT = Number(process.argv[2] || 8732);
const OUT = path.join(ROOT, 'js_trace.json');
fs.writeFileSync(OUT, '');

http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/result') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      fs.writeFileSync(OUT, body);
      res.writeHead(200, {'Access-Control-Allow-Origin': '*'});
      res.end('ok');
      console.log('TRACE SAVED: ' + body.length + ' bytes -> ' + OUT);
    });
    return;
  }
  const rel = req.url === '/' ? '_js_test.html' : decodeURIComponent(req.url.split('?')[0]);
  const file = path.join(ROOT, rel);
  if (!file.startsWith(ROOT)) { res.writeHead(403); res.end('no'); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, {'Content-Type': file.endsWith('.html')
      ? 'text/html; charset=utf-8' : 'application/octet-stream'});
    res.end(data);
  });
}).listen(PORT, '127.0.0.1', () => console.log('parity server on ' + PORT));
