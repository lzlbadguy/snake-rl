<script>
/*
 * 验证"直接打开 snake_template.html"这条路径：
 *   ① 脚本没崩（window.__snakeDebug 存在 —— 它在脚本末尾才挂上，崩了就没有）
 *   ② 手动玩可用（state() 能读、能 start）
 *   ③ 界面给出了明确提示（提示行有 ⚠ 与构建命令；AI 按钮变灰并改名）
 */
setTimeout(() => {
  const out = { hasDebug: typeof window.__snakeDebug, ok: false, err: null, hint: "", aiLabel: "" };
  try {
    const dbg = window.__snakeDebug;
    if (dbg) {
      dbg.start();
      for (let i = 0; i < 30; i++) dbg.apply(1);   // 直行 30 步：证明手动逻辑可用
      const st = dbg.state();
      out.ok = !!(st && typeof st.score === "number");
      out.score = st && st.score;
      out.len = st && st.snake && st.snake.length;
    }
    out.hint = (document.body.innerText.match(/⚠[^\n]*/) || [""])[0];
    out.bodyHead = document.body.innerText.replace(/\s+/g, " ").slice(0, 260);
    const b = document.getElementById("aiBtn");
    out.aiLabel = b ? b.textContent.trim() : "";
    out.aiOpacity = b ? b.style.opacity : "";
  } catch (e) { out.err = String(e); }
  fetch("/result", { method: "POST", body: JSON.stringify(out) });
}, 700);
</script>
