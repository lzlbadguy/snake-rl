<script>
/*
 * 截图用（README 展示图）。
 * 背景：用 debug API 的 apply() 驱动可以稳定养出"中局长蛇"，但它绕过状态机，
 *      于是遮罩会残留成"准备开始"（真实对局中此刻它是隐藏的）。所以这里：
 *   1) sync=true（主循环只重绘不推进，画面可确定地冻结）
 *   2) 反复重开直到 AI 打到 TARGET 分且**仍活着**，避免截到刚死/正在重开那一瞬
 *   3) 用 !important 隐藏残留遮罩（draw() 每帧会按 state 重新显示它，普通 class 覆盖不掉）
 */
(function () {
  const dbg = window.__snakeDebug;
  if (!dbg) return;
  dbg.sync = true;
  const btn = document.getElementById("aiBtn");
  if (btn && btn.textContent.indexOf("停止") < 0) btn.click();
  const TARGET = window.__SHOT_TARGET__ || 80;
  let st = dbg.state();
  for (let attempt = 0; attempt < 40; attempt++) {
    dbg.reset();
    for (let t = 0; t < 4000; t++) {
      dbg.apply(dbg.aiAction());
      st = dbg.state();
      if (st.over || st.score >= TARGET) break;
    }
    if (!st.over && st.score >= TARGET) break;
  }
  const ov = document.querySelector(".overlay");
  if (ov) ov.style.setProperty("display", "none", "important");
  document.title = "SHOT score=" + st.score + " len=" + st.snake.length + " over=" + st.over;
})();
</script>
