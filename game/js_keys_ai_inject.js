/*
 * 键盘路径实测（跑在 snake_ai.html 上，用 __snakeDebug 读状态）。
 * 派发真实 KeyboardEvent 到 window/document，验证：
 *   ① 方向键能开局 ② 转向生效 ③ 空格能暂停/恢复 ④ AI 开关仍正常
 * 原始 snake.html 与 AI 版共用同一套按键处理代码。
 */
(function () {
  const dbg = window.__snakeDebug;
  const res = { steps: [] };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const snap = (tag) => res.steps.push({ tag: tag, st: dbg.state() });

  function press(key) {
    const ev = { key: key, code: key, bubbles: true, cancelable: true };
    window.dispatchEvent(new KeyboardEvent("keydown", ev));
    document.dispatchEvent(new KeyboardEvent("keydown", ev));
  }

  (async function () {
    dbg.setAI(false);
    dbg.setShield(true);
    dbg.reset();
    await sleep(50);
    snap("t0_初始");

    press("ArrowRight");
    await sleep(700);
    snap("t1_按右键(应开局并移动)");

    press("ArrowDown");
    await sleep(700);
    snap("t2_按下键转向");

    press("ArrowLeft");
    await sleep(700);
    snap("t3_按左键转向");

    press(" ");
    await sleep(400);
    snap("t4_空格(应暂停)");

    press(" ");
    await sleep(400);
    snap("t5_再空格(应恢复)");

    press("r");
    await sleep(400);
    snap("t6_按r(应重开)");

    dbg.setAI(true);
    await sleep(1500);
    res.aiState = dbg.state();
    res.aiOn = true;
    dbg.setAI(false);

    res.ok = true;
    document.title = "DONE";
    try {
      await fetch("/result", { method: "POST", body: JSON.stringify(res) });
    } catch (e) {
      res.err = String(e);
    }
  })();
})();
