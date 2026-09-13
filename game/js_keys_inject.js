/*
 * 原始 snake.html 的键盘实测：不依赖任何调试接口，纯读 DOM + 派发真实 KeyboardEvent。
 * 目的：确认今天所有改动都没有影响"人用方向键玩"这条路径。
 */
(function () {
  const res = { steps: [] };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function snap(tag) {
    const overlay = document.querySelector("[class*=overlay], #overlay, .overlay");
    const txt = (document.body.innerText || "").replace(/\s+/g, " ").trim().slice(0, 120);
    const cand = [...document.querySelectorAll("div,span")].filter(
      (e) => /得分|分数|score|最高|最佳/i.test(e.textContent || "") && e.children.length === 0);
    res.steps.push({
      tag: tag,
      overlayShown: overlay ? !/hidden|none/.test(getComputedStyle(overlay).display + overlay.className) : null,
      text: txt,
      scoreEls: cand.slice(0, 3).map((e) => e.textContent.trim().slice(0, 30)),
    });
  }

  function press(key) {
    const ev = { key: key, code: key, bubbles: true, cancelable: true };
    window.dispatchEvent(new KeyboardEvent("keydown", ev));
    document.dispatchEvent(new KeyboardEvent("keydown", ev));
  }

  // 画布上"蛇"的绿色像素数（用离屏采样，不改变原游戏状态）
  function greenPixels() {
    const cv = document.querySelector("canvas");
    if (!cv) return -1;
    try {
      const g = document.createElement("canvas");
      g.width = cv.width; g.height = cv.height;
      const ctx = g.getContext("2d");
      ctx.drawImage(cv, 0, 0);
      const d = ctx.getImageData(0, 0, g.width, g.height).data;
      let n = 0;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i + 1] > 120 && d[i + 1] > d[i] + 30 && d[i + 1] > d[i + 2] + 30) n++;
      }
      return n;
    } catch (e) { return -2; }
  }

  (async function () {
    snap("t0_初始");
    res.green0 = greenPixels();
    press("ArrowRight");                    // 方向键开局
    await sleep(500);
    snap("t1_按右键后");
    res.green1 = greenPixels();
    press("ArrowDown");
    await sleep(500);
    press("ArrowLeft");
    await sleep(500);
    snap("t2_两次转向后");
    res.green2 = greenPixels();
    press(" ");                             // 空格暂停
    await sleep(300);
    snap("t3_按空格后");
    press(" ");                             // 再按恢复
    await sleep(300);
    snap("t4_再按空格后");
    res.green3 = greenPixels();
    res.ok = true;
    document.title = "RESULT " + JSON.stringify(res);
    await fetch("/result", { method: "POST", body: JSON.stringify(res) }).catch(() => {});
  })();
})();
