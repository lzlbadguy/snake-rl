/*
 * 注入到游戏里的对拍脚本（自动化测试用，不参与正常游玩）。
 *
 * 做法：
 *   1. 把 Math.random 换成确定性的 32 位 LCG —— 食物流变得可复现
 *   2. 用另一条 LCG 生成"共享动作序列"（不是 AI 的动作！）驱动游戏，
 *      这样 JS 与 Python 会走到**完全相同**的状态序列上
 *   3. 每一步记录：观测(170维)、JS 策略的贪心动作、当时的游戏状态
 *   4. 全部 POST 回本地 node 服务，交给 Python 侧逐项比对
 *
 * 关键点：两个实现的占位必须一致 —— 食物流都按"重开 1 次 + 每次吃到 1 次"消耗随机数。
 */
<script>
(function () {
  function makeLCG(seed) {
    let s = seed >>> 0;
    return function () {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  Math.random = makeLCG(12345);      // 食物流（游戏只用 Math.random 放食物）
  const actRnd = makeLCG(999);       // 动作流

  const dbg = window.__snakeDebug;
  const STEPS = 400;

  dbg.sync = true;                   // 主循环不再自己推进，由本脚本逐步驱动
  dbg.setAI(false);
  dbg.setShield(false);              // 关掉安全层，先验证"裸策略"的前向一致性
  dbg.reset();

  const trace = [];
  let deaths = 0;
  for (let t = 0; t < STEPS; t++) {
    const st = dbg.state();
    if (st.over) {                   // 上一轮死了：JS 停在终局画面，重开后继续
      trace.push({dead: true, score: st.score});
      deaths++;
      dbg.reset();
      continue;
    }
    const a = Math.floor(actRnd() * 3);
    const obs = dbg.obs();
    const aiAct = dbg.aiAction();
    const safe = dbg.safety();       // 安全层判据（与 Python 的 safety_flags() 比对）
    trace.push({t: t, act: a, aiAct: aiAct, obs: obs, st: st, safe: safe});
    dbg.apply(a);
  }

  const payload = JSON.stringify({layers: dbg.layers(), deaths: deaths, steps: STEPS, trace: trace});
  fetch('/result', {method: 'POST', body: payload})
    .then(function () { document.title = 'PARITY-DONE'; })
    .catch(function (e) {
      var x = new XMLHttpRequest();
      x.open('POST', '/result'); x.send(payload);
      document.title = 'PARITY-DONE';
    });
})();
</script>
