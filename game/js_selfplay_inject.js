/*
 * 自玩验证 v2：在真实游戏里对比"开/关安全层"的分数，并验证 UI 链路。
 *   阶段 1a：安全层开，同步驱动 8000 步（绕过 rAF），统计每局分数
 *   阶段 1b：安全层关，同样 8000 步 —— 这就是部署时的收益对比
 *   阶段 2 ：真实主循环（点按钮 + 16× + 自动重开），读界面统计
 *
 * 注意：这个游戏没有回合上限，所以分数直接反映"活到死之前吃了多少"，
 * 比 Python 侧带 1600/4000 步上限的评估更能代表真实部署表现。
 */
<script>
(function () {
  const dbg = window.__snakeDebug;
  dbg.sync = true;
  dbg.setAI(false);

  function selfPlay(steps, shield) {
    dbg.setShield(shield);
    dbg.reset();
    const scores = [], eps = [];
    let cur = 0;
    for (let t = 0; t < steps; t++) {
      const st = dbg.state();
      if (st.over) { scores.push(st.score); eps.push(cur); cur = 0; dbg.reset(); continue; }
      cur++;
      dbg.apply(dbg.aiAction());
    }
    const n = scores.length;
    const sum = scores.reduce(function (a, b) { return a + b; }, 0);
    const lsum = eps.reduce(function (a, b) { return a + b; }, 0);
    return {
      steps: steps, episodes: n,
      mean: n ? +(sum / n).toFixed(2) : 0,
      max: n ? Math.max.apply(null, scores) : 0,
      min: n ? Math.min.apply(null, scores) : 0,
      meanStepsPerEp: n ? Math.round(lsum / n) : 0,
      scores: scores
    };
  }

  const STEPS = 24000;      // 单阶段步数：安全层开着时每局 3000+ 步，够跑 6~8 局
  const withShield = selfPlay(STEPS, true);
  const noShield = selfPlay(STEPS, false);

  // 阶段 2：真实主循环
  dbg.setShield(true);
  dbg.sync = false;
  const spd = document.getElementById('aiSpeed');
  spd.value = '16';
  spd.dispatchEvent(new Event('change'));
  document.getElementById('aiBtn').click();

  setTimeout(function () {
    const payload = {
      withShield: withShield,
      noShield: noShield,
      phase2: {
        speed: spd.value,
        aiLabel: document.getElementById('aiBtn').textContent,
        shieldChecked: document.getElementById('shieldChk').checked,
        stats: document.getElementById('aiStats').textContent,
        score: document.getElementById('score').textContent,
        best: document.getElementById('best').textContent
      }
    };
    const body = JSON.stringify(payload);
    fetch('/result', {method: 'POST', body: body})
      .then(function () { document.title = 'SELFPLAY-DONE'; })
      .catch(function () {
        const x = new XMLHttpRequest(); x.open('POST', '/result'); x.send(body);
        document.title = 'SELFPLAY-DONE';
      });
  }, 15000);
})();
</script>
