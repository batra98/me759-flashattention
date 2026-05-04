/**
 * ME759 FlashAttention site: interactive benchmark lab and UI polish
 * Data: T4, d=64, from data/results/*.csv (checked into repo)
 */
(function () {
  'use strict';

  const SEQ_LENS = [512, 1024, 2048, 4096, 8192];

  const MODE_META = {
    naive:         { label: 'Naive',         color: '#f87171', short: '3-kernel, materializes N×N scores in HBM.' },
    naive_causal:  { label: 'Naive causal',  color: '#fb923c', short: 'Same + upper-tri mask before softmax.' },
    flash:         { label: 'Flash v1',      color: '#818cf8', short: 'Tiled FP32, online softmax, O(N) HBM for O.' },
    flash_causal:  { label: 'Flash causal',  color: '#34d399', short: 'Skips future K/V tiles + boundary masks.' },
    flash_v2:      { label: 'Flash v2',      color: '#e879f9', short: 'Warp-centric experiment; wrong B_r cost IO.' },
    flash_wmma:    { label: 'WMMA',          color: '#fbbf24', short: 'FP16 Tensor Core scores, FP32 accum.' },
  };

  /** ms per mode, same order as SEQ_LENS */
  const LATENCY_MS = {
    naive:         [2.0479, 3.3592, 7.8537, 27.1224, 113.5454],
    naive_causal:  [1.7225, 2.3046, 5.4378, 17.9202, 73.6597],
    flash:         [1.5155, 1.7355, 3.7739, 15.0482, 58.6775],
    flash_causal:  [1.3053, 1.8689, 3.2894, 9.5183, 30.6487],
    flash_v2:      [1.9815, 5.6126, 18.6593, 66.3316, 252.8825],
    flash_wmma:    [1.2976, 1.9398, 3.9301, 15.7925, 50.1890],
  };

  const HBM_READ_MB = {
    naive:         [394.27, 1608.13, 6444.25, 25835.52, 103352.32],
    naive_causal:  [244.32, 975.17, 3976.41, 15943.68, 63784.96],
    flash:         [34.60, 136.31, 541.05, 2211.84, 8816.64],
    flash_causal:  [18.87, 71.30, 276.81, 1116.16, 4423.68],
    flash_v2:      [68.12, 270.13, 1105.92, 4403.20, 17582.08],
    flash_wmma:    [8.52, 33.82, 134.74, 537.92, 2201.60],
  };

  const HBM_WRITE_MB = {
    naive:         [17.96, 71.56, 285.73, 1163.84, 4663.50],
    naive_causal:  [17.97, 71.59, 285.78, 1163.95, 4663.72],
    flash:         [1.05, 2.10, 4.19, 8.39, 16.78],
    flash_causal:  [1.05, 2.10, 4.19, 8.39, 16.78],
    flash_v2:      [1.05, 2.10, 4.19, 8.39, 16.78],
    flash_wmma:    [1.05, 2.10, 4.19, 8.39, 16.78],
  };

  let chartInstance = null;
  let selectedN = 8192;
  const defaultVisible = new Set(['naive', 'flash', 'flash_causal', 'flash_wmma']);

  function idxN(n) {
    return SEQ_LENS.indexOf(n);
  }

  function formatNum(x, d) {
    if (x >= 1000) return x.toFixed(0);
    if (x >= 100) return x.toFixed(1);
    if (x >= 10) return x.toFixed(2);
    return x.toFixed(3);
  }

  function initScrollProgress() {
    const bar = document.getElementById('scroll-progress');
    if (!bar) return;
    const onScroll = () => {
      const st = document.documentElement.scrollTop || document.body.scrollTop;
      const sh = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const p = sh > 0 ? (st / sh) * 100 : 0;
      bar.style.width = Math.min(100, p) + '%';
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  function initNavActive() {
    const nav = document.getElementById('nav');
    if (!nav) return;
    const links = [...nav.querySelectorAll('a[href^="#"]')];
    const sections = links
      .map((a) => {
        const id = a.getAttribute('href').slice(1);
        return { id, el: document.getElementById(id) };
      })
      .filter((s) => s.el);

    const onScroll = () => {
      const y = window.scrollY + 120;
      let current = sections[0]?.id;
      for (const s of sections) {
        const top = s.el.offsetTop;
        if (top <= y) current = s.id;
      }
      links.forEach((a) => a.classList.toggle('nav-active', a.getAttribute('href') === '#' + current));
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  function updateMetricCards() {
    const i = idxN(selectedN);
    if (i < 0) return;

    const naiveMs = LATENCY_MS.naive[i];
    const row = (mode) => {
      const ms = LATENCY_MS[mode][i];
      const sp = naiveMs / ms;
      return {
        ms,
        read: HBM_READ_MB[mode][i],
        write: HBM_WRITE_MB[mode][i],
        speedup: sp,
      };
    };

    document.querySelectorAll('[data-metric-mode]').forEach((el) => {
      const mode = el.getAttribute('data-metric-mode');
      if (!LATENCY_MS[mode]) return;
      const m = row(mode);
      el.querySelector('[data-field="lat"]').textContent = formatNum(m.ms, 3) + ' ms';
      el.querySelector('[data-field="read"]').textContent = formatNum(m.read, 2) + ' MB';
      el.querySelector('[data-field="write"]').textContent = formatNum(m.write, 2) + ' MB';
      const su = el.querySelector('[data-field="speedup"]');
      if (su) su.textContent = m.speedup >= 1 ? '×' + formatNum(m.speedup, 2) : 'n/a';
    });

    const heroStat = document.getElementById('hero-speedup');
    const heroN = document.getElementById('hero-n-label');
    if (heroStat) {
      const fc = row('flash_causal');
      heroStat.textContent = '×' + formatNum(fc.speedup, 2);
    }
    if (heroN) heroN.textContent = 'N=' + selectedN;
  }

  function buildChart() {
    const canvas = document.getElementById('latencyChart');
    if (!canvas || typeof Chart === 'undefined') return;

    const visible = [...document.querySelectorAll('.kernel-chip input[type="checkbox"]:checked')].map((c) => c.value);

    const datasets = visible.map((mode) => ({
      label: MODE_META[mode].label,
      data: SEQ_LENS.map((_, j) => LATENCY_MS[mode][j]),
      borderColor: MODE_META[mode].color,
      backgroundColor: MODE_META[mode].color + '33',
      tension: 0.22,
      fill: false,
      pointRadius: 4,
      pointHoverRadius: 7,
      borderWidth: 2,
    }));

    if (chartInstance) chartInstance.destroy();

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    chartInstance = new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: SEQ_LENS.map((n) => 'N=' + n),
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: reduced ? false : { duration: 600 },
        interaction: { intersect: false, mode: 'index' },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: '#a1a7c3',
              font: { family: "'Inter', sans-serif", size: 11 },
              padding: 14,
              usePointStyle: true,
            },
          },
          tooltip: {
            backgroundColor: 'rgba(14,16,23,0.95)',
            titleColor: '#fff',
            bodyColor: '#d1d5db',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            padding: 12,
            callbacks: {
              label(ctx) {
                return ctx.dataset.label + ': ' + formatNum(ctx.parsed.y, 3) + ' ms';
              },
            },
          },
          title: {
            display: true,
            text: 'Wall-clock latency (lower is better)',
            color: '#7b8099',
            font: { size: 12, weight: '500' },
          },
        },
        scales: {
          x: {
            grid: { color: 'rgba(255,255,255,0.06)' },
            ticks: { color: '#7b8099', font: { size: 11 } },
          },
          y: {
            type: 'logarithmic',
            min: 0.8,
            grid: { color: 'rgba(255,255,255,0.06)' },
            ticks: {
              color: '#7b8099',
              callback(v) {
                return v + ' ms';
              },
            },
            title: {
              display: true,
              text: 'ms (log scale)',
              color: '#5b6078',
              font: { size: 11 },
            },
          },
        },
      },
    });

    const hint = document.getElementById('labSelectedN');
    if (hint) hint.textContent = 'Metrics below use N = ' + selectedN + ' · d = 64 · Tesla T4';
  }

  function initNpills() {
    document.querySelectorAll('.n-pill').forEach((btn) => {
      btn.addEventListener('click', () => {
        selectedN = parseInt(btn.getAttribute('data-n'), 10);
        document.querySelectorAll('.n-pill').forEach((b) => b.classList.toggle('active', parseInt(b.getAttribute('data-n'), 10) === selectedN));
        updateMetricCards();
        buildChart();
      });
    });
    document.querySelectorAll('.n-pill').forEach((b) => {
      if (parseInt(b.getAttribute('data-n'), 10) === selectedN) b.classList.add('active');
    });
  }

  function syncKernelChipStyles() {
    document.querySelectorAll('.kernel-chip input').forEach((input) => {
      input.closest('.kernel-chip').classList.toggle('is-on', input.checked);
    });
  }

  function initKernelChips() {
    document.querySelectorAll('.kernel-chip input').forEach((input) => {
      input.checked = defaultVisible.has(input.value);
      input.addEventListener('change', () => {
        const checked = [...document.querySelectorAll('.kernel-chip input:checked')];
        if (checked.length === 0) {
          input.checked = true;
        }
        syncKernelChipStyles();
        buildChart();
      });
    });
    syncKernelChipStyles();
  }

  function initKernelCards() {
    document.querySelectorAll('.kernel-card').forEach((card) => {
      const btn = card.querySelector('.kernel-card-toggle');
      if (!btn) return;
      btn.addEventListener('click', () => {
        const open = card.classList.toggle('is-open');
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      });
    });
  }

  function initReducedMotion() {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const apply = () => {
      document.documentElement.classList.toggle('reduce-motion', mq.matches);
      if (mq.matches && window.tileReset) window.tileReset();
    };
    mq.addEventListener('change', apply);
    apply();
  }

  document.addEventListener('DOMContentLoaded', () => {
    initScrollProgress();
    initNavActive();
    initNpills();
    initKernelChips();
    initKernelCards();
    initReducedMotion();
    updateMetricCards();
    buildChart();
  });
})();
