/* ─── Leaderboard Data ──────────────────────────────────────── */
const DATA = {
  image: {
    tnm: {
      cols: ['Model', 'Acc ZH (%)', 'Acc EN (%)', 'RQ ZH', 'RQ EN'],
      higher: [true, true, true, true],
      rows: [
        { name: 'Qwen3.5-397B',           vals: [61.46, 58.94, 87.43, 87.35], lcagent: false },
        { name: 'Qwen3.5-397B + LCAgent', vals: [66.30, 69.21, 91.58, 90.96], lcagent: true  },
        { name: 'Kimi-K2.5',              vals: [48.96, 46.88, 83.61, 83.54], lcagent: false },
        { name: 'Kimi-K2.5 + LCAgent',   vals: [67.71, 50.00, 91.39, 87.29], lcagent: true  },
        { name: 'GLM-4.6V',               vals: [38.54, 34.37, 78.06, 77.64], lcagent: false },
        { name: 'HuatuoGPT-Vision',       vals: [7.29,  11.46, 44.93, 56.32], lcagent: false },
        { name: 'DeepMedix-R1',           vals: [0.00,   1.04, 26.32, 26.94], lcagent: false },
        { name: 'Grok 3*',               vals: [47.49, 11.51, 58.26, 64.56], lcagent: false },
        { name: 'Claude Sonnet 4.6',      vals: [25.00, 28.13, 78.19, 80.69], lcagent: false },
        { name: 'GPT-5.2',               vals: [36.46, 35.41, 81.04, 81.25], lcagent: false },
        { name: 'GPT-5.2 + LCAgent',     vals: [47.92, 50.00, 89.24, 87.08], lcagent: true  },
        { name: 'Llama-4-Maverick',       vals: [21.10, 17.44, 64.73, 70.51], lcagent: false },
      ]
    },
    treat: {
      cols: ['Model', 'Precision ZH (%)', 'Precision EN (%)', 'BERT-F1 ZH', 'BERT-F1 EN'],
      higher: [true, true, true, true],
      rows: [
        { name: 'Qwen3.5-397B',           vals: [35.22, 31.29, 41.25, 39.37], lcagent: false },
        { name: 'Qwen3.5-397B + LCAgent', vals: [59.29, 47.54, 55.63, 12.90], lcagent: true  },
        { name: 'Kimi-K2.5',              vals: [38.61, 25.61, 32.43, 34.38], lcagent: false },
        { name: 'GLM-4.6V',               vals: [44.70, 33.85, 39.37, 40.62], lcagent: false },
        { name: 'Grok 3*',               vals: [65.48, 10.84, 54.35, 40.00], lcagent: false },
        { name: 'Claude Sonnet 4.6',      vals: [25.39, 22.99, 38.13, 30.00], lcagent: false },
        { name: 'GPT-5.2',               vals: [33.31, 24.25, 35.00, 37.50], lcagent: false },
        { name: 'GPT-5.2 + LCAgent',     vals: [53.50, 48.14, 55.63, 29.38], lcagent: true  },
        { name: 'Llama-4-Maverick',       vals: [20.89, 17.43, 34.38, 38.75], lcagent: false },
      ]
    },
    e2e: {
      cols: ['Model', 'Precision ZH (%)', 'Precision EN (%)', 'BERT-F1 ZH', 'BERT-F1 EN'],
      higher: [true, true, true, true],
      rows: [
        { name: 'Qwen3.5-397B',           vals: [31.60, 29.84, 34.59, 33.54], lcagent: false },
        { name: 'Qwen3.5-397B + LCAgent', vals: [61.98, 49.51, 55.00, 14.38], lcagent: true  },
        { name: 'Kimi-K2.5',              vals: [36.80, 30.34, 41.04, 28.04], lcagent: false },
        { name: 'Kimi-K2.5 + LCAgent',   vals: [54.55, 38.19, 57.50, 33.12], lcagent: true  },
        { name: 'GLM-4.6V',               vals: [51.78, 32.66, 51.88, 30.42], lcagent: false },
        { name: 'Grok 3*',               vals: [47.75, 53.07, 51.25, 35.02], lcagent: false },
        { name: 'Claude Sonnet 4.6',      vals: [32.08, 27.62, 37.39, 25.41], lcagent: false },
        { name: 'GPT-5.2',               vals: [36.00, 31.25, 35.63, 35.83], lcagent: false },
        { name: 'GPT-5.2 + LCAgent',     vals: [56.57, 42.14, 49.38, 23.50], lcagent: true  },
        { name: 'Llama-4-Maverick',       vals: [40.34, 32.94, 37.13, 39.52], lcagent: false },
      ]
    }
  },
  text: {
    tnm: {
      cols: ['Model', 'Acc ZH (%)', 'Acc EN (%)', 'RQ ZH', 'RQ EN'],
      higher: [true, true, true, true],
      rows: [
        { name: 'Qwen3.5-397B',           vals: [59.37, 36.46, 84.10, 75.90], lcagent: false },
        { name: 'Qwen3.5-397B + LCAgent', vals: [74.65, 42.41, 90.89, 79.63], lcagent: true  },
        { name: 'Kimi-K2.5',              vals: [55.21, 41.67, 82.99, 79.51], lcagent: false },
        { name: 'Kimi-K2.5 + LCAgent',   vals: [55.21, 41.67, 88.76, 82.50], lcagent: true  },
        { name: 'GLM-4.6V',               vals: [38.54, 51.08, 79.03, 63.06], lcagent: false },
        { name: 'HuatuoGPT-Vision',       vals: [4.17,  11.21, 42.15, 51.69], lcagent: false },
        { name: 'DeepMedix-R1',           vals: [0.00,   0.71, 25.24, 23.02], lcagent: false },
        { name: 'Grok 3*',               vals: [47.49, 12.35, 79.91, 75.32], lcagent: false },
        { name: 'Claude Sonnet 4.5',      vals: [28.40, 28.13, 81.27, 75.28], lcagent: false },
        { name: 'GPT-5.2',               vals: [38.54, 31.25, 79.30, 73.68], lcagent: false },
        { name: 'GPT-5.2 + LCAgent',     vals: [41.97, 32.29, 86.23, 79.03], lcagent: true  },
        { name: 'Llama-4-Maverick',       vals: [22.92, 10.48, 77.15, 59.35], lcagent: false },
      ]
    },
    treat: {
      cols: ['Model', 'Precision ZH (%)', 'Precision EN (%)', 'BERT-F1 ZH', 'BERT-F1 EN'],
      higher: [true, true, true, true],
      rows: [
        { name: 'Qwen3.5-397B',           vals: [25.44, 25.96, 37.29, 37.11], lcagent: false },
        { name: 'Qwen3.5-397B + LCAgent', vals: [64.26, 41.20, 69.05, 14.95], lcagent: true  },
        { name: 'Kimi-K2.5',              vals: [30.66, 30.86, 32.43, 31.64], lcagent: false },
        { name: 'Kimi-K2.5 + LCAgent',   vals: [59.54, 33.46, 63.16, 26.97], lcagent: true  },
        { name: 'GLM-4.6V',               vals: [34.68, 36.44, 31.37, 39.04], lcagent: false },
        { name: 'Grok 3*',               vals: [32.81, 26.43, 35.43, 37.06], lcagent: false },
        { name: 'Claude Sonnet 4.5',      vals: [29.24, 30.05, 30.10, 28.05], lcagent: false },
        { name: 'GPT-5.2',               vals: [19.94, 25.93, 33.27, 36.90], lcagent: false },
        { name: 'GPT-5.2 + LCAgent',     vals: [55.36, 43.50, 60.43, 18.94], lcagent: true  },
        { name: 'Llama-4-Maverick',       vals: [23.83, 11.80, 33.91, 36.84], lcagent: false },
      ]
    },
    e2e: {
      cols: ['Model', 'Precision ZH (%)', 'Precision EN (%)', 'BERT-F1 ZH', 'BERT-F1 EN'],
      higher: [true, true, true, true],
      rows: [
        { name: 'Qwen3.5-397B',           vals: [23.54, 17.10, 32.71, 22.92], lcagent: false },
        { name: 'Qwen3.5-397B + LCAgent', vals: [66.45, 47.41, 61.25, 10.63], lcagent: true  },
        { name: 'Kimi-K2.5',              vals: [34.56, 26.61, 39.38, 29.08], lcagent: false },
        { name: 'Kimi-K2.5 + LCAgent',   vals: [56.26, 42.41, 57.50, 25.00], lcagent: true  },
        { name: 'GLM-4.6V',               vals: [27.23, 33.74, 36.88, 36.25], lcagent: false },
        { name: 'Grok 3*',               vals: [40.59, 37.41, 42.10, 35.51], lcagent: false },
        { name: 'Claude Sonnet 4.5',      vals: [34.63, 32.60, 36.25, 29.59], lcagent: false },
        { name: 'GPT-5.2',               vals: [35.13, 29.90, 40.00, 35.42], lcagent: false },
        { name: 'GPT-5.2 + LCAgent',     vals: [55.62, 34.17, 62.29, 16.46], lcagent: true  },
        { name: 'Llama-4-Maverick',       vals: [31.00, 32.93, 38.96, 37.92], lcagent: false },
      ]
    }
  }
};

/* ─── State ─────────────────────────────────────────────────── */
let currentInput = 'image';
let currentTask  = 'tnm';
let sortCol      = 1;
let sortAsc      = false;

/* ─── Render ─────────────────────────────────────────────────── */
function render() {
  const d    = DATA[currentInput][currentTask];
  const head = document.getElementById('lb-head');
  const body = document.getElementById('lb-body');

  // Best values among base (non-LCAgent) rows per metric column
  const baseRows = d.rows.filter(r => !r.lcagent);
  const best = d.cols.slice(1).map((_, ci) => {
    const vals = baseRows.map(r => r.vals[ci]).filter(v => v != null);
    return d.higher[ci] ? Math.max(...vals) : Math.min(...vals);
  });

  // Sort all rows by selected column
  const sorted = [...d.rows].sort((a, b) => {
    const va = a.vals[sortCol - 1] ?? -Infinity;
    const vb = b.vals[sortCol - 1] ?? -Infinity;
    return sortAsc ? va - vb : vb - va;
  });

  // Header
  head.innerHTML = '<tr>' + d.cols.map((c, i) => {
    const activeClass = i === sortCol ? ' sorted' : '';
    const arrow       = i === sortCol ? (sortAsc ? ' ▲' : ' ▼') : ' ⇅';
    const icon        = i > 0 ? `<span class="sort-icon">${arrow}</span>` : '';
    return `<th class="${activeClass}" onclick="sortBy(${i})">${c}${icon}</th>`;
  }).join('') + '</tr>';

  // Body
  let rank = 0;
  let html = '';
  sorted.forEach(row => {
    if (!row.lcagent) rank++;
    const rankClass  = !row.lcagent && rank <= 3 ? `rank-${rank}` : '';
    const agentClass = row.lcagent ? 'lcagent-row' : '';
    const cells = row.vals.map((v, ci) => {
      if (v == null) return '<td>—</td>';
      const isBest = !row.lcagent && d.higher[ci] && v === best[ci];
      return `<td${isBest ? ' class="best-val"' : ''}>${v.toFixed(2)}</td>`;
    });
    html += `<tr class="${rankClass} ${agentClass}"><td>${row.name}</td>${cells.join('')}</tr>`;
  });
  body.innerHTML = html;
}

/* ─── Public interaction handlers (called from HTML) ────────── */
function switchInput(type) {
  currentInput = type;
  sortCol = 1;
  sortAsc = false;
  document.querySelectorAll('.lb-tab').forEach((t, i) => {
    t.classList.toggle('active', (i === 0 && type === 'image') || (i === 1 && type === 'text'));
  });
  render();
}

function switchTask(task) {
  currentTask = task;
  sortCol = 1;
  sortAsc = false;
  document.querySelectorAll('.task-tab').forEach(t => {
    const label = t.textContent.toLowerCase();
    t.classList.toggle('active',
      (task === 'tnm'   && label.includes('tnm'))        ||
      (task === 'treat' && label.includes('treatment'))  ||
      (task === 'e2e'   && label.includes('end'))
    );
  });
  render();
}

function sortBy(col) {
  if (col === 0) return;
  sortAsc = sortCol === col ? !sortAsc : false;
  sortCol = col;
  render();
}

/* ─── Init ──────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', render);
