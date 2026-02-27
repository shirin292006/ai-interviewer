/**
 * =============================================
 * Main Application Script
 * =============================================
 */

'use strict';

// ── Config ─────────────────────────────────────
const API = 'http://localhost:8000';

// ── State ───────────────────────────────────────
const state = {
  currentQuestion: '',
  uploading: false,
  generating: false,
  evaluating: false,
  running: false,
};

// ── DOM Helper ──────────────────────────────────
const $ = id => document.getElementById(id);

// ══════════════════════════════════════════════
// UPLOAD RESUME
// ══════════════════════════════════════════════
async function uploadResume() {
  if (state.uploading) return;

  const fileInput = $('resume-file');
  const statusEl = $('upload-status');
  const btn = $('upload-btn');
  const zone = $('upload-zone');

  // Check for drag-dropped file first, then input file
  const file = zone._droppedFile || fileInput.files[0];

  if (!file) {
    setStatus(statusEl, 'error', '⚠ Please select or drop a PDF file first.');
    return;
  }

  if (!file.name.toLowerCase().endsWith('.pdf')) {
    setStatus(statusEl, 'error', '⚠ Only PDF files are supported.');
    return;
  }

  state.uploading = true;
  setBtnLoading(btn, true, 'Uploading...');
  setStatus(statusEl, 'loading', 'Uploading resume to agent memory...');

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`${API}/upload-resume`, { method: 'POST', body: form });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned ${res.status}`);
    }

    const data = await res.json();
    setStatus(statusEl, 'success', `✓ Resume ingested. ${data.message || 'Agent is ready.'}`);

    // Visual feedback on zone
    zone.style.borderColor = 'var(--green)';
    zone.style.borderStyle = 'solid';
    zone._droppedFile = null;

  } catch (err) {
    const msg = isNetworkError(err)
      ? '⚠ Unable to reach backend. Is the server running on port 8000?'
      : `⚠ ${err.message}`;
    setStatus(statusEl, 'error', msg);
  } finally {
    state.uploading = false;
    setBtnLoading(btn, false);
  }
}

// ══════════════════════════════════════════════
// GENERATE QUESTION
// Two-phase loading: Context → Generation
// ══════════════════════════════════════════════
async function generateQuestion() {
  if (state.generating) return;

  const btn = $('gen-btn');
  const card = $('question-card');
  const textEl = $('question-text');
  const stageCtx = $('stage-context');
  const stageGen = $('stage-generate');
  const stages = $('gen-stages');

  state.generating = true;
  setBtnLoading(btn, true, 'Generating...');

  // Reset stages
  resetStages(stageCtx, stageGen);

  // Fade out old question
  if (!card.classList.contains('hidden')) {
    card.style.opacity = '0';
    card.style.transform = 'translateY(-8px)';
    await sleep(200);
    card.classList.add('hidden');
  }

  // ── Phase 1: Retrieving Context ──────────────
  activateStage(stageCtx);
  stages.style.opacity = '1';

  await sleep(900); // Simulate context retrieval delay (or real API call)

  try {
    // Start the actual fetch during "Generating" phase
    doneStage(stageCtx);
    activateStage(stageGen);

    const res = await fetch(`${API}/generate-question`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned ${res.status}`);
    }

    const data = await res.json();
    state.currentQuestion = data.question || 'The agent did not return a question.';

    // ── Phase 2 done: show question ─────────────
    doneStage(stageGen);

    await sleep(150);

    textEl.textContent = state.currentQuestion;
    card.classList.remove('hidden');
    card.style.opacity = '1';
    card.style.transform = 'translateY(0)';

    // Fade stages back out after success
    setTimeout(() => { stages.style.opacity = '0.4'; }, 1500);

  } catch (err) {
    const msg = isNetworkError(err)
      ? 'Cannot reach backend. Make sure the server is running on port 8000.'
      : err.message;

    errorStage(stageGen);
    textEl.textContent = `⚠ ${msg}`;
    card.classList.remove('hidden');
    card.style.opacity = '1';
    card.style.transform = 'translateY(0)';
  } finally {
    state.generating = false;
    setBtnLoading(btn, false);
  }
}

// ══════════════════════════════════════════════
// SUBMIT ANSWER
// ══════════════════════════════════════════════
async function submitAnswer() {
  if (state.evaluating) return;

  const answer = $('answer-input').value.trim();
  const btn = $('submit-btn');
  const statusEl = $('eval-status');

  if (!answer) {
    toast('Type your answer before submitting.', 'warn');
    return;
  }

  if (!state.currentQuestion) {
    toast('Generate a question first, then submit your answer.', 'warn');
    return;
  }

  state.evaluating = true;
  setBtnLoading(btn, true, 'Evaluating Response...');
  showInlineStatus(statusEl, 'Evaluating Response...');

  try {
    const res = await fetch(`${API}/evaluate-answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: state.currentQuestion,
        answer,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned ${res.status}`);
    }

    const data = await res.json();
    renderResult(data);

    // Scroll to result
    $('module-result').scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    const msg = isNetworkError(err)
      ? 'Cannot reach backend. Check that the server is running on port 8000.'
      : err.message;
    toast(`⚠ ${msg}`, 'error');
  } finally {
    state.evaluating = false;
    setBtnLoading(btn, false);
    hideInlineStatus(statusEl);
  }
}

// ── Render Result ────────────────────────────
function renderResult(data) {
  const score = parseFloat(data.score) || 0;
  const clamped = Math.max(0, Math.min(10, score));

  // Show content
  $('result-empty').classList.add('hidden');
  const content = $('result-content');
  content.classList.remove('hidden');

  // Score number
  $('score-display').textContent = clamped.toFixed(1);

  // Score grade label
  const gradeEl = $('score-grade');
  if (clamped >= 9) { gradeEl.textContent = 'Exceptional'; gradeEl.style.color = '#4ade80'; }
  else if (clamped >= 7.5) { gradeEl.textContent = 'Proficient'; gradeEl.style.color = '#22d3ee'; }
  else if (clamped >= 6) { gradeEl.textContent = 'Competent'; gradeEl.style.color = '#3b82f6'; }
  else if (clamped >= 4) { gradeEl.textContent = 'Developing'; gradeEl.style.color = '#fbbf24'; }
  else { gradeEl.textContent = 'Needs Work'; gradeEl.style.color = '#f87171'; }

  // Animate SVG arc — circumference = 2π × 68 ≈ 427.26
  const offset = 427.26 - (427.26 * clamped / 10);
  requestAnimationFrame(() => {
    $('score-arc').style.strokeDashoffset = offset;
  });

  // Arc gradient colours by score
  let c0, c1;
  if (clamped >= 8) { c0 = '#4ade80'; c1 = '#22d3ee'; }
  else if (clamped >= 6) { c0 = '#22d3ee'; c1 = '#818cf8'; }
  else if (clamped >= 4) { c0 = '#fbbf24'; c1 = '#f97316'; }
  else { c0 = '#f87171'; c1 = '#e11d48'; }

  $('arc-stop-0').setAttribute('stop-color', c0);
  $('arc-stop-1').setAttribute('stop-color', c1);

  // Also update score number gradient
  $('score-display').style.backgroundImage = `linear-gradient(135deg, ${c0} 0%, ${c1} 100%)`;

  // Insight text fields
  $('strengths-text').textContent = data.strengths || data.strength || '—';
  $('weaknesses-text').textContent = data.weaknesses || data.weakness || '—';
  $('improvement-text').textContent = data.improvement || data.improvements || '—';
  $('followup-text').textContent = data.follow_up || data.followup || data.follow_up_question || '—';
}

// ══════════════════════════════════════════════
// RUN CODE
// ══════════════════════════════════════════════
async function runCode() {
  if (state.running) return;

  const code = $('code-input').value.trim();
  const language = $('lang-select').value;
  const btn = $('run-btn');
  const statusEl = $('run-status');
  const output = $('code-output');

  if (!code) {
    toast('Write some code before running.', 'warn');
    return;
  }

  state.running = true;
  setBtnLoading(btn, true, 'Running...');
  showInlineStatus(statusEl, 'Running code...');

  output.innerHTML = `<span class="t-prompt">ag@interview:~$</span> Executing ${language}...\n`;

  try {
    const res = await fetch(`${API}/run-code`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, language }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || data.error || `Server returned ${res.status}`);
    }

    const outText = data.output || data.result || '(no output)';
    const isErr = Boolean(data.error || data.stderr);
    const outClass = isErr ? ' class="t-error"' : '';

    output.innerHTML =
      `<span class="t-prompt">ag@interview:~$</span> Run complete · ${language}\n\n` +
      `<span${outClass}>${escHtml(String(outText))}</span>`;

  } catch (err) {
    const msg = isNetworkError(err)
      ? 'Cannot reach backend. Is the server running on port 8000?'
      : err.message;
    output.innerHTML =
      `<span class="t-prompt">ag@interview:~$</span> ` +
      `<span class="t-error">Error: ${escHtml(msg)}</span>`;
  } finally {
    state.running = false;
    setBtnLoading(btn, false);
    hideInlineStatus(statusEl);
  }
}

// ── Clear editor / terminal ──────────────────
function clearEditor() {
  $('code-input').value = '';
  updateGutter();
}

function clearOutput() {
  $('code-output').innerHTML =
    '<span class="t-prompt">ag@interview:~$</span> Sandbox ready. Awaiting execution...';
}

// ══════════════════════════════════════════════
// UI HELPERS
// ══════════════════════════════════════════════

function setStatus(el, type, msg) {
  if (!el) return;
  el.className = 'status-bar' + (type ? ` ${type}` : '');
  el.textContent = msg;
  if (!msg) el.style.display = 'none';
}

function showInlineStatus(el, msg) {
  if (!el) return;
  el.innerHTML = `<span class="spin"></span>${msg}`;
  el.classList.add('show');
}

function hideInlineStatus(el) {
  if (!el) return;
  el.innerHTML = '';
  el.classList.remove('show');
}

function setBtnLoading(btn, loading, label = '') {
  if (!btn) return;
  btn.disabled = loading;
  if (loading) {
    btn.dataset.orig = btn.innerHTML;
    btn.innerHTML = `<span class="spin"></span>${label}`;
  } else {
    btn.innerHTML = btn.dataset.orig || '';
  }
}

// Stage helpers for question generation
function resetStages(s1, s2) {
  [s1, s2].forEach(s => {
    s.classList.remove('active', 'done', 'error');
  });
}

function activateStage(s) {
  s.classList.remove('done', 'error');
  s.classList.add('active');
}

function doneStage(s) {
  s.classList.remove('active', 'error');
  s.classList.add('done');
}

function errorStage(s) {
  s.classList.remove('active', 'done');
  s.classList.add('error');
}

function isNetworkError(err) {
  return err instanceof TypeError && err.message.includes('fetch');
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Toast Notification ────────────────────────
function toast(msg, type = 'info') {
  const old = document.getElementById('ag-toast');
  if (old) old.remove();

  const colors = {
    info: ['rgba(34,211,238,0.1)', 'rgba(34,211,238,0.30)', '#22d3ee'],
    error: ['rgba(248,113,113,0.1)', 'rgba(248,113,113,0.30)', '#f87171'],
    warn: ['rgba(251,191,36,0.1)', 'rgba(251,191,36,0.30)', '#fbbf24'],
  };
  const [bg, border, color] = colors[type] || colors.info;

  const el = document.createElement('div');
  el.id = 'ag-toast';
  Object.assign(el.style, {
    position: 'fixed', bottom: '28px', right: '28px', zIndex: '9999',
    padding: '12px 20px', maxWidth: '380px',
    background: bg, border: `1px solid ${border}`, borderRadius: '10px',
    color, fontFamily: 'Inter, sans-serif', fontSize: '13px', fontWeight: '600',
    lineHeight: '1.5', backdropFilter: 'blur(20px)',
    boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
    animation: 'toastIn 0.3s cubic-bezier(0.34,1.56,0.64,1)',
  });
  el.textContent = msg;
  document.body.appendChild(el);

  setTimeout(() => {
    el.style.animation = 'toastOut 0.25s ease forwards';
    setTimeout(() => el.remove(), 250);
  }, 3800);
}

// Inject toast keyframes
const _ts = document.createElement('style');
_ts.textContent = `
@keyframes toastIn  { from { opacity:0; transform:translateY(16px) scale(0.95); } to { opacity:1; transform:translateY(0) scale(1); } }
@keyframes toastOut { from { opacity:1; transform:translateY(0); } to { opacity:0; transform:translateY(8px); } }
`;
document.head.appendChild(_ts);

// ══════════════════════════════════════════════
// PARTICLE CANVAS
// ══════════════════════════════════════════════
(function initParticles() {
  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, pts;
  const COUNT = 60;

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function mkPts() {
    pts = Array.from({ length: COUNT }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.6 + 0.3,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3 - 0.05,
      a: Math.random() * 0.55 + 0.15,
      // Alternate cyan/purple
      col: Math.random() > 0.45 ? '34,211,238' : '129,140,248',
    }));
  }

  function drawConnections() {
    for (let i = 0; i < pts.length - 1; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].x - pts[j].x;
        const dy = pts[i].y - pts[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 140) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(34,211,238,${(1 - dist / 140) * 0.18})`;
          ctx.lineWidth = 0.6;
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.stroke();
        }
      }
    }
  }

  let raf;
  function frame() {
    ctx.clearRect(0, 0, W, H);
    drawConnections();

    for (const p of pts) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < -8) p.x = W + 8;
      if (p.x > W + 8) p.x = -8;
      if (p.y < -8) p.y = H + 8;
      if (p.y > H + 8) p.y = -8;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.col},${p.a})`;
      ctx.fill();
    }

    raf = requestAnimationFrame(frame);
  }

  resize();
  mkPts();
  frame();

  window.addEventListener('resize', () => {
    cancelAnimationFrame(raf);
    resize();
    mkPts();
    frame();
  }, { passive: true });
})();

// ══════════════════════════════════════════════
// DRAG & DROP (Upload Zone)
// ══════════════════════════════════════════════
(function initDragDrop() {
  const zone = $('upload-zone');
  if (!zone) return;

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  }, { passive: false });

  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));

  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file) {
      zone._droppedFile = file;
      $('upload-selected').textContent = `📄 ${file.name}`;
      setStatus($('upload-status'), 'loading', `Ready: ${file.name} — click Upload Resume.`);
    }
  });
})();

// ── File input change ─────────────────────────
const _fi = $('resume-file');
if (_fi) {
  _fi.addEventListener('change', () => {
    const f = _fi.files[0];
    if (f) {
      $('upload-selected').textContent = `📄 ${f.name}`;
      setStatus($('upload-status'), 'loading', `Ready: ${f.name} — click Upload Resume.`);
    }
  });
}
