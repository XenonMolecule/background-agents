(function () {
  const CONTEXT_PATH = '../context_log.csv';
  const SURVEY_PATH = '../survey_responses.csv';
  const PROJ_PRED_PATH = (new URL(window.location.href)).searchParams.get('pred')
    || '../analysis/project_classification/results/context_project_predictions_latest.csv';
  const TRANSITIONS_PATH = (new URL(window.location.href)).searchParams.get('trans')
    || '../analysis/transition_classifier/results/transitions_latest.csv';
  const SCRATCHPAD_PATH = (new URL(window.location.href)).searchParams.get('spad')
    || '../analysis/project_scratchpad/results/project_scratchpad_latest.csv';
  const REPO_ROOT_PREFIX = '/Users/michaelryan/Documents/School/Stanford/Research/background-agents';

  const shotEl = document.getElementById('shot');
  const timestampEl = document.getElementById('timestamp');
  const badgeEl = document.getElementById('badge');
  const transitionTagEl = document.getElementById('transitionTag');
  const panelEl = document.getElementById('panel');
  const counterEl = document.getElementById('counter');
  const btnPrev = document.getElementById('prev');
  const btnNext = document.getElementById('next');
  const btnExportPng = document.getElementById('exportPng');
  const btnExportAllPng = document.getElementById('exportAllPng');
  const btnExportPptx = document.getElementById('exportPptx');
  const btnRefresh = document.getElementById('refresh');

  let slides = [];
  let idx = 0;

  // Cache-buster: enable with ?refresh=1 or ?nocache=1 or ?force=1
  const PAGE_URL = new URL(window.location.href);
  const FORCE_REFRESH = PAGE_URL.searchParams.has('refresh') || PAGE_URL.searchParams.has('nocache') || PAGE_URL.searchParams.has('force');
  const CACHE_BUSTER = FORCE_REFRESH ? `cb=${Date.now()}` : '';

  const RUBRIC_ACCURACY = {
    1: 'This is completely irrelevant to the task at hand',
    2: 'This is related to the task but not useful',
    3: 'This is useful/relevant but not what I am working on',
    4: 'This is close to what I am working on but not exact',
    5: 'This is exactly what I am working on'
  };
  const RUBRIC_CONTEXT = {
    1: 'A computer-use agent would be led astray by this',
    2: 'Not enough information to meaningfully help',
    3: 'Ambiguous but there is a chance the agent works',
    4: 'A strong agent would be able to work with this, but missing some detail',
    5: 'Provides enough context for a capable agent to take over the task'
  };

  function toRelativePath(absPath) {
    if (!absPath) return '';
    const p = String(absPath).trim();
    if (p.startsWith(REPO_ROOT_PREFIX)) {
      return p.replace(REPO_ROOT_PREFIX, '').replace(/^\/+/, '');
    }
    return p;
  }

  function getExportScale() {
    try {
      const url = new URL(window.location.href);
      const raw = url.searchParams.get('scale');
      const val = raw ? parseFloat(raw) : NaN;
      const dpr = window.devicePixelRatio || 1;
      const desired = isNaN(val) ? Math.max(3, dpr) : val;
      return Math.max(1, Math.min(5, desired));
    } catch {
      return 3;
    }
  }

  function fmtDate(ts) {
    if (!ts) return '';
    try {
      // Try ISO first
      const d = new Date(ts);
      if (!isNaN(d.getTime())) {
        const date = d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: '2-digit' });
        const time = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
        return `${date} ${time}`;
      }
      // Fallback: context timestamp like 20251016_174439 → yyyy-mm-dd hh:mm
      const m = String(ts).match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
      if (m) {
        const [_, y, mo, da, hh, mm] = m;
        return `${y}-${mo}-${da} ${hh}:${mm}`;
      }
      return String(ts);
    } catch {
      return String(ts);
    }
  }

  function clampText(text, maxChars) {
    if (!text) return '';
    const s = String(text);
    if (s.length <= maxChars) return s;
    return s.slice(0, Math.max(0, maxChars - 1)) + '…';
  }

  function createDots(selected) {
    const wrap = document.createElement('div');
    wrap.className = 'scale';
    for (let i = 1; i <= 5; i++) {
      const dot = document.createElement('span');
      dot.className = 'dot' + (Number(selected) === i ? ' active' : '');
      dot.textContent = String(i);
      wrap.appendChild(dot);
    }
    return wrap;
  }

  function makeSection(titleText) {
    const sec = document.createElement('div');
    sec.className = 'section';
    const title = document.createElement('div');
    title.className = 'section-title';
    title.textContent = titleText;
    sec.appendChild(title);
    return { sec, title };
  }

  function makeCollapsible(headerText, bodyNode) {
    const wrap = document.createElement('div');
    wrap.className = 'collapsible';
    const head = document.createElement('div');
    head.className = 'colla-head';
    const h = document.createElement('div');
    h.className = 'h';
    h.textContent = headerText;
    const toggle = document.createElement('div');
    toggle.className = 'toggle';
    toggle.textContent = 'Expand';
    head.appendChild(h);
    head.appendChild(toggle);
    const body = document.createElement('div');
    body.className = 'colla-body';
    body.appendChild(bodyNode);
    let expanded = false;
    function set(open) {
      expanded = open;
      body.style.display = open ? 'block' : 'none';
      toggle.textContent = open ? 'Collapse' : 'Expand';
    }
    head.addEventListener('click', () => set(!expanded));
    set(false);
    wrap.appendChild(head);
    wrap.appendChild(body);
    return wrap;
  }

  function prettyJsonNode(val, options = {}) {
    const { collapsed = false, maxItems = 20 } = options;
    let parsed = val;
    if (typeof val === 'string') {
      try { parsed = JSON.parse(val); } catch { /* keep as string */ }
    }
    if (typeof parsed !== 'object' || parsed === null) {
      const pre = document.createElement('pre');
      pre.className = 'monospace';
      pre.textContent = String(val);
      return pre;
    }
    const container = document.createElement('div');
    container.className = 'cards';
    const entries = Array.isArray(parsed) ? parsed.map((v, i) => [String(i + 1), v]) : Object.entries(parsed);
    const limited = entries.slice(0, maxItems);
    for (const [k, v] of limited) {
      const card = document.createElement('div');
      card.className = 'card';
      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = k;
      const body = document.createElement('div');
      body.className = 'monospace';
      try {
        body.textContent = typeof v === 'string' ? v : JSON.stringify(v, null, 2);
      } catch {
        body.textContent = String(v);
      }
      card.appendChild(title);
      card.appendChild(body);
      container.appendChild(card);
    }
    if (entries.length > limited.length) {
      const note = document.createElement('div');
      note.className = 'muted';
      note.textContent = `(+${entries.length - limited.length} more…)`;
      container.appendChild(note);
    }
    return container;
  }

  function renderContextSlide(s) {
    badgeEl.textContent = 'Context';
    timestampEl.textContent = fmtDate(s._ts_display);
    const rel = toRelativePath(s.screenshot_path);
    if (rel) {
      let url = rel.startsWith('/') ? rel : `/${rel}`;
      if (CACHE_BUSTER) url += (url.includes('?') ? '&' : '?') + CACHE_BUSTER;
      shotEl.src = url;
      shotEl.parentElement.parentElement.classList.remove('hidden');
    } else {
      shotEl.removeAttribute('src');
      shotEl.parentElement.parentElement.classList.remove('hidden');
    }

    panelEl.innerHTML = '';

    // Transition indicator (robust handling)
    const tr = s._transition;
    const trFlag = tr && String(tr.is_transition || '').trim().toUpperCase() === 'TRUE';
    if (trFlag) {
      const fromP = (tr.from_project || '').trim();
      const toP = (tr.to_project || '').trim();
      transitionTagEl.textContent = (fromP && toP) ? `Transition: ${fromP} → ${toP}` : 'Transition';
      transitionTagEl.classList.remove('hidden');
    } else {
      transitionTagEl.textContent = '';
      transitionTagEl.classList.add('hidden');
    }

    // Goals (top, compact + collapsible detailed)
    if (Array.isArray(s.goals) && s.goals.length) {
      const { sec } = makeSection('Inferred Goals');
      const cards = document.createElement('div');
      cards.className = 'cards';
      s.goals.slice(0, 5).forEach((g, i) => {
        const card = document.createElement('div');
        card.className = 'card';
        const title = document.createElement('div');
        title.className = 'title';
        title.textContent = g.title || g.name || `Goal ${i + 1}`;
        const desc = document.createElement('div');
        desc.className = 'muted';
        desc.textContent = clampText(g.description || g.desc || '', 220);
        card.appendChild(title);
        if (desc.textContent) card.appendChild(desc);
        cards.appendChild(card);
      });
      sec.appendChild(cards);
      panelEl.appendChild(sec);
    }

    // Predicted Project (if available)
    if (s.predicted_project) {
      const { sec } = makeSection('Predicted Project');
      const card = document.createElement('div');
      card.className = 'card';
      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = s.predicted_project;
      card.appendChild(title);
      sec.appendChild(card);
      panelEl.appendChild(sec);
    }

    // Scratchpad (collapsible, with project label if available)
    if (s.scratchpad_text) {
      const pre = document.createElement('pre');
      pre.className = 'monospace';
      pre.textContent = s.scratchpad_text;
      const label = s.scratchpad_project ? `Scratchpad (${s.scratchpad_project})` : 'Scratchpad';
      panelEl.appendChild(makeCollapsible(label, pre));
    }

    // Scratchpad Edit Summary (collapsible)
    if (s.scratchpad_summary) {
      const pre = document.createElement('pre');
      pre.className = 'monospace';
      pre.textContent = s.scratchpad_summary;
      panelEl.appendChild(makeCollapsible('Scratchpad Edit Summary', pre));
    }

    // Propositions / User details (collapsible)
    if (Array.isArray(s.user_details) && s.user_details.length) {
      const list = document.createElement('div');
      list.className = 'kv';
      s.user_details.slice(0, 20).forEach((p, i) => {
        const k = document.createElement('div');
        k.className = 'k';
        k.textContent = `Proposition ${i + 1}`;
        const v = document.createElement('div');
        v.className = 'v';
        v.textContent = clampText(p.text || p || '', 360);
        list.appendChild(k);
        list.appendChild(v);
      });
      panelEl.appendChild(makeCollapsible('Propositions', list));
    }

    // Calendar Events (collapsible)
    if (s.calendar_events_text) {
      const pre = document.createElement('pre');
      pre.className = 'monospace';
      pre.textContent = s.calendar_events_text;
      panelEl.appendChild(makeCollapsible('Calendar', pre));
    }

    // Context Update (collapsible)
    if (s.context_update_text) {
      const pre = document.createElement('pre');
      pre.className = 'monospace';
      pre.textContent = s.context_update_text;
      panelEl.appendChild(makeCollapsible('Context Update', pre));
    }

    // Recent Observations (collapsible)
    if (s.recent_observations_text) {
      const node = prettyJsonNode(s.recent_observations_text, { maxItems: 24 });
      panelEl.appendChild(makeCollapsible('Recent Observations', node));
    }

    // Reasoning (collapsible)
    if (s.reasoning_text) {
      const pre = document.createElement('pre');
      pre.className = 'monospace';
      pre.textContent = s.reasoning_text;
      panelEl.appendChild(makeCollapsible('Reasoning', pre));
    }
  }

  function renderSurveySlide(s) {
    badgeEl.textContent = 'Survey';
    timestampEl.textContent = fmtDate(s._ts_display);
    // Surveys generally have no screenshot; keep frame visible
    // Reuse previous screenshot if available
    if (!shotEl.getAttribute('src')) {
      const prevIdx = (idx - 1 + slides.length) % slides.length;
      const prev = slides[prevIdx];
      if (prev && prev.screenshot_path) {
        let url = toRelativePath(prev.screenshot_path);
        if (url) {
          url = url.startsWith('/') ? url : `/${url}`;
          if (CACHE_BUSTER) url += (url.includes('?') ? '&' : '?') + CACHE_BUSTER;
          shotEl.src = url;
        }
      }
    }
    shotEl.parentElement.parentElement.classList.remove('hidden');
    panelEl.innerHTML = '';
    // Tint background
    const slideRoot = document.getElementById('slide');
    slideRoot.classList.add('survey');

    // Helper for a titled section with QA boxes
    const qa = (q, a) => {
      const box = document.createElement('div');
      box.className = 'qa';
      const qEl = document.createElement('div');
      qEl.className = 'q';
      qEl.textContent = q;
      const aEl = document.createElement('div');
      aEl.className = 'a';
      aEl.textContent = a || '—';
      box.appendChild(qEl);
      box.appendChild(aEl);
      return box;
    };
    const addQASection = (title, items) => {
      const { sec } = makeSection(title);
      const wrap = document.createElement('div');
      wrap.className = 'qa-grid';
      items.forEach(([label, val]) => wrap.appendChild(qa(label, val)));
      sec.appendChild(wrap);
      panelEl.appendChild(sec);
    };

    // 1) What are you working on (project — task)
    addQASection('What I was actually doing', [
      ['What project are you working on right now?', s.project_now],
      ['What task are you working on right now?', s.task_now],
    ]);

    // 2) Background context (project and task)
    addQASection('Background context that could be helpful', [
      ['What background context could be helpful with this project?', s.helpful_project_context_now],
      ['What background context could be helpful with this task?', s.helpful_task_context_now],
    ]);

    // 3) Background work (project and task)
    addQASection('Background work that would have been helpful', [
      ['What background work would have been helpful with this project?', s.helpful_project_background_work_past],
      ['What background work would have been helpful for this task?', s.helpful_task_background_work_past],
    ]);
  }

  function renderCurrent() {
    if (!slides.length) return;
    const s = slides[idx];
    const slideRoot = document.getElementById('slide');
    slideRoot.classList.remove('survey');
    if (s._type === 'context') renderContextSlide(s);
    else renderSurveySlide(s);
    counterEl.textContent = `${idx + 1} / ${slides.length}`;
  }

  function next() { idx = (idx + 1) % slides.length; renderCurrent(); }
  function prev() { idx = (idx - 1 + slides.length) % slides.length; renderCurrent(); }

  btnNext.addEventListener('click', next);
  btnPrev.addEventListener('click', prev);
  if (btnExportPng) btnExportPng.addEventListener('click', exportCurrentPng);
  if (btnExportAllPng) btnExportAllPng.addEventListener('click', exportAllPngSequential);
  if (btnExportPptx) btnExportPptx.addEventListener('click', exportPptxAll);
  if (btnRefresh) btnRefresh.addEventListener('click', () => {
    const url = new URL(window.location.href);
    url.searchParams.set('refresh', '1');
    url.searchParams.set('cb', String(Date.now()));
    window.location.replace(url.toString());
  });

  function autoScale() {
    const slide = document.getElementById('slide');
    if (!slide) return;
    const wrap = document.getElementById('slideWrap');
    const vw = wrap ? wrap.clientWidth : window.innerWidth;
    const vh = wrap ? wrap.clientHeight : (window.innerHeight - 80);
    const scale = Math.min(vw / 1920, vh / 1080);
    slide.style.transform = `translate(-50%, -50%) scale(${scale})`;
    slide.style.marginLeft = '';
    slide.style.marginTop = '';
  }

  async function captureSlideCanvas(overrideScale) {
    const liveSlide = document.getElementById('slide');
    if (!liveSlide) throw new Error('Slide element not found');
    const container = document.createElement('div');
    container.style.position = 'fixed';
    container.style.left = '-99999px';
    container.style.top = '0';
    container.style.width = '1920px';
    container.style.height = '1080px';
    container.style.background = '#ffffff';
    container.style.overflow = 'hidden';
    const clone = liveSlide.cloneNode(true);
    clone.id = 'slide-capture';
    clone.style.position = 'static';
    clone.style.transform = 'none';
    clone.style.width = '1920px';
    clone.style.height = '1080px';
    container.appendChild(clone);
    document.body.appendChild(container);
    const canvas = await html2canvas(clone, {
      backgroundColor: '#ffffff',
      scale: overrideScale ?? getExportScale(),
      width: 1920,
      height: 1080,
      windowWidth: 1920,
      windowHeight: 1080
    });
    document.body.removeChild(container);
    return canvas;
  }

  async function exportCurrentPng() {
    const s = slides[idx];
    const ts = (s._ts_iso || s._ts_display || 'slide').toString().replaceAll(':', '-');
    const canvas = await captureSlideCanvas();
    const url = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = `slide_${String(idx + 1).padStart(2, '0')}_${ts}.png`;
    a.click();
  }

  async function exportAllPngSequential() {
    const original = idx;
    for (let i = 0; i < slides.length; i++) {
      idx = i;
      renderCurrent();
      await new Promise(r => setTimeout(r, 50));
      await exportCurrentPng();
      await new Promise(r => setTimeout(r, 30));
    }
    idx = original; renderCurrent();
  }

  async function exportPptxAll() {
    const pres = new PptxGenJS();
    pres.layout = 'LAYOUT_16x9';
    const original = idx;
    const safeScale = Math.min(getExportScale(), 3);
    for (let i = 0; i < slides.length; i++) {
      idx = i; renderCurrent();
      await new Promise(r => setTimeout(r, 50));
      const canvas = await captureSlideCanvas(safeScale);
      const dataUrl = canvas.toDataURL('image/png');
      const slide = pres.addSlide({ bkgd: 'FFFFFF' });
      slide.addImage({ data: dataUrl, x: -0.02, y: -0.01, w: 10.04, h: 5.645 });
    }
    idx = original; renderCurrent();
    await pres.writeFile({ fileName: 'survey_context_slides.pptx' });
  }

  // Parse timestamps robustly; return epoch ms (number) and pretty display string
  function parseAnyTimestamp(tsRaw) {
    if (!tsRaw) return { epoch: 0, display: '' };
    const s = String(tsRaw);
    // ISO and RFC
    const d = new Date(s);
    if (!isNaN(d.getTime())) return { epoch: d.getTime(), display: s };
    // 20251016_174439
    const m = s.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
    if (m) {
      const [, y, mo, da, hh, mm, ss] = m;
      const local = new Date(Number(y), Number(mo) - 1, Number(da), Number(hh), Number(mm), Number(ss));
      return { epoch: local.getTime(), display: s };
    }
    return { epoch: 0, display: s };
  }

  // Parse context log row to normalized structure
  function normalizeContextRow(r) {
    const tsRaw = r.timestamp || r.ts || '';
    const tsParsed = parseAnyTimestamp(tsRaw);
    let goals = [];
    try {
      goals = JSON.parse(r.goals || '[]');
    } catch {}
    let userDetails = [];
    try {
      userDetails = JSON.parse(r.user_details || '[]');
    } catch {}
    let calText = '';
    try {
      calText = JSON.parse(r.calendar_events || '""');
    } catch {
      calText = r.calendar_events || '';
    }
    return {
      _type: 'context',
      _ts_epoch: tsParsed.epoch,
      _ts_display: tsParsed.display,
      screenshot_path: r.screenshot_path || '',
      goals: goals,
      user_details: userDetails,
      calendar_events_text: calText,
      context_update_text: r.context_update || '',
      recent_observations_text: r.recent_observations || '',
      reasoning_text: r.reasoning || ''
    };
  }

  function normalizeSurveyRow(r) {
    const ts = r.timestamp || r.survey_timestamp || '';
    const tsParsed = parseAnyTimestamp(ts);
    return {
      _type: 'survey',
      _ts_epoch: tsParsed.epoch,
      _ts_display: tsParsed.display || ts,
      project_now: r.project_now || '',
      task_now: r.task_now || '',
      helpful_task_context_now: r.helpful_task_context_now || '',
      helpful_project_context_now: r.helpful_project_context_now || '',
      helpful_task_background_work_past: r.helpful_task_background_work_past || '',
      helpful_project_background_work_past: r.helpful_project_background_work_past || ''
    };
  }

  function parseCsv(path) {
    return new Promise((resolve, reject) => {
      const src = CACHE_BUSTER ? (path + (path.includes('?') ? '&' : '?') + CACHE_BUSTER) : path;
      Papa.parse(src, {
        download: true,
        header: true,
        skipEmptyLines: true,
        complete: (res) => resolve(res.data || []),
        error: (err) => reject(err)
      });
    });
  }

  function autoScaleAndRender() {
    autoScale();
    renderCurrent();
    window.addEventListener('resize', autoScale);
  }

  async function init() {
    try {
      const [ctxRows, surveyRows, predsRows, scratchRows, transRows] = await Promise.all([
        parseCsv(CONTEXT_PATH),
        parseCsv(SURVEY_PATH),
        parseCsv(PROJ_PRED_PATH).catch(() => []),
        parseCsv(SCRATCHPAD_PATH).catch(() => []),
        parseCsv(TRANSITIONS_PATH).catch(() => [])
      ]);
      const ctxSlides = ctxRows.map(normalizeContextRow);
      const surveySlides = surveyRows.map(normalizeSurveyRow);
      const predsMap = new Map();
      (predsRows || []).forEach(r => {
        if (r && r.timestamp) predsMap.set(String(r.timestamp), r.predicted_project);
      });
      const transMap = new Map();
      (transRows || []).forEach(r => {
        if (!r || !r.timestamp) return;
        const norm = {
          timestamp: String(r.timestamp),
          is_transition: String(r.is_transition || '').trim().toUpperCase(),
          from_project: r.from_project || '',
          to_project: r.to_project || '',
          smoothed_project: r.smoothed_project || ''
        };
        transMap.set(norm.timestamp, norm);
      });
      const scratchMap = new Map();
      (scratchRows || []).forEach(r => {
        if (r && r.timestamp) {
          scratchMap.set(String(r.timestamp), {
            project: r.project,
            scratchpad: r.scratchpad,
            summary: r.summary
          });
        }
      });
      // attach predictions onto context slides for display later if desired
      ctxSlides.forEach(s => {
        const key = s._ts_display || '';
        if (predsMap.has(key)) s.predicted_project = predsMap.get(key);
        if (transMap.has(key)) s._transition = transMap.get(key);
        if (scratchMap.has(key)) {
          const sp = scratchMap.get(key);
          s.scratchpad_text = sp.scratchpad || '';
          s.scratchpad_summary = sp.summary || '';
          s.scratchpad_project = sp.project || '';
        }
      });
      slides = [...ctxSlides, ...surveySlides].sort((a, b) => (a._ts_epoch || 0) - (b._ts_epoch || 0));
      // Jump params: i (index), ts (timestamp substring match)
      const url = new URL(window.location.href);
      const iParam = url.searchParams.get('i');
      const tsParam = url.searchParams.get('ts');
      if (iParam) {
        const n = parseInt(iParam, 10);
        if (!isNaN(n)) idx = Math.max(0, Math.min(slides.length - 1, n));
      } else if (tsParam) {
        const target = tsParam.trim();
        const found = slides.findIndex(s => String(s._ts_display).includes(target));
        if (found >= 0) idx = found;
      }
      if (!slides.length) { counterEl.textContent = '0 / 0'; return; }
      autoScaleAndRender();
    } catch (e) {
      console.error('Failed to load CSVs', e);
      counterEl.textContent = 'Failed to load CSVs';
    }
  }

  init();
})();


