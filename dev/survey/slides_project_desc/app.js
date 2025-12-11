(function () {
  // Defaults (can be overridden via URL params)
  // Serve from REPO ROOT for these absolute paths to work:
  //   python3 -m http.server 8080  (in repo root)
  const DEFAULT_DESC = '/dev/survey/data_collection/12_3_experiments/project_descriptions.csv';
  const DEFAULT_LOG = '/dev/survey/data_collection/11_20_experiments/no_user/log.csv';
  const DEFAULT_PIPELINE = '/dev/survey/pipeline_run.csv';
  const REPO_ROOT_PREFIX = (new URL(window.location.href)).searchParams.get('root')
    || '/Users/michaelryan/Documents/School/Stanford/Research/background-agents';

  const PAGE_URL = new URL(window.location.href);
  const FORCE_REFRESH = PAGE_URL.searchParams.has('refresh') || PAGE_URL.searchParams.has('force') || PAGE_URL.searchParams.has('nocache');
  const CACHE_BUSTER = FORCE_REFRESH ? `cb=${Date.now()}` : '';

  // Elements
  const projectSelect = document.getElementById('projectSelect');
  const stepSelect = document.getElementById('stepSelect');
  const counterEl = document.getElementById('counter');
  const btnPrev = document.getElementById('prev');
  const btnNext = document.getElementById('next');
  const btnRefresh = document.getElementById('refresh');
  const shotEl = document.getElementById('shot');
  const timestampEl = document.getElementById('timestamp');
  const panelEl = document.getElementById('panel');
  const descPathEl = document.getElementById('descPath');
  const logPathEl = document.getElementById('logPath');
  const prPathEl = document.getElementById('prPath');

  // State
  let allRows = []; // normalized joined rows (for the selected project)
  let byProject = new Map(); // project -> rows[]
  let idx = 0;

  // Helpers
  function toRelativePath(absPath) {
    if (!absPath) return '';
    const p = String(absPath).trim();
    if (p.startsWith(REPO_ROOT_PREFIX)) {
      return p.replace(REPO_ROOT_PREFIX, '').replace(/^\/+/, '');
    }
    return p;
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

  function parseAnyTimestamp(tsRaw) {
    if (!tsRaw) return { epoch: 0, display: '' };
    const s = String(tsRaw);
    const d = new Date(s);
    if (!isNaN(d.getTime())) return { epoch: d.getTime(), display: s };
    const m = s.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
    if (m) {
      const [, y, mo, da, hh, mm, ss] = m;
      const local = new Date(Number(y), Number(mo) - 1, Number(da), Number(hh), Number(mm), Number(ss));
      return { epoch: local.getTime(), display: s };
    }
    return { epoch: 0, display: s };
  }

  function fmtDate(ts) {
    if (!ts) return '';
    try {
      const d = new Date(ts);
      if (!isNaN(d.getTime())) {
        const date = d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: '2-digit' });
        const time = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
        return `${date} ${time}`;
      }
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

  function uniq(arr) { return Array.from(new Set(arr)); }

  function populateProjects() {
    const projects = uniq(Array.from(byProject.keys()).filter(p => p));
    projects.sort((a, b) => a.localeCompare(b));
    projectSelect.innerHTML = '';
    projects.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p;
      projectSelect.appendChild(opt);
    });
    const fromUrl = PAGE_URL.searchParams.get('project');
    if (fromUrl && projects.includes(fromUrl)) {
      projectSelect.value = fromUrl;
    }
  }

  function populateSteps() {
    stepSelect.innerHTML = '';
    allRows.forEach((r, i) => {
      const opt = document.createElement('option');
      const label = `${String(r.step_index || (i + 1)).padStart(2,'0')} — ${r.timestamp}`;
      opt.value = String(i);
      opt.textContent = label;
      stepSelect.appendChild(opt);
    });
    stepSelect.value = String(idx);
  }

  function setImage(path) {
    const rel = toRelativePath(path);
    // Reset previous handlers
    shotEl.onerror = null;
    shotEl.onload = null;
    if (rel) {
      let url = rel.startsWith('/') ? rel : `/${rel}`;
      if (CACHE_BUSTER) url += (url.includes('?') ? '&' : '?') + CACHE_BUSTER;
      shotEl.onload = () => { /* ok */ };
      shotEl.onerror = () => {
        // Clear image and show a warning banner in the panel header
        shotEl.removeAttribute('src');
        const warn = document.createElement('div');
        warn.className = 'muted';
        warn.textContent = `Image not found: ${path}`;
        // Insert warning at top of panel
        if (panelEl.firstChild) panelEl.insertBefore(warn, panelEl.firstChild);
        else panelEl.appendChild(warn);
      };
      shotEl.src = url;
    } else {
      shotEl.removeAttribute('src');
    }
  }

  function renderCurrent() {
    counterEl.textContent = `${allRows.length ? (idx + 1) : 0} / ${allRows.length}`;
    stepSelect.value = String(idx);
    if (!allRows.length) {
      timestampEl.textContent = '';
      panelEl.innerHTML = '';
      shotEl.removeAttribute('src');
      return;
    }
    const r = allRows[idx];
    timestampEl.textContent = fmtDate(r.timestamp);
    setImage(r.screenshot_path || r.pipeline_screenshot_path || r.log_screenshot_path || '');

    panelEl.innerHTML = '';
    // Primary: Inferred Project Description (first-class)
    const { sec: descSec } = makeSection('Inferred Project Description');
    const descCard = document.createElement('div');
    descCard.className = 'card';
    const title = document.createElement('div');
    title.className = 'title';
    title.textContent = r.project || '';
    const body = document.createElement('div');
    body.className = 'monospace';
    body.textContent = r.updated_project_description || 'N/A';
    descCard.appendChild(title);
    descCard.appendChild(body);
    descSec.appendChild(descCard);
    panelEl.appendChild(descSec);

    // Context Update (collapsible)
    if (r.log_context_update) {
      const pre = document.createElement('pre');
      pre.className = 'monospace';
      pre.textContent = r.log_context_update || '';
      panelEl.appendChild(makeCollapsible('Context Update', pre));
    }

    // Scratchpad (collapsible)
    if (r.log_scratchpad_text) {
      const pre = document.createElement('pre');
      pre.className = 'monospace';
      pre.textContent = r.log_scratchpad_text || '';
      panelEl.appendChild(makeCollapsible('Scratchpad', pre));
    }

    // Raw Joined Row (collapsible)
    const pre = document.createElement('pre');
    pre.className = 'monospace';
    try {
      pre.textContent = JSON.stringify(r, null, 2);
    } catch {
      pre.textContent = String(r);
    }
    panelEl.appendChild(makeCollapsible('Raw (joined)', pre));
  }

  function next() { idx = (idx + 1) % (allRows.length || 1); renderCurrent(); }
  function prev() { idx = (idx - 1 + (allRows.length || 1)) % (allRows.length || 1); renderCurrent(); }

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

  function autoScaleAndRender() {
    autoScale();
    renderCurrent();
    window.addEventListener('resize', autoScale);
  }

  async function loadAndBuild() {
    const descCsv = PAGE_URL.searchParams.get('desc') || DEFAULT_DESC;
    const logCsv = PAGE_URL.searchParams.get('log') || DEFAULT_LOG;
    const prCsv = PAGE_URL.searchParams.get('pr') || DEFAULT_PIPELINE;
    if (descPathEl) descPathEl.value = descCsv;
    if (logPathEl) logPathEl.value = logCsv;
    if (prPathEl) prPathEl.value = prCsv;

    const [descRows, logRows, prRows] = await Promise.all([
      parseCsv(descCsv),
      parseCsv(logCsv).catch(() => []),
      parseCsv(prCsv).catch(() => []),
    ]);

    // Build pipeline index: (project, timestamp) -> screenshot_path
    const pipeIdx = new Map();
    (prRows || []).forEach(r => {
      const k = `${(r.project || '').trim()}__${(r.timestamp || '').trim()}`;
      const shot = (r.screenshot_path || '').trim();
      if (k && shot) pipeIdx.set(k, shot);
    });

    // Build log index: (project, timestamp) -> {context_update, scratchpad_text, screenshot_path}
    const logIdx = new Map();
    (logRows || []).forEach(r => {
      const proj = (r.project || '').trim();
      const ts = (r.timestamp || '').trim();
      const k = `${proj}__${ts}`;
      if (!k) return;
      logIdx.set(k, {
        context_update: r.context_update || '',
        scratchpad_text: r.scratchpad_text || '',
        screenshot_path: r.screenshot_path || ''
      });
    });

    // Join to normalized rows, then group by project
    const normalized = (descRows || []).map(r => {
      const proj = (r.project || '').trim();
      const ts = (r.timestamp || '').trim();
      const k = `${proj}__${ts}`;
      const pipe = pipeIdx.get(k) || '';
      const l = logIdx.get(k) || { context_update: '', scratchpad_text: '', screenshot_path: '' };
      const tsParsed = parseAnyTimestamp(ts);
      return {
        project: proj,
        timestamp: ts,
        _ts_epoch: tsParsed.epoch,
        step_index: r.step_index || '',
        updated_project_description: r.updated_project_description || '',
        // screenshot precedence: project_descriptions -> pipeline_run -> log
        screenshot_path: r.screenshot_path || pipe || l.screenshot_path || '',
        pipeline_screenshot_path: pipe || '',
        log_screenshot_path: l.screenshot_path || '',
        log_context_update: l.context_update || '',
        log_scratchpad_text: l.scratchpad_text || '',
      };
    }).filter(r => r.project && r.timestamp);

    byProject = new Map();
    normalized.forEach(r => {
      const list = byProject.get(r.project) || [];
      list.push(r);
      byProject.set(r.project, list);
    });
    // Sort each project by timestamp, then step_index numeric
    for (const [p, list] of byProject.entries()) {
      list.sort((a, b) => {
        if (a._ts_epoch !== b._ts_epoch) return a._ts_epoch - b._ts_epoch;
        const ai = Number(a.step_index || 0) || 0;
        const bi = Number(b.step_index || 0) || 0;
        return ai - bi;
      });
    }

    populateProjects();
    const initialProject = projectSelect.value || Array.from(byProject.keys())[0] || '';
    setProject(initialProject);
  }

  function setProject(proj) {
    if (!proj || !byProject.has(proj)) {
      allRows = [];
      idx = 0;
      populateSteps();
      renderCurrent();
      return;
    }
    allRows = byProject.get(proj) || [];
    idx = 0;
    populateSteps();
    renderCurrent();
  }

  function jumpToIndex(i) {
    const n = parseInt(i, 10);
    if (isNaN(n)) return;
    idx = Math.max(0, Math.min(allRows.length - 1, n));
    renderCurrent();
  }

  function initControls() {
    btnNext.addEventListener('click', next);
    btnPrev.addEventListener('click', prev);
    btnRefresh.addEventListener('click', () => {
      const url = new URL(window.location.href);
      url.searchParams.set('refresh', '1');
      url.searchParams.set('cb', String(Date.now()));
      if (projectSelect.value) url.searchParams.set('project', projectSelect.value);
      if (descPathEl && descPathEl.value) url.searchParams.set('desc', descPathEl.value.trim());
      if (logPathEl && logPathEl.value) url.searchParams.set('log', logPathEl.value.trim());
      if (prPathEl && prPathEl.value) url.searchParams.set('pr', prPathEl.value.trim());
      window.location.replace(url.toString());
    });
    projectSelect.addEventListener('change', () => setProject(projectSelect.value));
    stepSelect.addEventListener('change', () => jumpToIndex(stepSelect.value));
    // Press Enter in any path box to refresh with new params
    [descPathEl, logPathEl, prPathEl].forEach(inp => {
      if (!inp) return;
      inp.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          btnRefresh.click();
        }
      });
    });
  }

  function autoScaleAndInit() {
    autoScale();
    window.addEventListener('resize', autoScale);
  }

  async function init() {
    try {
      initControls();
      autoScaleAndInit();
      await loadAndBuild();
    } catch (e) {
      console.error('Failed to initialize timeline viewer', e);
      counterEl.textContent = 'Failed to load CSVs';
    }
  }

  init();
})();


