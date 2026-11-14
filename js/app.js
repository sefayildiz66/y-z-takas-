// Yüz Takas Stüdyosu — telefonda çalışan sürüm (iPhone Safari, WebGPU)
import * as ort from '../vendor/ort.webgpu.bundle.min.mjs';
import { FaceEngine, meanEmbedding, cosine, probeFeeds, PROBE_STEP, PROBE_TOL } from './core.js';
import { demux, decoderSupported, decodeAll, playFrames, createWriter } from './video.js';

const $ = id => document.getElementById(id);
const L = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const sleep = ms => new Promise(r => setTimeout(r, ms));
const store = (k, v) => { try { if (v === undefined) return localStorage.getItem('yts-' + k); localStorage.setItem('yts-' + k, v); } catch {} return null; };
const fmtMB = b => Math.round(b / 1048576) + ' MB';

/* ================= modeller ================= */
const HF = repo => `https://huggingface.co/facefusion/${repo}/resolve/main/`;
const MODELS = {
  detector:   { file: 'scrfd_2.5g.onnx',         repo: 'models-3.0.0', size: 3295067,   label: 'Yüz tespiti',  gpu: false },
  recognizer: { file: 'arcface_w600k_r50.onnx',  repo: 'models-3.0.0', size: 174388474, label: 'Kimlik tanıma', gpu: true },
  swapper:    { file: 'inswapper_128_fp16.onnx', repo: 'models-3.0.0', size: 277680829, label: 'Yüz takası · InSwapper', gpu: true },
  hyperswap:  { file: 'hyperswap_1c_256.onnx',   repo: 'models-3.3.0', size: 402742682, label: 'Yüz takası · HyperSwap', gpu: true, preset: 'max' },
  enhancer:   { file: 'gfpgan_1.4.onnx',         repo: 'models-3.0.0', size: 340299087, label: 'İyileştirme · GFPGAN', gpu: true, preset: 'high' },
  occluder:   { file: 'xseg_1.onnx',             repo: 'models-3.1.0', size: 70324286,  label: 'Engel maskesi · XSeg', gpu: true, preset: 'high' },
  parser:     { file: 'bisenet_resnet_18.onnx',  repo: 'models-3.1.0', size: 53205356,  label: 'Yüz bölgesi · BiSeNet', gpu: true, preset: 'max' },
};
const PRESETS = {
  fast: { model: 'swapper', size: 128, enhance: false, occlusion: false, region: false, short: 720, bpp: 0.12,
    hint: 'InSwapper 128 px. En hızlısı; uzun videolar ve deneme için.' },
  high: { model: 'swapper', size: 256, enhance: false, occlusion: true, region: false, short: 1080, bpp: 0.16,
    hint: 'InSwapper 256 px + engel maskesi (el, saç, gözlük önündeyse korunur). Telefonda en iyi denge.' },
  max:  { model: 'hyperswap', size: 512, enhance: true, occlusion: true, region: true, short: 2160, bpp: 0.22,
    hint: 'HyperSwap 512 px + iyileştirme + engel ve yüz bölgesi maskeleri, orijinal çözünürlük. En iyi kalite ama çok bellek ister; uzun videolarda sayfa kapanabilir.' },
};
const needFor = p => ['detector', 'recognizer', PRESETS[p].model, ...(PRESETS[p].enhance ? ['enhancer'] : []), ...(PRESETS[p].occlusion ? ['occluder'] : []), ...(PRESETS[p].region ? ['parser'] : [])];

const CACHE = 'yts-models-v1', ckey = f => 'https://yts.cache/models/' + f;
ort.env.logLevel = 'error';
ort.env.wasm.wasmPaths = new URL('../vendor/', import.meta.url).href;
const IOS = /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
// iPhone'da tek iş parçacığı: daha az bellek (asıl iş zaten ekran kartında)
ort.env.wasm.numThreads = !IOS && self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;

const sessions = {}, modelEP = {}, modelState = {};
let engine = null, gpuOK = false, starting = false;

async function cacheHas(file) { try { return !!(await (await caches.open(CACHE)).match(ckey(file))); } catch { return false; } }
// İndirme belleğe alınmadan doğrudan telefonun depolamasına akar (iPhone'da sayfanın kapanmasını önler)
async function download(spec, onp) {
  let cache = null; try { cache = await caches.open(CACHE); } catch {}
  if (cache && await cache.match(ckey(spec.file))) { onp(1); return; }
  if (!cache) return; // önbellek yoksa (gizli sekme) model yüklenirken doğrudan indirilir
  let lastErr;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const r = await fetch(HF(spec.repo) + spec.file);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const total = +r.headers.get('content-length') || spec.size; let got = 0;
      const counter = new TransformStream({ transform(chunk, ctl) { got += chunk.byteLength; onp(Math.min(1, got / total)); ctl.enqueue(chunk); } });
      await cache.put(ckey(spec.file), new Response(r.body.pipeThrough(counter), { headers: { 'content-type': 'application/octet-stream', 'content-length': String(total) } }));
      if (got !== total) { await cache.delete(ckey(spec.file)); throw new Error('eksik indirme'); }
      return;
    } catch (e) {
      lastErr = e; console.warn('indirme', spec.file, e);
      try { await cache.delete(ckey(spec.file)); } catch {}
      // Depolama dolu / yazılamıyor: kaydetmeden devam edilir, model yüklenirken doğrudan indirilir
      if (/Quota|Cache|internal/i.test(String(e?.name) + String(e?.message))) { storageWarn(); return; }
      await sleep(2000 * attempt);
    }
  }
  throw new Error(`${spec.label} indirilemedi (${lastErr?.message}). Wi-Fi bağlantını kontrol edip tekrar dene.`);
}
function storageWarn() {
  $('modelHint').innerHTML = '<span class="err">Modeller telefona kaydedilemedi (depolama dolu olabilir). Uygulama çalışır ama her açılışta modeller yeniden iner. iPhone\'da yer açıp sayfayı yenile.</span>';
}
async function getModel(spec, onp = () => {}) {
  let cache = null; try { cache = await caches.open(CACHE); } catch {}
  if (cache) { const hit = await cache.match(ckey(spec.file)); if (hit) { onp(1); return new Uint8Array(await hit.arrayBuffer()); } }
  const r = await fetch(HF(spec.repo) + spec.file);
  if (!r.ok) throw new Error(`${spec.label} indirilemedi (HTTP ${r.status})`);
  return new Uint8Array(await r.arrayBuffer());
}
// Önce eksik modellerin hepsi indirilir, sonra yüklenir (aynı anda bellekte daha az veri)
async function downloadAll(keys) {
  const missing = keys.filter(k => modelState[k] === 'missing');
  const total = missing.reduce((a, k) => a + MODELS[k].size, 0); let doneBytes = 0;
  for (const k of missing) {
    const spec = MODELS[k];
    setStatus('busy', 'İndiriliyor');
    crumb(`${k}:dl`);
    await download(spec, p => { modelState[k] = '%' + Math.floor(p * 100); renderModels(); setBar('model', (doneBytes + p * spec.size) / total); });
    crumb(''); doneBytes += spec.size; modelState[k] = 'cached'; renderModels();
  }
  $('modelProg').hidden = true;
}

/* Tüm model çalıştırmaları tek sırada yürür. Ekran kartı bir çalıştırmada hata verirse
   o model otomatik olarak işlemciye taşınır ve aynı iş tekrar denenir. */
let chain = Promise.resolve();
const serial = fn => { const p = chain.then(fn, fn); chain = p.catch(() => {}); return p; };
const opts = ep => ({ executionProviders: [ep], graphOptimizationLevel: 'all', enableCpuMemArena: false, enableMemPattern: false });
class SafeSession {
  constructor(key, s) { this.key = key; this.s = s; }
  get inputNames() { return this.s.inputNames; }
  get outputNames() { return this.s.outputNames; }
  run(feeds) {
    return serial(async () => {
      try { return await this.s.run(feeds); }
      catch (e) {
        if (modelEP[this.key] !== 'GPU') throw e;
        console.warn('Ekran kartı hatası, işlemciye geçiliyor:', this.key, e?.message || e);
        const bytes = await getModel(MODELS[this.key], () => {});
        const cpu = await ort.InferenceSession.create(bytes, opts('wasm'));
        try { await this.s.release(); } catch {}
        this.s = cpu; modelEP[this.key] = 'CPU'; renderModels();
        $('modelHint').innerHTML = '<span class="err">Ekran kartı bir hata verdi; işlem durmadan işlemciye geçildi (daha yavaş). Hızı geri almak için sayfayı yenile.</span>';
        return await this.s.run(feeds);
      }
    });
  }
  release() { return this.s.release(); }
}
let probeRef = null;
async function validate(sess, key) {
  probeRef ??= await (await fetch(new URL('../assets/probe_ref.json', import.meta.url))).json();
  const ref = probeRef[key], feeds = probeFeeds(key, ort.Tensor), fd = {};
  sess.inputNames.forEach((n, i) => fd[n] = feeds[n] || Object.values(feeds)[i]);
  const out = (await serial(() => sess.run(fd)))[sess.outputNames[0]].data;
  let d = 0; for (let i = 0, j = 0; j < ref.length; i += PROBE_STEP, j++) d += Math.abs(out[i] - ref[j]);
  d /= ref.length; console.info('GPU doğrulama', key, d.toFixed(5));
  return Number.isFinite(d) && d < PROBE_TOL[key];
}
async function loadModel(key) {
  if (sessions[key]) return sessions[key];
  const spec = MODELS[key];
  const prog = p => { modelState[key] = '%' + Math.floor(p * 100); renderModels(); setBar('model', p); };
  setStatus('busy', spec.label.split(' · ').pop() + ' yükleniyor');
  await download(spec, prog);
  let bytes = await getModel(spec), s = null;
  setStatus('busy', spec.label.split(' · ').pop() + ' hazırlanıyor');
  if (gpuOK && spec.gpu && !store('nogpu-' + key)) {
    try {
      crumb(`${key}:gpu`);
      s = await ort.InferenceSession.create(bytes, opts('webgpu'));
      if (!(await validate(s, key))) { await s.release(); s = null; }
    } catch (e) { console.warn('WebGPU açılamadı', key, e?.message || e); s = null; }
  }
  modelEP[key] = s ? 'GPU' : 'CPU';
  if (!s) { crumb(`${key}:cpu`); s = await ort.InferenceSession.create(bytes, opts('wasm')); }
  crumb('');
  bytes = null;
  sessions[key] = new SafeSession(key, s); modelState[key] = 'loaded'; $('modelProg').hidden = true; renderModels();
  return sessions[key];
}
async function unloadModel(key) { const s = sessions[key]; if (!s) return; delete sessions[key]; modelState[key] = 'cached'; try { await s.release(); } catch {} renderModels(); }

// Seçili profil için gereken modelleri yükler, gereksiz takas modelini bırakır (telefon belleği için)
async function prepare(presetKey) {
  const p = PRESETS[presetKey];
  const other = p.model === 'swapper' ? 'hyperswap' : 'swapper';
  if (sessions[other]) await unloadModel(other);
  const need = new Set(needFor(presetKey));
  for (const k of ['enhancer', 'occluder', 'parser']) if (!need.has(k) && sessions[k]) await unloadModel(k);
  await downloadAll([...need]);
  for (const k of needFor(presetKey)) await loadModel(k);
  engine.s.swapper = sessions[p.model]; engine.s.swapperType = p.model === 'hyperswap' ? 'hyperswap' : 'inswapper';
  engine.s.enhancer = sessions.enhancer; engine.s.occluder = sessions.occluder; engine.s.parser = sessions.parser;
  readyStatus();
}

async function detectGPU() {
  if (!navigator.gpu) return false;
  try { return !!(await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })); } catch { return false; }
}
async function startEngine() {
  if (starting || engine) return; starting = true; renderModels(); refresh();
  try {
    try { await navigator.storage?.persist?.(); } catch {}
    const emap = new Float32Array(await (await fetch(new URL('../assets/inswapper_emap.bin', import.meta.url))).arrayBuffer());
    await downloadAll(needFor(S.preset));
    for (const k of needFor(S.preset)) await loadModel(k);
    engine = new FaceEngine(ort, { detector: sessions.detector, recognizer: sessions.recognizer }, emap);
    await prepare(S.preset);
  } catch (e) {
    console.error(e); setStatus('error', 'Model hatası');
    $('modelHint').innerHTML = `<span class="err">${e.message || e}</span>`;
  } finally { starting = false; renderModels(); refresh(); }
}
function readyStatus() {
  const gpu = modelEP[PRESETS[S.preset].model] === 'GPU';
  setStatus('ready', 'Hazır · ' + (gpu ? 'Ekran kartı' : 'İşlemci'));
  if (!$('modelHint').querySelector('.err')) $('modelHint').textContent = gpu ? 'Yapay zekâ telefonun ekran kartında çalışıyor.' :
    'Ekran kartı kullanılamadı; işlemcide çalışıyor (yavaş). Safari\'yi güncelle (Ayarlar → Genel → Yazılım Güncelleme).';
}
function renderModels() {
  const box = $('modelList'); box.innerHTML = '';
  const need = new Set(needFor(S.preset));
  for (const [k, m] of Object.entries(MODELS)) {
    const s = modelState[k] || '…', d = document.createElement('div');
    d.className = 'model' + (need.has(k) ? '' : ' dim');
    const txt = s === 'loaded' ? 'hazır · ' + (modelEP[k] === 'GPU' ? 'GPU' : 'CPU') : s === 'cached' ? 'telefonda' : s === 'missing' ? fmtMB(m.size) : s;
    d.innerHTML = `<span>${m.label}</span><span class="st ${s === 'loaded' ? 'ok' : ''}">${txt}</span>`;
    box.appendChild(d);
  }
  const dl = [...need].filter(k => modelState[k] === 'missing').reduce((a, k) => a + MODELS[k].size, 0);
  $('nModels').classList.toggle('off', !engine);
  $('loadModels').hidden = !!engine; $('loadModels').disabled = starting;
  $('loadModels').textContent = starting ? 'Hazırlanıyor…' : dl ? `Modelleri indir ve başlat (${fmtMB(dl)})` : 'Başlat';
}
async function refreshModelState() {
  for (const k of Object.keys(MODELS)) if (modelState[k] !== 'loaded') modelState[k] = (await cacheHas(MODELS[k].file)) ? 'cached' : 'missing';
  renderModels();
}
$('loadModels').onclick = startEngine;

/* ================= tanılama ================= */
// Ağır adımlardan önce bir iz bırakılır. iPhone sayfayı bellek yüzünden kapatırsa, bir sonraki açılışta
// hangi adımda kapandığı ekranda görünür ve o adım bir dahaki sefere daha hafif yoldan yapılır.
function crumb(step) { store('step', step); }
const STEP_TR = { gpu: 'ekran kartında açılırken', cpu: 'işlemcide açılırken', dl: 'indirilirken' };
function lastCrash() {
  const step = store('step'); if (!step) return null;
  crumb('');
  const [key, phase] = step.split(':'), n = +(store('crash-' + step) || 0) + 1; store('crash-' + step, String(n));
  if (phase === 'gpu') store('nogpu-' + key, '1');
  return { key, phase, n, text: `Önceki açılışta sayfa “${MODELS[key]?.label || key}” ${STEP_TR[phase] || ''} kapandı (telefonun belleği yetmedi).` +
    (phase === 'gpu' ? ' Bu model bundan sonra işlemcide açılacak.' : phase === 'cpu' ? ' Daha az model kullanan “Hızlı” profili seçildi.' : '') };
}
function showError(msg) { const h = $('modelHint'); h.innerHTML = `<span class="err">${String(msg).replace(/</g, '&lt;').slice(0, 300)}</span>`; }
addEventListener('error', e => showError('Hata: ' + (e.message || e.error)));
addEventListener('unhandledrejection', e => showError('Hata: ' + (e.reason?.message || e.reason)));

/* ================= durum ================= */
const S = { preset: PRESETS[store('preset')] ? store('preset') : 'high', identities: [{ photos: [], emb: null }], target: null, busy: false, running: false, cancel: false, result: null };
function setStatus(s, t) { $('status').dataset.s = s; $('statusText').textContent = t; }
function setBar(which, p) { const el = which === 'model' ? $('modelProg') : $('jobProg'); el.hidden = false; (which === 'model' ? $('modelBar') : $('jobBar')).style.width = (p * 100).toFixed(1) + '%'; }

/* ================= görüntü yardımcıları ================= */
const work = document.createElement('canvas'), wctx = work.getContext('2d', { willReadFrequently: true });
const MAX_AREA = 16e6; // iOS tuval sınırı
async function decodeImage(file, maxSide = 4096) {
  let src, w, h;
  try { src = await createImageBitmap(file, { imageOrientation: 'from-image' }); w = src.width; h = src.height; }
  catch {
    const url = URL.createObjectURL(file); src = new Image(); src.src = url;
    try { await src.decode(); } catch { URL.revokeObjectURL(url); throw new Error('Bu dosya açılamadı'); }
    w = src.naturalWidth; h = src.naturalHeight; setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const sc = Math.min(1, maxSide / Math.max(w, h), Math.sqrt(MAX_AREA / (w * h)));
  work.width = Math.round(w * sc); work.height = Math.round(h * sc);
  wctx.drawImage(src, 0, 0, work.width, work.height); src.close?.();
  return wctx.getImageData(0, 0, work.width, work.height);
}
function thumbOf(img, box, size = 128) {
  work.width = img.width; work.height = img.height; wctx.putImageData(img, 0, 0);
  const c = document.createElement('canvas'); c.width = c.height = size;
  const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2, s = Math.max(box[2] - box[0], box[3] - box[1]) * 1.4;
  c.getContext('2d').drawImage(work, cx - s / 2, cy - s / 2, s, s, 0, 0, size, size);
  return c.toDataURL('image/jpeg', .85);
}
const area = f => (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]);
const iou = (a, b) => { const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i); };
function watermark(ctx, W, H) {
  const h = Math.max(20, Math.round(H * 0.034)), m = Math.round(H * 0.022);
  ctx.save(); ctx.font = `600 ${h * 0.5}px -apple-system, "Segoe UI", sans-serif`;
  const label = 'ile düzenlendi', tw = ctx.measureText(label).width, w = tw + h * 1.75;
  const x = W - m - w, y = H - m - h;
  ctx.fillStyle = 'rgba(10,12,14,.55)'; ctx.beginPath(); ctx.roundRect(x, y, w, h, h / 2); ctx.fill();
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#f0b44c'; ctx.font = `700 ${h * 0.56}px ui-monospace, Menlo, monospace`; ctx.fillText('AI', x + h * 0.45, y + h / 2 + 1);
  ctx.fillStyle = '#e9e6df'; ctx.font = `600 ${h * 0.5}px -apple-system, "Segoe UI", sans-serif`; ctx.fillText(label, x + h * 1.3, y + h / 2 + 1);
  ctx.restore();
}

/* ================= kaynak kimlikler ================= */
function renderIdentities(msg = '', pending = {}) {
  const box = $('identities'); box.innerHTML = '';
  S.identities.forEach((id, i) => {
    const el = $('tplIdentity').content.firstElementChild.cloneNode(true);
    el.querySelector('b').textContent = L[i];
    const cons = id.photos.length > 1 ? id.photos.reduce((s, p) => s + cosine(p.emb, id.emb), 0) / id.photos.length : null;
    el.querySelector('span').textContent = id.photos.length ? `${id.photos.length} fotoğraf` + (cons != null ? ` · tutarlılık %${Math.round(cons * 100)}` : '') : 'Fotoğraf ekle';
    const rm = el.querySelector('[data-rm]');
    if (S.identities.length < 2) rm.remove();
    else rm.onclick = () => { S.identities.splice(i, 1); if (S.target) for (const p of S.target.persons.values()) { if (p.assign === i) p.assign = 0; else if (p.assign > i) p.assign--; } renderIdentities(); renderPersons(); refresh(); };
    const ph = el.querySelector('.photos');
    id.photos.forEach((p, j) => { const d = document.createElement('div'); d.className = 'ph';
      d.innerHTML = `<img alt="" src="${p.thumb}"><button type="button" aria-label="Kaldır">×</button>`;
      d.querySelector('button').onclick = () => { id.photos.splice(j, 1); updateIdentity(id); renderIdentities(); refresh(); };
      ph.appendChild(d); });
    for (let k = 0; k < (pending[i] || 0); k++) { const d = document.createElement('div'); d.className = 'ph loading'; ph.appendChild(d); }
    const add = document.createElement('label'); add.className = 'ph add'; add.setAttribute('aria-label', `Kişi ${L[i]} için fotoğraf ekle`);
    add.innerHTML = '+<input type="file" accept="image/*" multiple>';
    add.querySelector('input').onchange = e => { const fs = [...e.target.files]; e.target.value = ''; addPhotos(i, fs); };
    ph.appendChild(add); box.appendChild(el);
  });
  $('sourceHint').innerHTML = msg ? `<span class="err">${msg}</span>` : 'Aynı kişiden 2–5 net fotoğraf (önden, iyi ışık) sonucu belirgin şekilde iyileştirir.';
}
function updateIdentity(id) { id.emb = id.photos.length ? meanEmbedding(id.photos.map(p => p.emb)) : null; id.source = null; }
async function addPhotos(i, files) {
  if (!files.length) return;
  if (!engine) { renderIdentities('Önce “Modelleri indir ve başlat” düğmesine bas.'); return; }
  const id = S.identities[i]; let miss = 0, bad = 0; const errs = [];
  renderIdentities('', { [i]: files.length }); setStatus('busy', 'Yüz analiz ediliyor');
  for (const f of files) {
    try {
      const img = await decodeImage(f, 2048), faces = await engine.detectRobust(img, { score: 0.4 });
      if (!faces.length) { miss++; continue; }
      const face = faces.sort((a, b) => area(b) - area(a))[0];
      id.photos.push({ emb: await engine.embed(img, face.kps), thumb: thumbOf(img, face.box) });
    } catch (e) { console.error(e); if (/açılamadı/.test(e.message)) bad++; else errs.push(e.message || String(e)); }
  }
  updateIdentity(id); readyStatus();
  renderIdentities([miss ? `${miss} fotoğrafta yüz bulunamadı; yüzün net göründüğü bir fotoğraf dene.` : '', bad ? `${bad} dosya açılamadı.` : '', errs.length ? 'Hata: ' + errs[0].slice(0, 140) : ''].filter(Boolean).join(' '));
  refresh();
}
$('addIdentity').onclick = () => { S.identities.push({ photos: [], emb: null }); renderIdentities(); renderPersons(); };

/* ================= hedef ================= */
function resolvePerson(t, emb, img, face, discovered, thr = 0.4) {
  let best = null, bs = -1;
  for (const p of t.persons.values()) { const s = cosine(p.emb, emb); if (s > bs) { bs = s; best = p; } }
  if (best && bs >= thr) {
    if (best.n < 40) { const n = best.n + 1; const e = Float32Array.from(best.emb, (v, i) => v + (emb[i] - v) / n); const l = Math.hypot(...e); best.emb = e.map(v => v / l); best.n = n; }
    if (discovered) best.seen++; return best;
  }
  const p = { id: t.nextPerson++, emb, n: 1, seen: 1, thumb: discovered ? thumbOf(img, face.box) : null, assign: discovered ? 0 : null, discovered };
  t.persons.set(p.id, p); if (discovered) renderPersons(); return p;
}
const view = $('view'), vctx = view.getContext('2d');
function show(img) { if (view.width !== img.width || view.height !== img.height) { view.width = img.width; view.height = img.height; } vctx.putImageData(img, 0, 0); $('stage').hidden = false; }
const isVideo = f => (f.type || '').startsWith('video/') || /\.(mov|mp4|m4v|webm|3gp|mkv)$/i.test(f.name || '');
const vel = document.createElement('video'); vel.playsInline = true; vel.muted = true; vel.preload = 'auto';

$('targetInput').onchange = async e => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  if (!engine) { $('targetMeta').innerHTML = '<span class="err">Önce modelleri başlat.</span>'; return; }
  S.busy = true; S.target = null; S.result = null; $('secResult').hidden = true; $('persons').hidden = true; $('compareBtn').hidden = true; refresh();
  $('targetMeta').textContent = 'Açılıyor…';
  try {
    const t = { file: f, name: f.name || 'dosya', persons: new Map(), nextPerson: 1 };
    if (isVideo(f)) await openVideo(t); else await openImage(t);
    S.target = t;
    if (!t.persons.size) $('targetMeta').textContent += ' · yüz bulunamadı';
  } catch (err) { console.error(err); $('targetMeta').innerHTML = `<span class="err">${err.message || err}</span>`; }
  finally { S.busy = false; readyStatus(); renderPersons(); refresh(); }
};
async function openImage(t) {
  setStatus('busy', 'Kişiler bulunuyor');
  t.kind = 'image'; t.img = await decodeImage(t.file);
  show(t.img);
  for (const fc of await engine.detectRobust(t.img, { score: 0.5 })) resolvePerson(t, await engine.embed(t.img, fc.kps), t.img, fc, true);
  $('targetMeta').textContent = `Fotoğraf · ${t.img.width}×${t.img.height} · ${t.persons.size} kişi`;
}
async function openVideo(t) {
  t.kind = 'video'; setStatus('busy', 'Video okunuyor');
  t.mp4 = await demux(t.file).catch(e => { console.warn('demux', e); return null; });
  t.useDecoder = !!t.mp4 && await decoderSupported(t.mp4);
  if (t.url) URL.revokeObjectURL(t.url);
  t.url = URL.createObjectURL(t.file); vel.src = t.url;
  await new Promise((res, rej) => { vel.onloadeddata = res; vel.onerror = () => rej(new Error('Bu video açılamadı')); setTimeout(res, 15000); });
  const vw = vel.videoWidth || t.mp4?.width, vh = vel.videoHeight || t.mp4?.height, dur = (Number.isFinite(vel.duration) && vel.duration) || t.mp4?.duration || 0;
  if (!vw) throw new Error('Bu video açılamadı');
  t.info = { width: vw, height: vh, duration: dur, fps: t.mp4?.fps || 30 };
  // kişileri bulmak için hızlı tarama
  const N = Math.max(6, Math.min(24, Math.round(dur * 1.2)));
  const sw = Math.round(vw * Math.min(1, 720 / Math.min(vw, vh))), sh = Math.round(vh * Math.min(1, 720 / Math.min(vw, vh)));
  work.width = sw; work.height = sh;
  for (let i = 0; i < N; i++) {
    setStatus('busy', `Kişiler taranıyor %${Math.round(i / N * 100)}`);
    await seek(dur * (i + 0.5) / N);
    wctx.drawImage(vel, 0, 0, sw, sh); const img = wctx.getImageData(0, 0, sw, sh);
    if (i === 0) show(img);
    for (const fc of await engine.detect(img, { score: 0.55 })) { resolvePerson(t, await engine.embed(img, fc.kps), img, fc, true); work.width = sw; work.height = sh; }
  }
  $('targetMeta').textContent = `Video · ${vw}×${vh} · ${Math.floor(dur / 60)}:${String(Math.round(dur % 60)).padStart(2, '0')} · ${Math.round(t.info.fps)} fps · ${t.persons.size} kişi${t.mp4?.audio ? ' · sesli' : ''}`;
}
function seek(tm) { return new Promise(r => { const d = () => setTimeout(r, 60); vel.addEventListener('seeked', d, { once: true }); vel.currentTime = Math.max(0, tm); setTimeout(r, 3000); }); }
function renderPersons() {
  const box = $('persons'), t = S.target; if (!t) { box.hidden = true; return; }
  const list = [...t.persons.values()].filter(p => p.discovered);
  box.hidden = !list.length; box.innerHTML = '';
  for (const p of list) {
    const row = document.createElement('div'); row.className = 'person';
    const a = p.assign ?? 0;
    row.innerHTML = `<img alt="" src="${p.thumb}"><div><small>KİŞİ ${p.id} · ${p.seen} kez</small><div class="chips">${
      S.identities.map((_, i) => `<button class="chip" type="button" data-v="${i}" aria-pressed="${a === i}">${L[i]} yüzü</button>`).join('')
    }<button class="chip" type="button" data-v="-1" aria-pressed="${a === -1}">Değiştirme</button></div></div>`;
    row.querySelectorAll('.chip').forEach(c => c.onclick = () => { p.assign = +c.dataset.v; row.querySelectorAll('.chip').forEach(x => x.setAttribute('aria-pressed', x === c)); refresh(); });
    box.appendChild(row);
  }
}

/* ================= kare işleme ================= */
class OneEuro { constructor(n) { this.x = new Float64Array(n); this.dx = new Float64Array(n); this.ok = false; }
  step(v, dt, minCut, beta) {
    if (!this.ok || !(dt > 0)) { this.x.set(v); this.dx.fill(0); this.ok = true; return Array.from(this.x); }
    const al = c => { const r = 2 * Math.PI * c * dt; return r / (r + 1); }, ad = al(1);
    for (let i = 0; i < v.length; i++) { const d = (v[i] - this.x[i]) / dt; this.dx[i] += ad * (d - this.dx[i]); this.x[i] += al(minCut + beta * Math.abs(this.dx[i])) * (v[i] - this.x[i]); }
    return Array.from(this.x); } }
async function processFrame(src, ctx) {
  const p = PRESETS[S.preset], t = S.target, P = ctx.prof, now = () => performance.now(); let t0 = now();
  const faces = ctx.tracks ? await engine.detect(src, { score: 0.5 }) : await engine.detectRobust(src, { score: 0.5 });
  P.det += now() - t0; t0 = now();
  const out = new ImageData(new Uint8ClampedArray(src.data), src.width, src.height);
  const jobs = [], next = [];
  for (const f of faces) {
    let tr = null, person;
    if (ctx.tracks) { let bi = 0.3; for (const x of ctx.tracks) { const v = iou(x.box, f.box); if (v > bi && !x.used) { bi = v; tr = x; } } if (tr) tr.used = true; }
    if (tr && tr.age < 24 && t.persons.has(tr.pid)) { person = t.persons.get(tr.pid); tr.age++; }
    else { person = resolvePerson(t, await engine.embed(src, f.kps), src, f, false); tr = { filter: tr?.filter || new OneEuro(10), age: 0 }; }
    let kps = f.kps;
    if (ctx.tracks) { const sm = tr.filter.step(f.kps.flat(), ctx.dt, 4.4, 0.27); kps = [0, 1, 2, 3, 4].map(i => [sm[2 * i], sm[2 * i + 1]]); next.push({ box: f.box, pid: person.id, age: tr.age, filter: tr.filter }); }
    const a = person.assign ?? (person.discovered ? 0 : -1), ident = a >= 0 ? S.identities[a] : null;
    if (ident?.emb) {
      if (!ident.source || ident.sourceType !== engine.s.swapperType) { ident.source = engine.sourceFor(ident.emb); ident.sourceType = engine.s.swapperType; }
      jobs.push({ kps, source: ident.source });
    }
  }
  if (ctx.tracks) ctx.tracks = next;
  P.id += now() - t0; t0 = now();
  for (const j of jobs) await engine.swap(out, j.kps, j.source, { size: p.size, blur: 0.3, occlusion: p.occlusion, region: p.region });
  P.swap += now() - t0; t0 = now();
  if (p.enhance) for (const j of jobs) await engine.enhance(out, j.kps, 0.8, { occlusion: p.occlusion });
  P.enh += now() - t0; P.n++;
  return { out, faces: faces.length, swapped: jobs.length };
}

/* ================= işlem ================= */
let wakeLock = null;
async function keepAwake(on) {
  try { if (on && !wakeLock && navigator.wakeLock) { wakeLock = await navigator.wakeLock.request('screen'); wakeLock.addEventListener('release', () => { wakeLock = null; }); }
    else if (!on && wakeLock) { await wakeLock.release(); wakeLock = null; } } catch {}
}
document.addEventListener('visibilitychange', () => { if (!document.hidden && S.running) keepAwake(true); });
const fmtDur = s => s > 90 ? `${Math.ceil(s / 60)} dk` : `${Math.ceil(s)} sn`;

$('startBtn').onclick = async () => {
  S.running = true; S.cancel = false; S.result = null; refresh();
  $('secResult').hidden = false; $('resultImg').hidden = true; $('resultVid').hidden = true; $('resultActions').hidden = true; $('cancelBtn').hidden = false;
  setBar('job', 0); $('jobMsg').textContent = 'Hazırlanıyor…';
  $('secResult').scrollIntoView({ behavior: 'smooth', block: 'start' });
  await keepAwake(true);
  const start = performance.now();
  try {
    await prepare(S.preset);
    if (S.target.kind === 'image') await runImage(start); else await runVideo(start);
  } catch (e) { console.error(e); $('jobMsg').innerHTML = `<span class="err">${e.message === 'Durduruldu' ? 'Durduruldu.' : (e.message || e)}</span>`; }
  finally { S.running = false; $('cancelBtn').hidden = true; await keepAwake(false); readyStatus(); refresh(); }
};
$('cancelBtn').onclick = () => { S.cancel = true; };

async function runImage(start) {
  setStatus('busy', 'İşleniyor'); $('jobMsg').textContent = 'Yüz değiştiriliyor…'; setBar('job', 0.3);
  const ctx = { prof: { det: 0, id: 0, swap: 0, enh: 0, n: 0 } };
  const r = await processFrame(S.target.img, ctx);
  const c = document.createElement('canvas'); c.width = r.out.width; c.height = r.out.height; const x = c.getContext('2d');
  x.putImageData(r.out, 0, 0); if ($('wm').checked) watermark(x, c.width, c.height);
  const blob = await new Promise(res => c.toBlob(res, 'image/jpeg', 0.95));
  S.after = x.getImageData(0, 0, c.width, c.height); show(S.after);
  setResult(blob, 'image', base(S.target.name) + '.jpg');
  setBar('job', 1); $('jobMsg').textContent = `${r.swapped}/${r.faces} yüz değiştirildi · ${fmtDur((performance.now() - start) / 1000)}`;
  $('compareBtn').hidden = false; $('compareBtn').setAttribute('aria-pressed', 'false');
}
async function runVideo(start) {
  const t = S.target, p = PRESETS[S.preset];
  const sc = Math.min(1, p.short / Math.min(t.info.width, t.info.height));
  const W = Math.round(t.info.width * sc / 2) * 2, H = Math.round(t.info.height * sc / 2) * 2;
  const fps = Math.max(1, Math.min(120, t.info.fps || 30)), total = t.mp4?.count || Math.round(t.info.duration * fps);
  const writer = await createWriter(W, H, fps, t.mp4?.audio || null, p.bpp);
  const frameCanvas = document.createElement('canvas'); frameCanvas.width = W; frameCanvas.height = H;
  const fctx = frameCanvas.getContext('2d', { willReadFrequently: true });
  const ctx = { prof: { det: 0, id: 0, swap: 0, enh: 0, n: 0 }, tracks: [], dt: 1 / fps };
  let i = 0, first = null, prevTs = null; const wm = $('wm').checked;
  const handle = async (source, tsUs) => {
    if (S.cancel) throw new Error('Durduruldu');
    if (first === null) first = tsUs;
    const ts = Math.max(0, tsUs - first); if (prevTs !== null && ts <= prevTs) return;
    ctx.dt = prevTs === null ? 1 / fps : (ts - prevTs) / 1e6; prevTs = ts;
    const rot = t.useDecoder ? t.mp4.rotation : 0;
    fctx.save(); fctx.translate(W / 2, H / 2); fctx.rotate(rot * Math.PI / 180);
    const [dw, dh] = rot % 180 ? [H, W] : [W, H]; fctx.drawImage(source, -dw / 2, -dh / 2, dw, dh); fctx.restore();
    const r = await processFrame(fctx.getImageData(0, 0, W, H), ctx);
    fctx.putImageData(r.out, 0, 0); if (wm) watermark(fctx, W, H);
    await writer.add(frameCanvas, ts, Math.round(1e6 / fps));
    i++;
    if (i % 4 === 1) { show(r.out); }
    const pr = Math.min(1, i / Math.max(1, total)), el = (performance.now() - start) / 1000;
    setBar('job', pr); setStatus('busy', `İşleniyor %${Math.round(pr * 100)}`);
    const ms = v => Math.round(v / Math.max(1, ctx.prof.n));
    $('jobMsg').textContent = `Kare ${i}/${total} · ${(el / i).toFixed(2)} sn/kare · kalan ~${fmtDur(Math.max(0, el / pr - el))} · takas ${ms(ctx.prof.swap)} ms`;
  };
  try {
    if (t.useDecoder) await decodeAll(t.mp4, f => handle(f, f.timestamp), () => S.cancel);
    else await playFrames(vel, (v, ts) => handle(v, ts), () => S.cancel);
  } catch (e) { writer.abort(); throw e; }
  $('jobMsg').textContent = 'Video kaydediliyor…';
  const blob = await writer.finish();
  setResult(blob, 'video', base(t.name) + '.mp4');
  setBar('job', 1); $('jobMsg').textContent = `Bitti · ${i} kare · ${W}×${H} · ${(blob.size / 1048576).toFixed(1)} MB · ${fmtDur((performance.now() - start) / 1000)}${t.mp4?.audio ? '' : ' · ses yok'}`;
}
const base = n => n.replace(/\.[^.]+$/, '').replace(/[^\w\-]+/g, '_').slice(0, 40) + '-yuz-takas';
function setResult(blob, kind, name) {
  if (S.result?.url) URL.revokeObjectURL(S.result.url);
  const url = URL.createObjectURL(blob); S.result = { blob, kind, name, url };
  const el = kind === 'image' ? $('resultImg') : $('resultVid'); el.src = url; el.hidden = false;
  $('dlLink').href = url; $('dlLink').download = name; $('resultActions').hidden = false;
}
// iPhone paylaşım menüsü: "Görüntüyü Kaydet" / "Videoyu Kaydet" ile Fotoğraflar'a gider
$('saveBtn').onclick = async () => {
  const r = S.result; if (!r) return;
  const file = new File([r.blob], r.name, { type: r.kind === 'video' ? 'video/mp4' : 'image/jpeg' });
  try {
    if (navigator.canShare?.({ files: [file] })) await navigator.share({ files: [file] });
    else $('dlLink').click();
  } catch (e) { if (e.name !== 'AbortError') $('jobMsg').textContent = 'Paylaşılamadı: ' + e.message; }
};
$('compareBtn').onclick = () => {
  const on = $('compareBtn').getAttribute('aria-pressed') !== 'true';
  $('compareBtn').setAttribute('aria-pressed', on); $('compareBtn').textContent = on ? 'Orijinal' : 'Önce / sonra';
  show(on ? S.target.img : S.after);
};

/* ================= kalite ================= */
function renderPreset() {
  for (const b of $('preset').children) b.setAttribute('aria-checked', b.dataset.v === S.preset);
  const extra = needFor(S.preset).filter(k => modelState[k] === 'missing').reduce((a, k) => a + MODELS[k].size, 0);
  $('presetHint').textContent = PRESETS[S.preset].hint + (extra ? ` İlk kullanımda +${fmtMB(extra)} iner.` : '');
}
$('preset').onclick = async e => {
  const b = e.target.closest('button'); if (!b || S.running) return;
  S.preset = b.dataset.v; store('preset', S.preset); renderPreset(); renderModels();
  if (engine) { try { S.busy = true; refresh(); await prepare(S.preset); } catch (err) { $('modelHint').innerHTML = `<span class="err">${err.message}</span>`; } finally { S.busy = false; renderPreset(); refresh(); } }
};
$('consent').onchange = refresh;

function refresh() {
  const hasId = S.identities.some(i => i.emb), t = S.target;
  const hasT = !!t && [...t.persons.values()].some(p => p.discovered && (p.assign ?? 0) >= 0);
  $('nSource').classList.toggle('off', !hasId); $('nTarget').classList.toggle('off', !hasT);
  $('targetPick').classList.toggle('disabled', !engine || S.busy || S.running);
  $('addIdentity').disabled = !engine;
  const ok = !!engine && hasId && hasT && $('consent').checked && !S.busy && !S.running;
  $('startBtn').disabled = !ok;
  $('startBtn').textContent = S.running ? 'İşleniyor…' : t?.kind === 'video' ? 'Videoyu işle' : 'Fotoğrafı işle';
  $('dockHint').textContent = !engine ? (starting ? 'Modeller hazırlanıyor…' : 'Önce modelleri başlat.') : !hasId ? 'Kaynak yüz fotoğrafı ekle.' : !t ? 'Hedef fotoğraf veya video seç.' :
    !hasT ? 'Hedefte değiştirilecek yüz yok.' : !$('consent').checked ? 'Onay kutusunu işaretle.' :
    S.running ? 'Ekranı açık tut ve Safari\'den çıkma; arka planda işlem durur.' : t.kind === 'video' ? 'İşlem sırasında ekran açık kalmalı.' : 'Hazır.';
}

/* ================= açılış ================= */
(async () => {
  renderIdentities(); renderPreset(); refresh();
  gpuOK = await detectGPU();
  const crash = lastCrash();
  if (crash?.phase === 'cpu' && S.preset !== 'fast') { S.preset = 'fast'; store('preset', 'fast'); }
  await refreshModelState(); renderPreset();
  if (crash) { setStatus('error', 'Sayfa kapanmıştı'); showError(crash.text + ' “Başlat”a basarak tekrar dene.'); }
  else if (needFor(S.preset).every(k => modelState[k] === 'cached')) startEngine();
  else { setStatus('idle', 'Modeller indirilmedi'); if (!gpuOK) $('modelHint').textContent = 'Bu tarayıcıda ekran kartı (WebGPU) yok; işlem çok yavaş olur. iPhone\'da Safari\'yi güncel tut.'; }
})();
