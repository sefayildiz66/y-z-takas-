// Yüz Takas Stüdyosu — çekirdek görüntü işleme hattı
// SCRFD (tespit + 5 nokta) → ArcFace (kimlik) → InSwapper 128 / HyperSwap 256 (takas) → XSeg / BiSeNet (maskeler) → GFPGAN 1.4 (iyileştirme)
// Görüntüler {data: Uint8ClampedArray RGBA, width, height}.

/* ---------- hizalama şablonları (FaceFusion / InsightFace ile aynı) ---------- */
export const TEMPLATES = {
  arcface_112_v2: [[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.500225, 0.64050536], [0.37097589, 0.82469196], [0.63151696, 0.82325089]],
  arcface_128: [[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219], [0.38710391, 0.72160547], [0.61507734, 0.72034453]],
  ffhq_512: [[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904], [0.39308822, 0.72541100], [0.61150205, 0.72490465]],
};

/* ---------- geometri ---------- */
// 2B benzerlik dönüşümü (ölçek + dönme + öteleme), en küçük kareler. 2x3 matris [a,b,c,d,e,f]
export function similarity(src, dst) {
  const n = src.length;
  let msx = 0, msy = 0, mdx = 0, mdy = 0;
  for (let i = 0; i < n; i++) { msx += src[i][0]; msy += src[i][1]; mdx += dst[i][0]; mdy += dst[i][1]; }
  msx /= n; msy /= n; mdx /= n; mdy /= n;
  let a = 0, b = 0, v = 0;
  for (let i = 0; i < n; i++) {
    const xx = src[i][0] - msx, xy = src[i][1] - msy, yx = dst[i][0] - mdx, yy = dst[i][1] - mdy;
    a += xx * yx + xy * yy; b += xx * yy - xy * yx; v += xx * xx + xy * xy;
  }
  const c = a / v, s = b / v;
  return [c, -s, mdx - (c * msx - s * msy), s, c, mdy - (s * msx + c * msy)];
}
export function invert(M) {
  const [a, b, c, d, e, f] = M, det = a * e - b * d;
  const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
  return [ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)];
}
export const apply = (M, x, y) => [M[0] * x + M[1] * y + M[2], M[3] * x + M[4] * y + M[5]];
export function templateMatrix(kps, name, size) {
  return similarity(kps, TEMPLATES[name].map(([x, y]) => [x * size, y * size]));
}

/* ---------- örnekleme ---------- */
function sample(img, x, y, out, o) {
  const W = img.width, H = img.height, d = img.data;
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  let r = 0, g = 0, b = 0;
  for (let k = 0; k < 4; k++) {
    const xi = x0 + (k & 1), yi = y0 + (k >> 1);
    if (xi < 0 || yi < 0 || xi >= W || yi >= H) continue;
    const w = ((k & 1) ? fx : 1 - fx) * ((k >> 1) ? fy : 1 - fy), p = (yi * W + xi) * 4;
    r += d[p] * w; g += d[p + 1] * w; b += d[p + 2] * w;
  }
  out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 255;
}
// M: kaynak → hedef. Sonuç W×H RGBA (OpenCV warpAffine düzeni, sınır = 0)
export function warp(img, M, W, H) {
  const inv = invert(M), out = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    let sx = inv[1] * y + inv[2], sy = inv[4] * y + inv[5];
    for (let x = 0; x < W; x++) { sample(img, sx, sy, out, (y * W + x) * 4); sx += inv[0]; sy += inv[3]; }
  }
  return { data: out, width: W, height: H };
}

/* ---------- maskeler ---------- */
const maskCache = new Map();
// FaceFusion "box" maskesi (Gauss bulanık kenarlı)
export function boxMask(W, H, blur = 0.3, pad = [0, 0, 0, 0]) {
  const key = [W, H, blur, ...pad].join(',');
  if (maskCache.has(key)) return maskCache.get(key);
  const amount = Math.floor(W * 0.5 * blur), area = Math.max(Math.floor(amount / 2), 1);
  const top = Math.max(area, Math.floor(H * pad[0] / 100)), right = Math.max(area, Math.floor(W * pad[1] / 100));
  const bottom = Math.max(area, Math.floor(H * pad[2] / 100)), left = Math.max(area, Math.floor(W * pad[3] / 100));
  let m = new Float32Array(W * H);
  for (let y = top; y < H - bottom; y++) for (let x = left; x < W - right; x++) m[y * W + x] = 1;
  if (amount > 0) m = gaussBlur(m, W, H, amount * 0.25);
  maskCache.set(key, m);
  return m;
}
function gaussBlur(src, W, H, sigma) {
  const r = Math.max(1, Math.ceil(sigma * 3)), k = new Float32Array(2 * r + 1);
  let s = 0; for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); s += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const refl = (i, n) => i < 0 ? Math.min(n - 1, -i - 1) : i >= n ? Math.max(0, 2 * n - i - 1) : i;
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let a = 0; for (let i = -r; i <= r; i++) a += src[y * W + refl(x + i, W)] * k[i + r]; tmp[y * W + x] = a; }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let a = 0; for (let i = -r; i <= r; i++) a += tmp[refl(y + i, H) * W + x] * k[i + r]; out[y * W + x] = a; }
  return out;
}
function resizeMask(m, w, h, W, H) { // çift doğrusal
  if (w === W && h === H) return Float32Array.from(m);
  const out = new Float32Array(W * H), sx = w / W, sy = h / H, c = v => Math.min(1, Math.max(0, v));
  for (let y = 0; y < H; y++) {
    const fy = Math.min(h - 1, Math.max(0, (y + 0.5) * sy - 0.5)), y0 = Math.floor(fy), y1 = Math.min(h - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < W; x++) {
      const fx = Math.min(w - 1, Math.max(0, (x + 0.5) * sx - 0.5)), x0 = Math.floor(fx), x1 = Math.min(w - 1, x0 + 1), tx = fx - x0;
      const a = c(m[y0 * w + x0]), b = c(m[y0 * w + x1]), cc = c(m[y1 * w + x0]), d = c(m[y1 * w + x1]);
      out[y * W + x] = (a * (1 - tx) + b * tx) * (1 - ty) + (cc * (1 - tx) + d * tx) * ty;
    }
  }
  return out;
}
function finishMask(m, W, H) { // FaceFusion: GaussianBlur(σ=5), sonra (clip(0.5,1) - 0.5) * 2
  const b = gaussBlur(m, W, H, 5);
  for (let i = 0; i < b.length; i++) b[i] = (Math.min(1, Math.max(0.5, b[i])) - 0.5) * 2;
  return b;
}

// Kırpılmış (hizalanmış) yüzü kareye geri yapıştırır. M: kare → kırpım
export function pasteBack(frame, crop, M, mask, blend = 1) {
  const inv = invert(M), Wc = crop.width, Hc = crop.height, W = frame.width, H = frame.height, fd = frame.data, cd = crop.data;
  let x0 = W, y0 = H, x1 = 0, y1 = 0;
  for (const [cx, cy] of [[0, 0], [Wc, 0], [0, Hc], [Wc, Hc]]) { const [x, y] = apply(inv, cx, cy); x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0)); x1 = Math.min(W - 1, Math.ceil(x1)); y1 = Math.min(H - 1, Math.ceil(y1));
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const u = M[0] * x + M[1] * y + M[2], v = M[3] * x + M[4] * y + M[5];
    if (u < 0 || v < 0 || u > Wc - 1 || v > Hc - 1) continue;
    const ui = Math.floor(u), vi = Math.floor(v), fu = u - ui, fv = v - vi;
    const u1 = Math.min(ui + 1, Wc - 1), v1 = Math.min(vi + 1, Hc - 1);
    const w00 = (1 - fu) * (1 - fv), w10 = fu * (1 - fv), w01 = (1 - fu) * fv, w11 = fu * fv;
    const i00 = vi * Wc + ui, i10 = vi * Wc + u1, i01 = v1 * Wc + ui, i11 = v1 * Wc + u1;
    const m = (mask[i00] * w00 + mask[i10] * w10 + mask[i01] * w01 + mask[i11] * w11) * blend;
    if (m <= 0.001) continue;
    const p = (y * W + x) * 4;
    for (let c = 0; c < 3; c++) {
      const cv = cd[i00 * 4 + c] * w00 + cd[i10 * 4 + c] * w10 + cd[i01 * 4 + c] * w01 + cd[i11 * 4 + c] * w11;
      fd[p + c] = fd[p + c] * (1 - m) + cv * m;
    }
  }
}

/* ---------- tensörler ---------- */
function toCHW(img, mean, std) {
  const n = img.width * img.height, out = new Float32Array(3 * n), d = img.data;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[c * n + i] = (d[i * 4 + c] - mean) / std;
  return out;
}
function fromCHW(arr, W, H, fn) {
  const n = W * H, out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) { for (let c = 0; c < 3; c++) out[i * 4 + c] = fn(arr[c * n + i]); out[i * 4 + 3] = 255; }
  return { data: out, width: W, height: H };
}
const l2 = v => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return Float32Array.from(v, x => x / s); };
export const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export function meanEmbedding(list) {
  const m = new Float32Array(512); for (const e of list) for (let i = 0; i < 512; i++) m[i] += e[i]; return l2(m);
}

/* ---------- ekran kartı doğrulaması (deterministik girdiler) ---------- */
export function probeInput(n, k = 0) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = 0.5 + 0.3 * Math.sin(i * 0.013 + k) * Math.cos(i * 0.0007 + 2 * k);
  return a;
}
export function probeLatent() { const a = new Float32Array(512); for (let i = 0; i < 512; i++) a[i] = Math.sin(i * 1.7 + 0.3); return l2(a); }
export const PROBE_STEP = 97;
export function probeFeeds(key, Tensor) {
  const T = (a, sh) => new Tensor('float32', a, sh);
  switch (key) {
    case 'recognizer': return { input: T(probeInput(3 * 112 * 112).map(v => v * 2 - 1), [1, 3, 112, 112]) };
    case 'swapper': return { target: T(probeInput(3 * 128 * 128), [1, 3, 128, 128]), source: T(probeLatent(), [1, 512]) };
    case 'hyperswap': return { target: T(probeInput(3 * 256 * 256).map(v => v * 2 - 1), [1, 3, 256, 256]), source: T(probeLatent(), [1, 512]) };
    case 'enhancer': return { input: T(probeInput(3 * 512 * 512).map(v => v * 2 - 1), [1, 3, 512, 512]) };
    case 'occluder': return { input: T(probeInput(256 * 256 * 3), [1, 256, 256, 3]) };
    case 'parser': return { input: T(probeInput(3 * 512 * 512).map(v => (v - 0.45) / 0.226), [1, 3, 512, 512]) };
  }
  return null;
}
export const PROBE_TOL = { recognizer: 0.05, swapper: 0.02, hyperswap: 0.03, enhancer: 0.04, occluder: 0.03, parser: 0.25 };

/* ---------- motor ---------- */
export class FaceEngine {
  constructor(ort, sessions, emap) { this.ort = ort; this.s = sessions; this.emap = emap; }

  // SCRFD 2.5G: kutular, güven, 5 nokta
  async detect(img, { size = 640, score = 0.5, nms = 0.4 } = {}) {
    const sc = Math.min(size / img.width, size / img.height);
    const det = warp(img, [sc, 0, 0, 0, sc, 0], size, size);
    const input = new this.ort.Tensor('float32', toCHW(det, 127.5, 128), [1, 3, size, size]);
    const out = await this.s.detector.run({ [this.s.detector.inputNames[0]]: input });
    const t = this.s.detector.outputNames.map(n => out[n]);
    const faces = [];
    for (const stride of [8, 16, 32]) {
      const fw = Math.ceil(size / stride), cnt = fw * fw * 2;
      const scores = t.find(x => x.dims[1] === 1 && x.dims[0] === cnt).data;
      const boxes = t.find(x => x.dims[1] === 4 && x.dims[0] === cnt).data;
      const kpss = t.find(x => x.dims[1] === 10 && x.dims[0] === cnt).data;
      for (let i = 0; i < cnt; i++) {
        if (scores[i] < score) continue;
        const a = i >> 1, cx = (a % fw) * stride, cy = Math.floor(a / fw) * stride;
        const box = [cx - boxes[i * 4] * stride, cy - boxes[i * 4 + 1] * stride, cx + boxes[i * 4 + 2] * stride, cy + boxes[i * 4 + 3] * stride].map(v => v / sc);
        const kps = []; for (let k = 0; k < 5; k++) kps.push([(cx + kpss[i * 10 + 2 * k] * stride) / sc, (cy + kpss[i * 10 + 2 * k + 1] * stride) / sc]);
        faces.push({ box, score: scores[i], kps });
      }
    }
    faces.sort((a, b) => b.score - a.score);
    const keep = [];
    const iou = (a, b) => { const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
      const i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i); };
    for (const f of faces) if (keep.every(k => iou(k.box, f.box) < nms)) keep.push(f);
    return keep;
  }
  // Yakın çekim (yüz kareyi dolduruyorsa): bulunamazsa kenar boşluğu ekleyip tekrar dener
  async detectRobust(img, opts = {}) {
    let faces = await this.detect(img, opts);
    if (faces.length) return faces;
    for (const f of [0.5, 1.0]) {
      const px = Math.round(Math.max(img.width, img.height) * f / 2), W = img.width + 2 * px, H = img.height + 2 * px;
      const padded = warp(img, [1, 0, px, 0, 1, px], W, H);
      faces = await this.detect(padded, { ...opts, score: Math.min(opts.score ?? 0.5, 0.45) });
      if (faces.length) return faces.map(x => ({ ...x, box: x.box.map(v => v - px), kps: x.kps.map(([a, b]) => [a - px, b - px]) }));
    }
    return [];
  }

  // ArcFace w600k_r50: 512 boyutlu, normalize kimlik vektörü
  async embed(img, kps) {
    const crop = warp(img, templateMatrix(kps, 'arcface_112_v2', 112), 112, 112);
    const input = new this.ort.Tensor('float32', toCHW(crop, 127.5, 127.5), [1, 3, 112, 112]);
    const out = await this.s.recognizer.run({ [this.s.recognizer.inputNames[0]]: input });
    return l2(out[this.s.recognizer.outputNames[0]].data);
  }
  latent(embedding) { // InsightFace: normed_embedding · emap, ardından normalize
    const e = this.emap, out = new Float32Array(512);
    for (let i = 0; i < 512; i++) { const v = embedding[i]; if (!v) continue; const row = i * 512; for (let j = 0; j < 512; j++) out[j] += v * e[row + j]; }
    return l2(out);
  }
  // InSwapper emap'ten geçmiş vektör ister, HyperSwap normalize ArcFace ister
  sourceFor(embedding) { return this.s.swapperType === 'hyperswap' ? embedding : this.latent(embedding); }

  // XSeg: yüzün önündeki el, saç, gözlük, mikrofon gibi engeller (1 = yüz görünür)
  async occlusionMask(crop) {
    const S = 256, small = crop.width === S ? crop : warp(crop, [S / crop.width, 0, 0, 0, S / crop.height, 0], S, S), n = S * S;
    const x = new Float32Array(3 * n); // NHWC, BGR, 0..1
    for (let i = 0; i < n; i++) { x[i * 3] = small.data[i * 4 + 2] / 255; x[i * 3 + 1] = small.data[i * 4 + 1] / 255; x[i * 3 + 2] = small.data[i * 4] / 255; }
    const out = await this.s.occluder.run({ [this.s.occluder.inputNames[0]]: new this.ort.Tensor('float32', x, [1, S, S, 3]) });
    return finishMask(resizeMask(out[this.s.occluder.outputNames[0]].data, S, S, crop.width, crop.height), crop.width, crop.height);
  }
  // BiSeNet yüz ayrıştırma: yalnızca cilt, kaş, göz, burun, dudak (saç ve arka plan korunur)
  async regionMask(crop, regions = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]) {
    const S = 512, big = warp(crop, [S / crop.width, 0, 0, 0, S / crop.height, 0], S, S), n = S * S;
    const mean = [0.485, 0.456, 0.406], sd = [0.229, 0.224, 0.225], x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) x[c * n + i] = (big.data[i * 4 + c] / 255 - mean[c]) / sd[c];
    const out = await this.s.parser.run({ [this.s.parser.inputNames[0]]: new this.ort.Tensor('float32', x, [1, 3, S, S]) });
    const L = out[this.s.parser.outputNames[0]].data, keep = new Uint8Array(19); for (const r of regions) keep[r] = 1;
    const m = new Float32Array(n);
    for (let i = 0; i < n; i++) { let bi = 0, bv = -Infinity; for (let c = 0; c < 19; c++) { const v = L[c * n + i]; if (v > bv) { bv = v; bi = c; } } m[i] = keep[bi]; }
    return finishMask(resizeMask(m, S, S, crop.width, crop.height), crop.width, crop.height);
  }

  // Yüz takası. size: çıktı netliği (128/256/512); model kendi boyutunun katlarında çalışır (piksel güçlendirme)
  async swap(frame, kps, source, { size = 256, blur = 0.3, pad = [0, 0, 0, 0], occlusion = false, region = false } = {}) {
    const hyper = this.s.swapperType === 'hyperswap', ms = hyper ? 256 : 128;
    const boost = Math.max(1, Math.round(size / ms)), full = ms * boost;
    const M = templateMatrix(kps, 'arcface_128', full);
    const crop = warp(frame, M, full, full);
    const result = new Uint8ClampedArray(full * full * 4), modelMask = hyper ? new Float32Array(full * full) : null;
    const src = new this.ort.Tensor('float32', source, [1, 512]);
    const sub = { data: new Uint8ClampedArray(ms * ms * 4), width: ms, height: ms }, n = ms * ms;
    for (let i = 0; i < boost; i++) for (let j = 0; j < boost; j++) {
      for (let y = 0; y < ms; y++) for (let x = 0; x < ms; x++) {
        const s = ((y * boost + i) * full + (x * boost + j)) * 4, d = (y * ms + x) * 4;
        sub.data[d] = crop.data[s]; sub.data[d + 1] = crop.data[s + 1]; sub.data[d + 2] = crop.data[s + 2];
      }
      const target = new this.ort.Tensor('float32', hyper ? toCHW(sub, 127.5, 127.5) : toCHW(sub, 0, 255), [1, 3, ms, ms]);
      const out = await this.s.swapper.run({ target, source: src });
      const o = out.output ? out.output.data : out[this.s.swapper.outputNames[0]].data, mk = hyper ? out.mask?.data : null;
      for (let y = 0; y < ms; y++) for (let x = 0; x < ms; x++) {
        const k = y * ms + x, q = (y * boost + i) * full + (x * boost + j), dpos = q * 4;
        for (let c = 0; c < 3; c++) result[dpos + c] = hyper ? (o[c * n + k] * 0.5 + 0.5) * 255 : o[c * n + k] * 255;
        result[dpos + 3] = 255;
        if (modelMask) modelMask[q] = mk ? Math.min(1, Math.max(0, mk[k])) : 1;
      }
    }
    const mask = Float32Array.from(boxMask(full, full, blur, pad));
    if (modelMask) for (let i = 0; i < mask.length; i++) mask[i] = Math.min(mask[i], modelMask[i]);
    if (occlusion && this.s.occluder) { const m = await this.occlusionMask(crop); for (let i = 0; i < mask.length; i++) mask[i] = Math.min(mask[i], m[i]); }
    if (region && this.s.parser) { const m = await this.regionMask(crop); for (let i = 0; i < mask.length; i++) mask[i] = Math.min(mask[i], m[i]); }
    pasteBack(frame, { data: result, width: full, height: full }, M, mask, 1);
  }

  // GFPGAN 1.4: yüz detaylarını 512 px'de yeniden üretir
  async enhance(frame, kps, blend = 0.8, { occlusion = false } = {}) {
    const M = templateMatrix(kps, 'ffhq_512', 512);
    const crop = warp(frame, M, 512, 512);
    const input = new this.ort.Tensor('float32', toCHW(crop, 127.5, 127.5), [1, 3, 512, 512]);
    const out = await this.s.enhancer.run({ [this.s.enhancer.inputNames[0]]: input });
    const res = fromCHW(out[this.s.enhancer.outputNames[0]].data, 512, 512, v => (Math.min(1, Math.max(-1, v)) + 1) * 127.5);
    let mask = boxMask(512, 512, 0.3);
    if (occlusion && this.s.occluder) { const m = await this.occlusionMask(crop); mask = Float32Array.from(mask, (v, i) => Math.min(v, m[i])); }
    pasteBack(frame, res, M, mask, blend);
  }
}
