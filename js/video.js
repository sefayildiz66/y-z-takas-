// Video giriş/çıkışı: mp4box ile ayrıştırma, WebCodecs ile kare kare çözme ve kodlama.
// iPhone videoları (MOV, HEVC/H.264) atlamasız okunur; ses yeniden kodlanmadan aynen kopyalanır.

// Dosyayı ayrıştırır: video örnekleri + kodek yapılandırması, varsa AAC ses örnekleri
export async function demux(file) {
  if (typeof MP4Box === 'undefined') return null;
  const mp4 = MP4Box.createFile();
  const vChunks = [], aSamples = []; let info = null, vt = null, at = null, failed = null;
  mp4.onError = e => { failed = e; };
  mp4.onReady = i => {
    info = i; vt = i.videoTracks[0] || null; at = i.audioTracks.find(t => /^mp4a/.test(t.codec)) || null;
    if (vt) mp4.setExtractionOptions(vt.id, 'v', { nbSamples: 300 });
    if (at) mp4.setExtractionOptions(at.id, 'a', { nbSamples: 500 });
    mp4.start();
  };
  mp4.onSamples = (id, user, samples) => {
    for (const s of samples) {
      if (user === 'v') vChunks.push(new EncodedVideoChunk({ type: s.is_sync ? 'key' : 'delta', timestamp: Math.round(1e6 * s.cts / s.timescale), duration: Math.round(1e6 * s.duration / s.timescale), data: s.data }));
      else aSamples.push({ data: new Uint8Array(s.data), ts: Math.round(1e6 * s.cts / s.timescale), dur: Math.round(1e6 * s.duration / s.timescale) });
    }
    if (samples.length) mp4.releaseUsedSamples(id, samples[samples.length - 1].number + 1);
  };
  const STEP = 8 << 20;
  for (let off = 0; off < file.size && !failed; off += STEP) {
    const buf = await file.slice(off, off + STEP).arrayBuffer(); buf.fileStart = off;
    mp4.appendBuffer(buf);
  }
  mp4.flush();
  if (failed || !vt || !vChunks.length) return null;
  const desc = boxBytes(mp4.getTrackById(vt.id), ['avcC', 'hvcC', 'vpcC', 'av1C']);
  const codec = vt.codec.startsWith('vp08') ? 'vp8' : vt.codec;
  const config = { codec, codedWidth: vt.video.width, codedHeight: vt.video.height, ...(desc ? { description: desc } : {}) };
  let audio = null;
  if (at && aSamples.length) {
    // MOV dosyalarında (iPhone) esds kutusu okunamayabilir: AAC-LC yapılandırması örnekleme hızından kurulur
    const asc = audioSpecificConfig(mp4.getTrackById(at.id)) || ascFor(at.audio.sample_rate, at.audio.channel_count);
    if (asc) audio = { samples: aSamples, sampleRate: at.audio.sample_rate, channels: at.audio.channel_count, description: asc };
  }
  const m = vt.matrix || [65536, 0, 0, 0, 65536, 0];
  const rotation = ((Math.round(Math.atan2(m[1], m[0]) * 180 / Math.PI) % 360) + 360) % 360;
  const duration = info.duration / info.timescale || vChunks.length / 30;
  return { config, chunks: vChunks, rotation, count: vChunks.length, duration, fps: vChunks.length / duration, audio,
    width: rotation % 180 ? vt.video.height : vt.video.width, height: rotation % 180 ? vt.video.width : vt.video.height };
}
function boxBytes(trak, names) {
  for (const e of trak.mdia.minf.stbl.stsd.entries) for (const n of names) if (e[n]) {
    const ds = new DataStream(undefined, 0, DataStream.BIG_ENDIAN); e[n].write(ds); return new Uint8Array(ds.buffer, 8);
  }
  return null;
}
function audioSpecificConfig(trak) {
  try {
    const esd = trak.mdia.minf.stbl.stsd.entries[0].esds.esd;
    const dci = esd.descs.find(d => d.tag === 4) || esd.descs[0];
    const dsi = (dci.descs || []).find(d => d.tag === 5) || dci.descs[0];
    return dsi?.data ? new Uint8Array(dsi.data) : null;
  } catch { return null; }
}
function ascFor(rate, ch) {
  const idx = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350].indexOf(rate);
  if (idx < 0 || !ch || ch > 7) return null;
  return new Uint8Array([(2 << 3) | (idx >> 1), ((idx & 1) << 7) | (ch << 3)]); // AAC-LC
}
export async function decoderSupported(src) {
  if (typeof VideoDecoder === 'undefined') return false;
  try { return !!(await VideoDecoder.isConfigSupported(src.config)).supported; } catch { return false; }
}

// Kareleri sırayla verir; consume(frame) her kare için beklenir, kare sonra kapatılır.
export async function decodeAll(src, consume, shouldStop = () => false) {
  const queue = []; let err = null, done = false, flushed = false, i = 0;
  const dec = new VideoDecoder({ output: f => queue.push(f), error: e => { err = e; } });
  dec.configure({ ...src.config });
  try {
    for (;;) {
      if (err) throw err;
      if (shouldStop()) throw new Error('Durduruldu');
      while (i < src.chunks.length && dec.decodeQueueSize < 6 && queue.length < 4) dec.decode(src.chunks[i++]);
      if (i >= src.chunks.length && !done) { done = true; dec.flush().then(() => { flushed = true; }, e => { err = e; }); }
      if (queue.length) { const f = queue.shift(); try { await consume(f); } finally { f.close(); } continue; }
      if (flushed) break;
      await new Promise(r => setTimeout(r, 2));
    }
  } finally { for (const f of queue) f.close(); try { dec.close(); } catch {} }
}

// Yedek yol: <video> öğesini oynatıp her karede durdurur (WebCodecs'in açamadığı dosyalar için)
export async function playFrames(video, consume, shouldStop = () => false) {
  video.muted = true; video.currentTime = 0;
  await new Promise(r => { if (video.readyState >= 2) r(); else video.addEventListener('loadeddata', r, { once: true }); });
  for (;;) {
    if (shouldStop()) throw new Error('Durduruldu');
    const meta = await new Promise(res => {
      let done = false; const fin = v => { if (!done) { done = true; res(v); } };
      video.requestVideoFrameCallback((n, m) => { video.pause(); fin(m); });
      video.onended = () => fin(null);
      video.play().catch(() => fin(null));
    });
    if (!meta) break;
    await consume(video, Math.round(meta.mediaTime * 1e6));
  }
  video.pause(); video.onended = null;
}

// Kodlayıcı + MP4 kabı. Ses verildiyse aynen eklenir.
export async function createWriter(W, H, fps, audio, bitsPerPixel = 0.16) {
  const bitrate = Math.round(W * H * fps * bitsPerPixel);
  let cfg = null;
  for (const codec of ['avc1.640033', 'avc1.640028', 'avc1.4d0028', 'avc1.42e01f']) {
    const c = { codec, width: W, height: H, bitrate, framerate: Math.round(fps), avc: { format: 'avc' } };
    try { if ((await VideoEncoder.isConfigSupported(c)).supported) { cfg = c; break; } } catch {}
  }
  if (!cfg) throw new Error('Bu cihaz video kodlamayı desteklemiyor');
  const target = new Mp4Muxer.ArrayBufferTarget();
  const muxer = new Mp4Muxer.Muxer({ target, fastStart: 'in-memory', firstTimestampBehavior: 'offset',
    video: { codec: 'avc', width: W, height: H, frameRate: Math.round(fps) },
    ...(audio ? { audio: { codec: 'aac', sampleRate: audio.sampleRate, numberOfChannels: audio.channels } } : {}) });
  let err = null, lastKey = -Infinity;
  const enc = new VideoEncoder({ output: (c, m) => muxer.addVideoChunk(c, m), error: e => { err = e; } });
  enc.configure(cfg);
  if (audio) {
    const meta = { decoderConfig: { codec: 'mp4a.40.2', sampleRate: audio.sampleRate, numberOfChannels: audio.channels, description: audio.description } };
    const t0 = audio.samples[0].ts;
    audio.samples.forEach((s, i) => muxer.addAudioChunkRaw(s.data, 'key', s.ts - t0, s.dur, i === 0 ? meta : undefined));
  }
  return {
    async add(canvas, tsUs, durUs) {
      if (err) throw err;
      const vf = new VideoFrame(canvas, { timestamp: tsUs, duration: durUs });
      const key = tsUs - lastKey >= 2e6; if (key) lastKey = tsUs;
      enc.encode(vf, { keyFrame: key }); vf.close();
      while (enc.encodeQueueSize > 4) await new Promise(r => setTimeout(r, 2));
    },
    async finish() { await enc.flush(); enc.close(); muxer.finalize(); return new Blob([target.buffer], { type: 'video/mp4' }); },
    abort() { try { enc.close(); } catch {} },
  };
}
