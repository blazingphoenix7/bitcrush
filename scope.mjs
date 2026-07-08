// WEIGHT SCOPE — a phosphor oscilloscope drawn over a real 8k-weight sample of one attention layer.
// Top trace: one real weight snapping to the nearest representable level (red residual stub).
// Main trace: the weight population (Gaussian bell) piling onto 2^bits levels; error mass in hot
// amber below the baseline. Pure canvas, no re-inference — updates live while the lever moves.
import { uniformLevels, percentileLevels, snapToLevels, niceRange } from "./grid-data.mjs";

const PHOS = "#45ff78", PHOS_DIM = "rgba(69,255,120,.34)", GRID = "rgba(69,255,120,.10)";
const HOT = "#ff9b3d", RED = "#ff5340";

export function createScope(canvas, sample) {
  const ctx = canvas.getContext("2d");
  const { min, max } = niceRange(sample, 3.4);
  const sorted = Float32Array.from(sample).sort();
  // pick a photogenic atom: clearly off-zero so the snap reads
  let atomIdx = 0, best = 1e9;
  for (let i = 0; i < sample.length; i++) { const d = Math.abs(Math.abs(sample[i]) - 0.042); if (d < best) { best = d; atomIdx = i; } }
  const wStar = sample[atomIdx];

  let W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    W = canvas.clientWidth || 300; H = canvas.clientHeight || 150;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  let curBits = 16, curMode = "smart";
  function levelsFor(bitsInt, mode) {
    const capped = Math.min(bitsInt, 12);
    return mode === "smart" ? percentileLevels(sorted, capped) : uniformLevels(min, max, capped);
  }

  function draw() {
    if (!W) resize();
    const bitsInt = Math.max(1, Math.min(16, Math.round(curBits)));
    const nLev = 1 << bitsInt;
    const levels = levelsFor(bitsInt, curMode);
    const lossless = nLev > 4096;
    const padX = 10, x0 = padX, x1 = W - padX;
    const xOf = (v) => x0 + (v - min) / (max - min) * (x1 - x0);

    ctx.clearRect(0, 0, W, H);
    // phosphor dot grid
    ctx.fillStyle = GRID;
    for (let gx = x0; gx <= x1; gx += 22) for (let gy = 8; gy < H - 6; gy += 18) ctx.fillRect(gx, gy, 1.4, 1.4);

    const atomY = 20, histBase = H - 26, histTop = 40, errH = 15;

    // representable levels (ticks) — countable only
    if (nLev <= 64) {
      ctx.strokeStyle = PHOS_DIM; ctx.lineWidth = 1;
      for (const lv of levels) { const x = xOf(lv); ctx.beginPath(); ctx.moveTo(x, histTop - 6); ctx.lineTo(x, histBase); ctx.stroke(); }
    } else {
      ctx.fillStyle = "rgba(69,255,120,.06)"; ctx.fillRect(x0, histTop - 6, x1 - x0, histBase - histTop + 6);
    }

    // population piles onto the levels; |error| mass below baseline
    const BINS = 130, qh = new Float32Array(BINS), eh = new Float32Array(BINS);
    for (let i = 0; i < sample.length; i++) {
      const w = sample[i], q = lossless ? w : snapToLevels(w, levels).q;
      let b = ((q - min) / (max - min) * BINS) | 0; if (b < 0) b = 0; else if (b >= BINS) b = BINS - 1;
      qh[b]++;
      let e = ((w - min) / (max - min) * BINS) | 0; if (e < 0) e = 0; else if (e >= BINS) e = BINS - 1;
      eh[e] += Math.abs(w - q);
    }
    let mq = 0, me = 0; for (let b = 0; b < BINS; b++) { if (qh[b] > mq) mq = qh[b]; if (eh[b] > me) me = eh[b]; }
    const bw = (x1 - x0) / BINS;
    for (let b = 0; b < BINS; b++) {
      const h = mq ? qh[b] / mq * (histBase - histTop) : 0;
      if (h < 0.4) continue;
      const x = x0 + b * bw;
      ctx.fillStyle = "rgba(69,255,120,.22)"; ctx.fillRect(x - 0.5, histBase - h - 1, bw + 0.4, h + 1);  // glow pass
      ctx.fillStyle = PHOS; ctx.fillRect(x, histBase - h, Math.max(bw - 0.7, 0.8), h);
    }
    if (me > 0) for (let b = 0; b < BINS; b++) {
      const h = eh[b] / me * errH; if (h < 0.4) continue;
      ctx.fillStyle = HOT; ctx.fillRect(x0 + b * bw, histBase + 3, Math.max(bw - 0.7, 0.8), h);
    }
    // baseline
    ctx.strokeStyle = PHOS_DIM; ctx.beginPath(); ctx.moveTo(x0, histBase + 1); ctx.lineTo(x1, histBase + 1); ctx.stroke();

    // atom trace: true value ghost → snapped dot, residual stub
    const snap = lossless ? { q: wStar, residual: 0 } : snapToLevels(wStar, levels);
    const xw = xOf(wStar), xq = xOf(snap.q);
    ctx.strokeStyle = "rgba(69,255,120,.22)"; ctx.beginPath(); ctx.moveTo(x0, atomY); ctx.lineTo(x1, atomY); ctx.stroke();
    ctx.fillStyle = PHOS_DIM; ctx.beginPath(); ctx.arc(xw, atomY, 3.2, 0, 7); ctx.fill();
    if (Math.abs(xq - xw) > 0.5) { ctx.strokeStyle = RED; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xw, atomY); ctx.lineTo(xq, atomY); ctx.stroke(); ctx.lineWidth = 1; }
    ctx.fillStyle = PHOS; ctx.beginPath(); ctx.arc(xq, atomY, 4, 0, 7); ctx.fill();
    ctx.font = "600 9px 'Geist Mono', monospace"; ctx.textAlign = "left";
    ctx.fillStyle = PHOS_DIM; ctx.fillText("w " + wStar.toFixed(4), x0, atomY - 8);
    if (Math.abs(snap.residual) > 1e-5) {
      ctx.fillStyle = RED; ctx.textAlign = "right";
      ctx.fillText("lost " + (snap.residual >= 0 ? "+" : "−") + Math.abs(snap.residual).toFixed(4), x1, atomY - 8);
    }
  }

  function update(bits, mode) { curBits = bits; curMode = mode; draw(); }
  function info() {
    const bitsInt = Math.max(1, Math.min(16, Math.round(curBits))), nLev = 1 << bitsInt;
    return { levels: nLev, step: (max - min) / (nLev - 1) };
  }
  const ro = new ResizeObserver(() => { resize(); draw(); });
  ro.observe(canvas);
  resize(); draw();
  return { update, info };
}
