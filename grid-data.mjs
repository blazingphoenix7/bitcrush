// Pure math for the Quantization Grid (Signature 1). No DOM, no WebGPU — unit-testable.
// The Grid illustrates the concept on a real sample of one layer's weights:
//   naive = uniform levels (evenly spaced across [min,max])
//   smart = percentile-spaced levels (cluster where the mass is — the NF4/AWQ intuition)
// This is an honest *illustration* of the idea, labeled as such — not a claim to be the engine's exact scheme.

// 2^bits evenly-spaced representable values across [min,max].
export function uniformLevels(min, max, bits) {
  const n = 1 << bits, out = new Float32Array(n);
  if (n === 1) { out[0] = (min + max) / 2; return out; }
  const step = (max - min) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = min + i * step;
  return out;
}

// 2^bits values placed at evenly-spaced quantiles of the sample → levels hug the dense middle.
export function percentileLevels(sample, bits) {
  const n = 1 << bits;
  const s = Float32Array.from(sample).sort();
  const out = new Float32Array(n);
  if (n === 1) { out[0] = s[(s.length - 1) >> 1]; return out; }
  for (let i = 0; i < n; i++) {
    const q = i / (n - 1);                         // quantile 0..1
    const pos = q * (s.length - 1), lo = Math.floor(pos), t = pos - lo;
    out[i] = lo + 1 < s.length ? s[lo] * (1 - t) + s[lo + 1] * t : s[lo];
  }
  return out;
}

// Snap w to the nearest value in a sorted `levels` array → { q, idx, residual = w - q }.
export function snapToLevels(w, levels) {
  let lo = 0, hi = levels.length - 1;
  if (w <= levels[0]) return { q: levels[0], idx: 0, residual: w - levels[0] };
  if (w >= levels[hi]) return { q: levels[hi], idx: hi, residual: w - levels[hi] };
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (levels[mid] <= w) lo = mid; else hi = mid; }
  const idx = (w - levels[lo]) <= (levels[hi] - w) ? lo : hi;
  return { q: levels[idx], idx, residual: w - levels[idx] };
}

// Affine RTN snap onto uniform levels (matches the engine's per-group scheme conceptually).
export function snapUniform(w, min, max, bits) {
  const n = 1 << bits;
  if (n === 1) { const q = (min + max) / 2; return { q, idx: 0, residual: w - q }; }
  const step = (max - min) / (n - 1);
  const idx = Math.max(0, Math.min(n - 1, Math.round((w - min) / step)));
  const q = min + idx * step;
  return { q, idx, residual: w - q, step };
}

// Histogram counts over [min,max] into `bins` buckets.
export function histogram(sample, bins, min, max) {
  const out = new Float32Array(bins), span = (max - min) || 1;
  for (let i = 0; i < sample.length; i++) {
    let b = Math.floor((sample[i] - min) / span * bins);
    if (b < 0) b = 0; else if (b >= bins) b = bins - 1;
    out[b]++;
  }
  return out;
}

// Robust axis range for a sample: symmetric around 0 at a few standard deviations (weights are ~Gaussian).
export function niceRange(sample, sigmas = 4) {
  let mean = 0; for (let i = 0; i < sample.length; i++) mean += sample[i]; mean /= sample.length;
  let v = 0; for (let i = 0; i < sample.length; i++) { const d = sample[i] - mean; v += d * d; }
  const sd = Math.sqrt(v / sample.length);
  const r = Math.max(Math.abs(mean) + sigmas * sd, 1e-4);
  return { min: -r, max: r, mean, sd };
}
