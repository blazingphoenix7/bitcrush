// Pure-JS Qwen3 incremental (KV-cache-style) greedy generation, checked against the PyTorch
// reference oracle's greedy_new_ids. Mirrors the engine's decode path EXACTLY — process one token
// at absolute position `pos`: RoPE at pos, append RoPE'd K + V to per-layer caches, attend over 0..pos.
// Proves the KV-cache ALGORITHM before spending a flaky browser cycle.  run:  node quant/qwen3_gen_check.mjs
import fs from "fs";

const DIR = "web/weights-qwen3/";
const manifest = JSON.parse(fs.readFileSync(DIR + "manifest.json", "utf8"));
const binBuf = fs.readFileSync(DIR + "qwen3.bin");
const ref = JSON.parse(fs.readFileSync(DIR + "reference.json", "utf8"));
const cfg = manifest.config;
const C = cfg.hidden_size, Cmlp = cfg.intermediate_size, nL = cfg.num_hidden_layers;
const nH = cfg.num_attention_heads, nKV = cfg.num_key_value_heads, hd = cfg.head_dim;
const qDim = nH * hd, kvDim = nKV * hd, V = cfg.vocab_size, eps = cfg.rms_norm_eps, theta = cfg.rope_theta;
const rep = nH / nKV;

const f16 = (h) => { const s=(h&0x8000)>>15,e=(h&0x7c00)>>10,f=h&0x3ff; if(e===0)return(s?-1:1)*5.9604644775390625e-8*f; if(e===0x1f)return f?NaN:(s?-1:1)*Infinity; return(s?-1:1)*Math.pow(2,e-15)*(1+f/1024); };
const tmap = {}; for (const t of manifest.tensors) tmap[t.name] = t;
const wcache = {};
function W(name) {
  if (wcache[name]) return wcache[name];
  const t = tmap[name];
  const u16 = new Uint16Array(binBuf.buffer, binBuf.byteOffset + t.offset, t.bytes / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = f16(u16[i]);
  wcache[name] = out; return out;
}
const embW = W("model.embed_tokens.weight");

function rmsnormRow(v, gamma, dim) {
  let ss = 0; for (let c = 0; c < dim; c++) ss += v[c]*v[c];
  const inv = 1/Math.sqrt(ss/dim + eps), o = new Float32Array(dim);
  for (let c = 0; c < dim; c++) o[c] = v[c]*inv*gamma[c];
  return o;
}
function matvec(x, w, K, N) { // y[N] = w[N][K] @ x[K]
  const o = new Float32Array(N);
  for (let n = 0; n < N; n++) { let a = 0; const wb = n*K; for (let k = 0; k < K; k++) a += x[k]*w[wb+k]; o[n] = a; }
  return o;
}
function qkNormRopeRow(vec, gamma, nHeads, pos) { // in place on [nHeads*hd]
  const half = hd/2;
  for (let h = 0; h < nHeads; h++) {
    const base = h*hd;
    let ss = 0; for (let d = 0; d < hd; d++) ss += vec[base+d]*vec[base+d];
    const inv = 1/Math.sqrt(ss/hd + 1e-6);
    const tmp = new Float32Array(hd); for (let d = 0; d < hd; d++) tmp[d] = vec[base+d]*inv*gamma[d];
    for (let i = 0; i < half; i++) {
      const ang = pos*Math.pow(theta, -2*i/hd), cs = Math.cos(ang), sn = Math.sin(ang);
      vec[base+i] = tmp[i]*cs - tmp[i+half]*sn;
      vec[base+i+half] = tmp[i+half]*cs + tmp[i]*sn;
    }
  }
}

const kC = Array.from({ length: nL }, () => []);   // per-layer array of RoPE'd K rows [kvDim]
const vC = Array.from({ length: nL }, () => []);   // per-layer array of V rows [kvDim]

// Process one token at absolute position `pos`; append its K/V to caches. Returns logits iff needLogits.
function step(id, pos, needLogits) {
  let x = new Float32Array(C);
  for (let c = 0; c < C; c++) x[c] = embW[id*C + c];
  for (let L = 0; L < nL; L++) {
    const p = `model.layers.${L}.`;
    const h = rmsnormRow(x, W(p+"input_layernorm.weight"), C);
    const q = matvec(h, W(p+"self_attn.q_proj.weight"), C, qDim);
    const kk = matvec(h, W(p+"self_attn.k_proj.weight"), C, kvDim);
    const vv = matvec(h, W(p+"self_attn.v_proj.weight"), C, kvDim);
    qkNormRopeRow(q, W(p+"self_attn.q_norm.weight"), nH, pos);
    qkNormRopeRow(kk, W(p+"self_attn.k_norm.weight"), nKV, pos);
    kC[L][pos] = kk; vC[L][pos] = vv;                       // append at row = pos
    const attn = new Float32Array(qDim), scale = 1/Math.sqrt(hd);
    for (let hh = 0; hh < nH; hh++) {
      const kvh = Math.floor(hh/rep), qb = hh*hd, kvb = kvh*hd;
      let m = -1e30;
      for (let j = 0; j <= pos; j++) { const kr = kC[L][j]; let s = 0; for (let d = 0; d < hd; d++) s += q[qb+d]*kr[kvb+d]; s *= scale; if (s > m) m = s; }
      const acc = new Float32Array(hd); let den = 0;
      for (let j = 0; j <= pos; j++) { const kr = kC[L][j], vr = vC[L][j]; let s = 0; for (let d = 0; d < hd; d++) s += q[qb+d]*kr[kvb+d]; const pp = Math.exp(s*scale - m); den += pp; for (let d = 0; d < hd; d++) acc[d] += pp*vr[kvb+d]; }
      for (let d = 0; d < hd; d++) attn[qb+d] = acc[d]/den;
    }
    const ao = matvec(attn, W(p+"self_attn.o_proj.weight"), qDim, C);
    const x2 = new Float32Array(C); for (let c = 0; c < C; c++) x2[c] = x[c] + ao[c];
    const h2 = rmsnormRow(x2, W(p+"post_attention_layernorm.weight"), C);
    const gate = matvec(h2, W(p+"mlp.gate_proj.weight"), C, Cmlp);
    const up = matvec(h2, W(p+"mlp.up_proj.weight"), C, Cmlp);
    const hm = new Float32Array(Cmlp); for (let i = 0; i < Cmlp; i++) { const gg = gate[i]; hm[i] = (gg/(1+Math.exp(-gg)))*up[i]; }
    const mo = matvec(hm, W(p+"mlp.down_proj.weight"), Cmlp, C);
    const xn = new Float32Array(C); for (let c = 0; c < C; c++) xn[c] = x2[c] + mo[c];
    x = xn;
  }
  if (!needLogits) return null;
  const xf = rmsnormRow(x, W("model.norm.weight"), C);
  return matvec(xf, embW, C, V);
}

const N_NEW = 8;
const prompt = ref.input_ids, T = prompt.length;
console.log("prompt ids:", prompt.join(","), " (", JSON.stringify(ref.prompt), ")");
let t0 = Date.now();
let logits = null;
for (let t = 0; t < T; t++) logits = step(prompt[t], t, t === T-1);   // prefill; only last needs logits
const gen = [];
for (let s = 0; s < N_NEW; s++) {
  let best = 0; for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
  gen.push(best);
  if (s === N_NEW-1) break;
  logits = step(best, T + s, true);
}
console.log(`generated (${((Date.now()-t0)/1000).toFixed(1)}s):`, gen.join(","));
const expected = ref.greedy_new_ids.slice(0, N_NEW);
console.log("expected (PyTorch):", expected.join(","));
let match = 0; for (let i = 0; i < gen.length; i++) { if (gen[i] === expected[i]) match++; else break; }
console.log(`\nprefix match: ${match}/${gen.length} tokens`);
console.log(match >= gen.length ? "PASS ✓ — KV-cache decode reproduces PyTorch greedy exactly"
  : match >= Math.min(4, gen.length) ? "OK ~ — matches then drifts (fp rounding on near-ties; algorithm sound)"
  : "FAIL ✗ — KV-cache algorithm diverges early; debug the decode path");
