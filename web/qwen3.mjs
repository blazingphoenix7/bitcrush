// Bitcrush WebGPU Qwen3-0.6B-Base inference engine.
// Forward pass in WGSL: embed -> 28x[RMSNorm, QKV(Linear), QK-norm, RoPE, GQA-attn, o_proj, +res,
//   RMSNorm, SwiGLU MLP, +res] -> final RMSNorm -> lm_head(tied). fp32 activations & weights (v1).
// Weights are nn.Linear [out,in]. Verified against a PyTorch reference oracle.

const WGSL = /* wgsl */`
struct EmbedP { T:u32, C:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<uniform> ep: EmbedP;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> emb: array<f32>;
@group(0) @binding(3) var<storage, read_write> xo: array<f32>;
@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= ep.T * ep.C) { return; }
  let t = i / ep.C; let c = i % ep.C;
  xo[i] = emb[ids[t]*ep.C + c];
}

struct RmsP { T:u32, C:u32, eps:f32, _a:u32 };
@group(0) @binding(0) var<uniform> rp: RmsP;
@group(0) @binding(1) var<storage, read> ri: array<f32>;
@group(0) @binding(2) var<storage, read> rg: array<f32>;   // gamma [C]
@group(0) @binding(3) var<storage, read_write> ro: array<f32>;
@compute @workgroup_size(64)
fn rmsnorm(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x; if (t >= rp.T) { return; }
  let base = t * rp.C;
  var ss = 0.0;
  for (var c=0u; c<rp.C; c++) { let v = ri[base+c]; ss += v*v; }
  let inv = inverseSqrt(ss / f32(rp.C) + rp.eps);
  for (var c=0u; c<rp.C; c++) { ro[base+c] = ri[base+c]*inv*rg[c]; }
}

// Linear matmul, no bias: y[T,N] = x[T,K] @ W[N,K]^T
struct MmP { T:u32, K:u32, N:u32, _a:u32 };
@group(0) @binding(0) var<uniform> mp: MmP;
@group(0) @binding(1) var<storage, read> mx: array<f32>;
@group(0) @binding(2) var<storage, read> mw: array<f32>;   // [N][K]
@group(0) @binding(3) var<storage, read_write> my: array<f32>;
@compute @workgroup_size(64)
fn matmul(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x; if (idx >= mp.T * mp.N) { return; }
  let t = idx / mp.N; let n = idx % mp.N;
  let wb = n*mp.K; let xb = t*mp.K;
  var acc = 0.0;
  for (var k=0u; k<mp.K; k++) { acc += mx[xb+k]*mw[wb+k]; }
  my[idx] = acc;
}

// per-head RMSNorm over head_dim (QK-norm), then RoPE (rotate_half). in place on [T][nHeads*hd].
// posBase = absolute position of row 0 (0 for prefill, cache length for decode).
struct QkP { T:u32, nHeads:u32, hd:u32, theta:f32, posBase:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<uniform> qp: QkP;
@group(0) @binding(1) var<storage, read_write> qkio: array<f32>;
@group(0) @binding(2) var<storage, read> qkg: array<f32>;   // norm gamma [hd]
@compute @workgroup_size(64)
fn qk_norm_rope(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x; if (idx >= qp.T * qp.nHeads) { return; }
  let t = idx / qp.nHeads; let h = idx % qp.nHeads;
  let base = t*(qp.nHeads*qp.hd) + h*qp.hd;
  var ss = 0.0;
  for (var d=0u; d<qp.hd; d++) { let v = qkio[base+d]; ss += v*v; }
  let inv = 1.0 / sqrt(ss / f32(qp.hd) + 1e-6);
  let half = qp.hd/2u; let pos = f32(qp.posBase + t);
  for (var i=0u; i<half; i++) {  // pairs are disjoint; read original values before writing
    let x1 = qkio[base+i]*inv*qkg[i];
    let x2 = qkio[base+i+half]*inv*qkg[i+half];
    let ang = pos * pow(qp.theta, -2.0*f32(i)/f32(qp.hd));
    let cs = cos(ang); let sn = sin(ang);
    qkio[base+i] = x1*cs - x2*sn;
    qkio[base+i+half] = x2*cs + x1*sn;
  }
}

// GQA attention over a K/V cache. q [Tq][nH*hd] (new rows), ak/av = cache [*][nKV*hd] -> out [Tq][nH*hd].
// Query row i sits at absolute position qPosBase+i and attends causally over cache rows 0..(qPosBase+i).
struct AtP { Tq:u32, nH:u32, nKV:u32, hd:u32, qPosBase:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<uniform> ap: AtP;
@group(0) @binding(1) var<storage, read> aq: array<f32>;
@group(0) @binding(2) var<storage, read> ak: array<f32>;   // cache
@group(0) @binding(3) var<storage, read> av: array<f32>;   // cache
@group(0) @binding(4) var<storage, read_write> aout: array<f32>;
@compute @workgroup_size(64)
fn attention(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x; if (idx >= ap.nH * ap.Tq) { return; }
  let h = idx / ap.Tq; let i = idx % ap.Tq;
  let qpos = ap.qPosBase + i;
  let kvh = h / (ap.nH / ap.nKV);
  let qbase = i*(ap.nH*ap.hd) + h*ap.hd;
  let scale = 1.0 / sqrt(f32(ap.hd));
  var m = -1e30;
  for (var j=0u; j<=qpos; j++) {
    let kb = j*(ap.nKV*ap.hd) + kvh*ap.hd;
    var s = 0.0; for (var d=0u; d<ap.hd; d++) { s += aq[qbase+d]*ak[kb+d]; }
    s *= scale; if (s > m) { m = s; }
  }
  var acc: array<f32, 128>;
  for (var d=0u; d<ap.hd; d++) { acc[d] = 0.0; }
  var denom = 0.0;
  for (var j=0u; j<=qpos; j++) {
    let kb = j*(ap.nKV*ap.hd) + kvh*ap.hd;
    var s = 0.0; for (var d=0u; d<ap.hd; d++) { s += aq[qbase+d]*ak[kb+d]; }
    let p = exp(s*scale - m); denom += p;
    for (var d=0u; d<ap.hd; d++) { acc[d] += p*av[kb+d]; }
  }
  let ob = i*(ap.nH*ap.hd) + h*ap.hd;
  for (var d=0u; d<ap.hd; d++) { aout[ob+d] = acc[d]/denom; }
}

struct ElP { n:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<uniform> sp: ElP;
@group(0) @binding(1) var<storage, read_write> sgate: array<f32>;  // in/out (avoid binding one buffer twice)
@group(0) @binding(2) var<storage, read> sup: array<f32>;
@compute @workgroup_size(64)
fn silu_mul(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= sp.n) { return; }
  let x = sgate[i];
  sgate[i] = (x / (1.0 + exp(-x))) * sup[i];
}

@group(0) @binding(0) var<uniform> addp: ElP;
@group(0) @binding(1) var<storage, read_write> adst: array<f32>;
@group(0) @binding(2) var<storage, read> asrc: array<f32>;
@compute @workgroup_size(64)
fn add_inplace(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= addp.n) { return; }
  adst[i] += asrc[i];
}

struct LhP { T:u32, C:u32, V:u32, _a:u32 };
@group(0) @binding(0) var<uniform> hp: LhP;
@group(0) @binding(1) var<storage, read> hx: array<f32>;
@group(0) @binding(2) var<storage, read> hemb: array<f32>;
@group(0) @binding(3) var<storage, read_write> ho: array<f32>;
@compute @workgroup_size(64)
fn lmhead(@builtin(global_invocation_id) g: vec3<u32>) {
  let v = g.x; if (v >= hp.V) { return; }
  let xoff = (hp.T-1u)*hp.C; let eb = v*hp.C;
  var acc = 0.0;
  for (var c=0u; c<hp.C; c++) { acc += hx[xoff+c]*hemb[eb+c]; }
  ho[v] = acc;
}

// lm_head for a RANGE of rows [r0, r0+R) → out rows at [oOff, oOff+R) of the output. Chunkable so the
// teacher-forced ghost can submit in slices and let user work interleave on the GPU queue.
struct LhrP { r0:u32, R:u32, C:u32, V:u32, oOff:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<uniform> hrp: LhrP;
@group(0) @binding(1) var<storage, read> hrx: array<f32>;    // hidden [T][C]
@group(0) @binding(2) var<storage, read> hremb: array<f32>;  // embed [V][C]
@group(0) @binding(3) var<storage, read_write> hro: array<f32>;  // [Rtotal][V]
// 2D dispatch: x over V (in 64-thread groups), y over R rows — keeps each dim under the 65535 workgroup cap.
@compute @workgroup_size(64, 1, 1)
fn lmhead_range(@builtin(global_invocation_id) g: vec3<u32>) {
  let v = g.x; let r = g.y;
  if (v >= hrp.V || r >= hrp.R) { return; }
  let xoff = (hrp.r0 + r)*hrp.C; let eb = v*hrp.C;
  var acc = 0.0;
  for (var c=0u; c<hrp.C; c++) { acc += hrx[xoff+c]*hremb[eb+c]; }
  hro[(hrp.oOff + r)*hrp.V + v] = acc;
}

// Live re-quant, memory-light: keep ONE master copy, precompute per-group (scale, zero), and
// dequantize each weight inline in the matmul (no full quantized copy → no memory doubling).
// Group-wise affine RTN along K. Weights are nn.Linear [N=out][K=in].

// Pass 1: per (row n, group grp) min/max -> scale & zero. Output arrays are [N * (K/G)].
struct QsP { N:u32, K:u32, G:u32, bits:u32 };
@group(0) @binding(0) var<uniform> qsp: QsP;
@group(0) @binding(1) var<storage, read> qsw: array<f32>;        // master [N][K]
@group(0) @binding(2) var<storage, read_write> qss: array<f32>;  // scales [N*nG]
@group(0) @binding(3) var<storage, read_write> qsz: array<f32>;  // zeros  [N*nG]
@compute @workgroup_size(64)
fn quant_stats(@builtin(global_invocation_id) g: vec3<u32>) {
  let nG = qsp.K / qsp.G;
  let idx = g.x; if (idx >= qsp.N * nG) { return; }
  let n = idx / nG; let grp = idx % nG;
  let k0 = grp*qsp.G; let rowb = n*qsp.K;
  var mn = 1e30; var mx = -1e30;
  for (var k=k0; k<k0+qsp.G; k++) { let w = qsw[rowb+k]; mn = min(mn,w); mx = max(mx,w); }
  let qmax = f32((1u << qsp.bits) - 1u);
  let scale = max(mx-mn, 1e-8) / qmax;
  qss[idx] = scale;
  qsz[idx] = round(-mn/scale);
}

// Pass 2: y[T,N] = x[T,K] @ dequant(W)[N,K]^T, dequantizing each master weight on read.
struct MmqP { T:u32, K:u32, N:u32, G:u32, bits:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<uniform> mqp: MmqP;
@group(0) @binding(1) var<storage, read> mqx: array<f32>;   // activations [T][K]
@group(0) @binding(2) var<storage, read> mqw: array<f32>;   // master weights [N][K]
@group(0) @binding(3) var<storage, read> mqs: array<f32>;   // scales [N*nG]
@group(0) @binding(4) var<storage, read> mqz: array<f32>;   // zeros  [N*nG]
@group(0) @binding(5) var<storage, read_write> mqy: array<f32>;
@compute @workgroup_size(64)
fn matmul_q(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x; if (idx >= mqp.T * mqp.N) { return; }
  let t = idx / mqp.N; let n = idx % mqp.N;
  let nG = mqp.K / mqp.G;
  let xb = t*mqp.K; let wb = n*mqp.K; let sb = n*nG;
  let qmax = f32((1u << mqp.bits) - 1u);
  var acc = 0.0;
  for (var grp=0u; grp<nG; grp++) {
    let s = mqs[sb+grp]; let z = mqz[sb+grp];
    let k0 = grp*mqp.G;
    for (var k=k0; k<k0+mqp.G; k++) {
      let q = clamp(round(mqw[wb+k]/s) + z, 0.0, qmax);
      acc += mqx[xb+k] * ((q - z)*s);
    }
  }
  mqy[idx] = acc;
}
`;

// ---------- fp16 -> fp32 ----------
let _f16tab = null;
function f16Table() {
  if (_f16tab) return _f16tab;
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) t[h] = (s ? -1 : 1) * 5.9604644775390625e-8 * f;
    else if (e === 0x1f) t[h] = f ? NaN : (s ? -1 : 1) * Infinity;
    else t[h] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  _f16tab = t; return t;
}
function f16ToF32Array(u16) {
  if (typeof Float16Array !== "undefined") return new Float32Array(new Float16Array(u16.buffer, u16.byteOffset, u16.length));
  const t = f16Table(), out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = t[u16[i]];
  return out;
}
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  const total = +res.headers.get("Content-Length") || 0;
  if (!res.body || !total) return await res.arrayBuffer();
  // Preallocate the full buffer and copy each chunk straight in — avoids holding both a
  // chunks[] array AND a concatenated copy (~2x the file size transiently, which OOM'd the iGPU).
  const buf = new Uint8Array(total);
  const reader = res.body.getReader();
  let received = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; buf.set(value, received); received += value.length; onProgress?.(received / total); }
  return buf.buffer;
}

export async function loadModel(baseUrl, opts = {}) {
  if (typeof opts === "function") opts = { log: opts };
  const log = opts.log || (() => {});
  if (!navigator.gpu) throw new Error("NO_WEBGPU");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("NO_ADAPTER");
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });

  log("fetching model…");
  const manifest = await (await fetch(baseUrl + "manifest.json")).json();
  const bin = await fetchWithProgress(baseUrl + "qwen3.bin", opts.onProgress);
  log(`uploading ${(bin.byteLength / 1e6).toFixed(0)}MB to GPU…`);

  const W = {};
  for (const t of manifest.tensors) {
    const u16 = new Uint16Array(bin, t.offset, t.bytes / 2);
    const f32 = f16ToF32Array(u16);
    const buf = device.createBuffer({ size: f32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(buf, 0, f32);        // COPY_SRC so the Grid can read a weight sample back
    W[t.name] = { buf, shape: t.shape };
  }

  const module = device.createShaderModule({ code: WGSL });
  const pipe = (e) => device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: e } });
  const pipes = {
    embed: pipe("embed"), rmsnorm: pipe("rmsnorm"), matmul: pipe("matmul"),
    qk_norm_rope: pipe("qk_norm_rope"), attention: pipe("attention"), silu_mul: pipe("silu_mul"),
    add: pipe("add_inplace"), lmhead: pipe("lmhead"), lmhead_range: pipe("lmhead_range"),
    quant_stats: pipe("quant_stats"), matmul_q: pipe("matmul_q"),
  };
  return { device, W, pipes, cfg: manifest.config, quantized: false, qbits: 16, qgroup: 0, scales: {}, zeros: {}, _G: {}, _sz: {} };
}

function ubuf(device, ints, floats = []) {
  const a = new ArrayBuffer(32);  // 8 slots; structs read only what they declare
  const iv = new Uint32Array(a), fv = new Float32Array(a);
  ints.forEach((x, i) => (iv[i] = x));
  floats.forEach(([i, x]) => (fv[i] = x));
  const b = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, a);
  return b;
}
function sbuf(device, n) { return device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }); }
// Record a compute dispatch into the shared encoder (ctx.enc). Nothing is submitted here —
// the whole forward batches into one queue.submit() (flush), cutting ~340 submits/token to 1.
// WebGPU auto-inserts barriers between passes in an encoder, so sequential kernels stay correct.
function run(ctx, pipeline, buffers, threads) {
  const bind = ctx.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const pass = ctx.enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(threads / 64)); pass.end();
}
function flush(ctx) { ctx.device.queue.submit([ctx.enc.finish()]); ctx.enc = ctx.device.createCommandEncoder(); }
// 2D dispatch (for kernels whose 1D workgroup count would exceed 65535).
function run2d(ctx, pipeline, buffers, gx, gy) {
  const bind = ctx.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const pass = ctx.enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(gx, gy, 1); pass.end();
}
async function readBuf(device, buf, n) {
  const rb = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap(); rb.destroy();
  return out;
}

function copyInto(ctx, src, dst, dstElemOffset, nElems) {
  ctx.enc.copyBufferToBuffer(src, 0, dst, dstElemOffset * 4, nElems * 4);
}

// Persistent per-layer K/V cache for incremental decoding. Reused across generation steps.
export function createKV(model, maxSeq) {
  const { device, cfg } = model;
  const kvDim = cfg.num_key_value_heads * cfg.head_dim, nL = cfg.num_hidden_layers;
  const kCache = [], vCache = [];
  for (let L = 0; L < nL; L++) { kCache.push(sbuf(device, maxSeq * kvDim)); vCache.push(sbuf(device, maxSeq * kvDim)); }
  return { kCache, vCache, len: 0, maxSeq, destroy() { for (const b of kCache) b.destroy(); for (const b of vCache) b.destroy(); } };
}

// The 7 Linear weight matrices per layer that get quantized (embed / lm_head / norms stay full precision).
export function quantNames(cfg) {
  const names = [];
  for (let L = 0; L < cfg.num_hidden_layers; L++) {
    const p = `model.layers.${L}.`;
    names.push(p+"self_attn.q_proj.weight", p+"self_attn.k_proj.weight", p+"self_attn.v_proj.weight",
      p+"self_attn.o_proj.weight", p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight");
  }
  return names;
}

// Set the active weight precision. bits>=16 → full precision (plain matmul). Otherwise precompute
// per-group (scale, zero) for every quantized matrix; matmul_q then dequantizes inline.
// groupSize<=0 (or > K) means per-output-channel ("naive"); a small value like 64 is "smart".
export function setQuant(model, bits, groupSize = 64) {
  const { device, W, pipes, cfg } = model;
  if (!bits || bits >= 16) { model.quantized = false; model.qbits = 16; return; }
  model.quantized = true; model.qbits = bits; model.qgroup = groupSize;
  const ctx = { device, enc: device.createCommandEncoder() };
  for (const name of quantNames(model.cfg)) {
    const [N, K] = W[name].shape;                                  // nn.Linear [out, in]
    let G = (groupSize > 0 && groupSize <= K) ? groupSize : K;     // clamp to per-channel
    if (K % G !== 0) G = K;                                        // group must divide K → fall back
    const nG = K / G, need = N * nG;
    model._G[name] = G;
    if (model._sz[name] !== need) {                               // (re)allocate scale/zero buffers on shape change
      model.scales[name]?.destroy(); model.zeros[name]?.destroy();
      model.scales[name] = sbuf(device, need); model.zeros[name] = sbuf(device, need);
      model._sz[name] = need;
    }
    run(ctx, pipes.quant_stats, [ubuf(device, [N, K, G, bits]), W[name].buf, model.scales[name], model.zeros[name]], need);
  }
  flush(ctx);
}

// Run Tq tokens at absolute positions kv.len..kv.len+Tq-1, append their K/V to the cache,
// advance kv.len, and return logits for the LAST row. Handles both prefill (Tq=prompt) and decode (Tq=1).
async function runTokens(model, tokenIds, kv, opts = {}) {
  const { device, W, pipes, cfg } = model;
  const C = cfg.hidden_size, Cmlp = cfg.intermediate_size, nL = cfg.num_hidden_layers;
  const nH = cfg.num_attention_heads, nKV = cfg.num_key_value_heads, hd = cfg.head_dim;
  const qDim = nH * hd, kvDim = nKV * hd, V = cfg.vocab_size, eps = cfg.rms_norm_eps, theta = cfg.rope_theta;
  const Tq = tokenIds.length, posBase = kv.len;

  const idsBuf = device.createBuffer({ size: Tq * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(idsBuf, 0, new Uint32Array(tokenIds));

  const x = sbuf(device, Tq * C), h = sbuf(device, Tq * C), attn = sbuf(device, Tq * qDim);
  const q = sbuf(device, Tq * qDim), ktmp = sbuf(device, Tq * kvDim), vtmp = sbuf(device, Tq * kvDim);
  const gate = sbuf(device, Tq * Cmlp), up = sbuf(device, Tq * Cmlp), tmp = sbuf(device, Tq * C), xf = sbuf(device, Tq * C);
  const logits = sbuf(device, V);
  const debug = {};
  const ctx = { device, enc: device.createCommandEncoder() };   // all kernels record here; one flush at the end

  // Uniforms constant across all 28 layers (only weight buffers change) — build once, reuse, free at end.
  const U = [];
  const u = (ints, floats) => { const b = ubuf(device, ints, floats); U.push(b); return b; };
  const uRms = u([Tq, C, 0], [[2, eps]]);
  const uRopeQ = u([Tq, nH, hd, 0, posBase], [[3, theta]]), uRopeK = u([Tq, nKV, hd, 0, posBase], [[3, theta]]);
  const uAttn = u([Tq, nH, nKV, hd, posBase]);
  const uAddC = u([Tq * C, 0, 0, 0]), uSilu = u([Tq * Cmlp, 0, 0, 0]);
  // Snapshot the precision for this whole pass — a setQuant() landing mid-flight (e.g. the user slams
  // the lever while the ghost is scanning) must not split-brain the layers of one forward.
  const useQuant = model.quantized, bits = model.qbits;
  // A Linear matmul at the active precision: full-precision plain matmul, or on-the-fly dequant (matmul_q).
  const mm = (name, xb, yb, K, N) => {
    if (useQuant) run(ctx, pipes.matmul_q, [u([Tq, K, N, model._G[name], bits]), xb, W[name].buf, model.scales[name], model.zeros[name], yb], Tq*N);
    else run(ctx, pipes.matmul, [u([Tq, K, N, 0]), xb, W[name].buf, yb], Tq*N);
  };

  run(ctx, pipes.embed, [u([Tq, C, 0, 0]), idsBuf, W["model.embed_tokens.weight"].buf, x], Tq * C);
  if (opts.debug) { flush(ctx); debug.embed = await readBuf(device, x, Tq * C); }

  for (let L = 0; L < nL; L++) {
    const p = `model.layers.${L}.`;
    const kC = kv.kCache[L], vC = kv.vCache[L];
    // debug reads must flush the pending encoder first so they see committed writes (verification path only)
    const dbg = async (name, buf, n) => { if (opts.debug && L === 0) { flush(ctx); debug[name] = await readBuf(device, buf, n); } };
    run(ctx, pipes.rmsnorm, [uRms, x, W[p+"input_layernorm.weight"].buf, h], Tq); await dbg("h1", h, Tq*C);
    mm(p+"self_attn.q_proj.weight", h, q, C, qDim);
    mm(p+"self_attn.k_proj.weight", h, ktmp, C, kvDim);
    mm(p+"self_attn.v_proj.weight", h, vtmp, C, kvDim);
    run(ctx, pipes.qk_norm_rope, [uRopeQ, q, W[p+"self_attn.q_norm.weight"].buf], Tq*nH); await dbg("q", q, Tq*qDim);
    run(ctx, pipes.qk_norm_rope, [uRopeK, ktmp, W[p+"self_attn.k_norm.weight"].buf], Tq*nKV); await dbg("k", ktmp, Tq*kvDim);
    copyInto(ctx, ktmp, kC, posBase*kvDim, Tq*kvDim);   // append RoPE'd keys to cache
    copyInto(ctx, vtmp, vC, posBase*kvDim, Tq*kvDim);   // append values to cache
    run(ctx, pipes.attention, [uAttn, q, kC, vC, attn], nH*Tq); await dbg("attn", attn, Tq*qDim);
    mm(p+"self_attn.o_proj.weight", attn, tmp, qDim, C); await dbg("ao", tmp, Tq*C);
    run(ctx, pipes.add, [uAddC, x, tmp], Tq*C);
    run(ctx, pipes.rmsnorm, [uRms, x, W[p+"post_attention_layernorm.weight"].buf, h], Tq);
    mm(p+"mlp.gate_proj.weight", h, gate, C, Cmlp);
    mm(p+"mlp.up_proj.weight", h, up, C, Cmlp);
    run(ctx, pipes.silu_mul, [uSilu, gate, up], Tq*Cmlp);
    mm(p+"mlp.down_proj.weight", gate, tmp, Cmlp, C); await dbg("mo", tmp, Tq*C);
    run(ctx, pipes.add, [uAddC, x, tmp], Tq*C);
    if (opts.debug && L === 0) { flush(ctx); debug.layer0 = await readBuf(device, x, Tq * C); }
    // cooperative mode (ghost scan): submit in slices and yield, so a user generation that starts
    // mid-pass interleaves on the GPU queue instead of waiting ~7s behind one giant batch
    if (opts.onYield && (L + 1) % (opts.yieldEvery || 4) === 0 && L < nL - 1) { flush(ctx); await opts.onYield(); }
  }

  run(ctx, pipes.rmsnorm, [uRms, x, W["model.norm.weight"].buf, xf], Tq);
  run(ctx, pipes.lmhead, [u([Tq, C, V, 0]), xf, W["model.embed_tokens.weight"].buf, logits], V);
  flush(ctx);                                     // submit the whole forward as one batch
  const out = await readBuf(device, logits, V);   // its own encoder; ordered after the flush

  // teacher-forced ghost: logits at every position in [fromPos, Tq) from the same xf, chunked so it
  // never monopolizes the queue
  if (opts.allLogitsFrom !== undefined) {
    const r0 = opts.allLogitsFrom, R = Tq - r0, allBuf = sbuf(device, R * V);
    const CH = opts.onYield ? 8 : R;               // row slices only in cooperative mode
    for (let c0 = 0; c0 < R; c0 += CH) {
      const rows = Math.min(CH, R - c0);
      const ctx2 = { device, enc: device.createCommandEncoder() };
      run2d(ctx2, pipes.lmhead_range, [ubuf(device, [r0 + c0, rows, C, V, c0]), xf, W["model.embed_tokens.weight"].buf, allBuf], Math.ceil(V / 64), rows);
      flush(ctx2);
      if (opts.onYield && c0 + CH < R) await opts.onYield();
    }
    const flat = await readBuf(device, allBuf, R * V);
    allBuf.destroy();
    debug.allLogits = []; for (let r = 0; r < R; r++) debug.allLogits.push(flat.slice(r * V, (r + 1) * V));
  }

  kv.len += Tq;
  for (const b of [idsBuf, x, h, attn, q, ktmp, vtmp, gate, up, tmp, xf, logits, ...U]) b.destroy();
  return { logits: out, debug };
}

// Teacher-forced fp16 pass over an emitted sequence: returns fp16 logits at each generated position
// so the UI can show "the word fp16 would have said" wherever the crushed model diverged.
export async function teacherForce(model, fullIds, fromPos) {
  model.quantized = false;                        // the ghost is always full precision
  // in a visible tab, yield between slices so a fresh user generation interleaves within ~1s;
  // hidden tabs have no user mid-drag (and throttle timers), so run as one fast batch there
  const onYield = typeof document !== "undefined" && !document.hidden
    ? () => new Promise((r) => setTimeout(r, 0)) : null;
  try {
    const kv = createKV(model, fullIds.length);
    try {
      const { debug } = await runTokens(model, fullIds, kv, { allLogitsFrom: fromPos, onYield, yieldEvery: 4 });
      return debug.allLogits;
    } finally { kv.destroy(); }
  } finally {
    // re-derive rather than restore a snapshot — setQuant may have run mid-scan
    model.quantized = model.qbits > 0 && model.qbits < 16;
  }
}

// Read a contiguous sample of a weight tensor to CPU (for the Quantization Grid). Warmup-only; ~8k floats.
export async function sampleWeights(model, name, maxCount = 8192) {
  const t = model.W[name];
  const n = Math.min(maxCount, t.shape[0] * t.shape[1]);
  return await readBuf(model.device, t.buf, n);
}

// Full non-cached forward over the whole sequence — used by the verification harness.
export async function forward(model, tokenIds, opts = {}) {
  const kv = createKV(model, tokenIds.length);
  try { return await runTokens(model, tokenIds, kv, opts); }
  finally { kv.destroy(); }
}

export function argmax(a) { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; }
// argmax skipping a Uint8Array ban mask (1 = never emit) and/or an explicit id list
export function argmaxMasked(a, mask, banned) {
  let k = -1;
  for (let i = 0; i < a.length; i++) {
    if (mask && mask[i]) continue;
    if (banned && banned.includes(i)) continue;
    if (k < 0 || a[i] > a[k]) k = i;
  }
  return k;
}

// Greedy generation with a KV cache: prefill the prompt once, then decode one token per step.
export async function generate(model, tokenIds, nNew, opts = {}) {
  const kv = createKV(model, tokenIds.length + nNew + 1);
  const gen = [];
  try {
    let { logits } = await runTokens(model, tokenIds, kv);   // prefill → distribution for the first new token
    for (let s = 0; s < nNew; s++) {
      let next = opts.banMask ? argmaxMasked(logits, opts.banMask, null) : argmax(logits);
      if (opts.eosIds && opts.eosIds.includes(next)) {
        // standard min-new-tokens: mask EOS early so an answer can't end after 3 tokens
        if (gen.length < (opts.minNew || 0)) next = argmaxMasked(logits, opts.banMask, opts.eosIds);
        else break;                                           // stop at end-of-turn; don't emit the token
      }
      gen.push(next);
      if (opts.onToken) await opts.onToken(next, gen, logits);   // logits = the distribution `next` was drawn from
      if (opts.signal?.aborted || s === nNew - 1) break;
      ({ logits } = await runTokens(model, [next], kv));      // decode the token we just emitted
    }
  } finally { kv.destroy(); }
  return gen;
}
