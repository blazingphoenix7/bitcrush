// Pure-JS Qwen3 layer-0 forward, checked against the PyTorch reference oracle — no WebGPU.
// Localizes the engine bug (embed ok, layer0 wrong) in fast Node cycles.
//   run:  node quant/qwen3_check.mjs
import fs from "fs";

const DIR = "web/weights-qwen3/";
const manifest = JSON.parse(fs.readFileSync(DIR + "manifest.json", "utf8"));
const bin = fs.readFileSync(DIR + "qwen3.bin");
const ref = JSON.parse(fs.readFileSync(DIR + "reference.json", "utf8"));
const refEmbed = new Float32Array(fs.readFileSync(DIR + ref.embed_file).buffer.slice(fs.readFileSync(DIR + ref.embed_file).byteOffset));
const refLayer0f = fs.readFileSync(DIR + ref.layer0_file);
const refLayer0 = new Float32Array(refLayer0f.buffer, refLayer0f.byteOffset, refLayer0f.byteLength / 4);

const C = manifest.config.hidden_size, Cmlp = manifest.config.intermediate_size;
const nH = manifest.config.num_attention_heads, nKV = manifest.config.num_key_value_heads, hd = manifest.config.head_dim;
const eps = manifest.config.rms_norm_eps, theta = manifest.config.rope_theta;
const qDim = nH * hd, kvDim = nKV * hd;

// fp16 -> fp32
const f16 = (h) => { const s=(h&0x8000)>>15,e=(h&0x7c00)>>10,f=h&0x3ff; if(e===0)return(s?-1:1)*5.9604644775390625e-8*f; if(e===0x1f)return f?NaN:(s?-1:1)*Infinity; return(s?-1:1)*Math.pow(2,e-15)*(1+f/1024); };
const tmap = {};
for (const t of manifest.tensors) tmap[t.name] = t;
function W(name) {
  const t = tmap[name];
  const u16 = new Uint16Array(bin.buffer, bin.byteOffset + t.offset, t.bytes / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = f16(u16[i]);
  return out;
}

const ids = ref.input_ids, T = ids.length;
const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i]-b[i]); if (d>m) m=d; } return m; };

// embed
const emb = W("model.embed_tokens.weight");
let x = new Float32Array(T * C);
for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) x[t*C+c] = emb[ids[t]*C+c];
console.log("embed maxAbs vs ref:", maxAbs(x, refEmbed).toExponential(3));

function rmsnorm(inp, gamma, rows, dim) {
  const out = new Float32Array(rows * dim);
  for (let t = 0; t < rows; t++) {
    let ss = 0; for (let c = 0; c < dim; c++) { const v = inp[t*dim+c]; ss += v*v; }
    const inv = 1 / Math.sqrt(ss/dim + eps);
    for (let c = 0; c < dim; c++) out[t*dim+c] = inp[t*dim+c]*inv*gamma[c];
  }
  return out;
}
function matmul(inp, w, rows, K, N) { // y[rows][N] = inp[rows][K] @ w[N][K]^T
  const out = new Float32Array(rows * N);
  for (let t = 0; t < rows; t++) for (let n = 0; n < N; n++) {
    let acc = 0; const xb = t*K, wb = n*K;
    for (let k = 0; k < K; k++) acc += inp[xb+k]*w[wb+k];
    out[t*N+n] = acc;
  }
  return out;
}
function qkNormRope(buf, gamma, nHeads) { // in place, per head: rmsnorm(hd) then rope
  const half = hd/2;
  for (let t = 0; t < T; t++) for (let h = 0; h < nHeads; h++) {
    const base = t*(nHeads*hd) + h*hd;
    let ss = 0; for (let d = 0; d < hd; d++) { const v = buf[base+d]; ss += v*v; }
    const inv = 1 / Math.sqrt(ss/hd + 1e-6);
    const tmp = new Float32Array(hd);
    for (let d = 0; d < hd; d++) tmp[d] = buf[base+d]*inv*gamma[d];
    for (let i = 0; i < half; i++) {
      const ang = t * Math.pow(theta, -2*i/hd);
      const cs = Math.cos(ang), sn = Math.sin(ang);
      buf[base+i] = tmp[i]*cs - tmp[i+half]*sn;
      buf[base+i+half] = tmp[i+half]*cs + tmp[i]*sn;
    }
  }
}
function attention(q, k, v) { // GQA -> [T][qDim]
  const out = new Float32Array(T * qDim);
  const scale = 1/Math.sqrt(hd), rep = nH/nKV;
  for (let h = 0; h < nH; h++) { const kvh = Math.floor(h/rep);
    for (let i = 0; i < T; i++) {
      const qb = i*qDim + h*hd;
      let m = -1e30;
      for (let j = 0; j <= i; j++) { const kb = j*kvDim + kvh*hd; let s=0; for(let d=0;d<hd;d++) s+=q[qb+d]*k[kb+d]; s*=scale; if(s>m)m=s; }
      const acc = new Float32Array(hd); let denom = 0;
      for (let j = 0; j <= i; j++) { const kb = j*kvDim + kvh*hd; let s=0; for(let d=0;d<hd;d++) s+=q[qb+d]*k[kb+d]; const p=Math.exp(s*scale-m); denom+=p; for(let d=0;d<hd;d++) acc[d]+=p*v[kb+d]; }
      const ob = i*qDim + h*hd;
      for (let d = 0; d < hd; d++) out[ob+d] = acc[d]/denom;
    }
  }
  return out;
}

const P = "model.layers.0.";
const h1 = rmsnorm(x, W(P+"input_layernorm.weight"), T, C);
const q = matmul(h1, W(P+"self_attn.q_proj.weight"), T, C, qDim);
const k = matmul(h1, W(P+"self_attn.k_proj.weight"), T, C, kvDim);
const v = matmul(h1, W(P+"self_attn.v_proj.weight"), T, C, kvDim);
qkNormRope(q, W(P+"self_attn.q_norm.weight"), nH);
qkNormRope(k, W(P+"self_attn.k_norm.weight"), nKV);
const attn = attention(q, k, v);
const ao = matmul(attn, W(P+"self_attn.o_proj.weight"), T, qDim, C);
const x2 = new Float32Array(T*C); for (let i=0;i<T*C;i++) x2[i] = x[i] + ao[i];
const h2 = rmsnorm(x2, W(P+"post_attention_layernorm.weight"), T, C);
const gate = matmul(h2, W(P+"mlp.gate_proj.weight"), T, C, Cmlp);
const up = matmul(h2, W(P+"mlp.up_proj.weight"), T, C, Cmlp);
const hm = new Float32Array(T*Cmlp); for (let i=0;i<T*Cmlp;i++) { const g=gate[i]; hm[i] = (g/(1+Math.exp(-g)))*up[i]; }
const mo = matmul(hm, W(P+"mlp.down_proj.weight"), T, Cmlp, C);
const out = new Float32Array(T*C); for (let i=0;i<T*C;i++) out[i] = x2[i] + mo[i];

console.log("layer0 maxAbs vs ref:", maxAbs(out, refLayer0).toExponential(3));
console.log(maxAbs(out, refLayer0) < 0.1 ? "LOGIC CORRECT -> bug is in the WGSL translation" : "LOGIC WRONG -> debug the JS math (add sub-stage refs)");

// dump verified sub-stage tensors so the WGSL engine can be diffed kernel-by-kernel
const wr = (name, arr) => fs.writeFileSync(DIR + "js_" + name + ".bin", Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
wr("h1", h1); wr("qroped", q); wr("kroped", k); wr("attn", attn); wr("ao", ao); wr("mo", mo);
console.log("wrote js_{h1,qroped,kroped,attn,ao,mo}.bin for WGSL diffing");
