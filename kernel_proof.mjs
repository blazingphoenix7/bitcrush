// Bitcrush WebGPU kernel proof: 4-bit group-wise dequant matmul.
//   Y[M,N] = X[M,K] @ dequant(Wq),  W is 4-bit group-quantized along K.
// This is the exact math the browser engine needs. Runs headless in Deno
// (navigator.gpu) or in a browser. Prints "RESULT PASS" if the GPU output
// matches a CPU/JS reference within tolerance.

const WGSL = `
struct Dims { M:u32, N:u32, K:u32, G:u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wq: array<u32>;      // 8 nibbles/u32, layout [N][K/8]
@group(0) @binding(3) var<storage, read> scales: array<f32>;  // [N][K/G]
@group(0) @binding(4) var<storage, read> zeros: array<f32>;   // [N][K/G]
@group(0) @binding(5) var<storage, read_write> Y: array<f32>; // [M][N]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= d.M * d.N) { return; }
  let m = idx / d.N;
  let n = idx % d.N;
  let kd8 = d.K / 8u;
  let kdG = d.K / d.G;
  var acc = 0.0;
  for (var k = 0u; k < d.K; k = k + 1u) {
    let word = Wq[n * kd8 + (k >> 3u)];
    let nib = (word >> ((k & 7u) * 4u)) & 15u;   // unpack 4-bit weight
    let g = k / d.G;
    let w = (f32(nib) - zeros[n * kdG + g]) * scales[n * kdG + g];  // dequant
    acc = acc + X[m * d.K + k] * w;
  }
  Y[m * d.N + n] = acc;
}
`;

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

async function main() {
  if (typeof navigator === "undefined" || !navigator.gpu) { console.log("RESULT NO_WEBGPU"); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { console.log("RESULT NO_ADAPTER"); return; }
  let info = {};
  try { info = adapter.info ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {}); } catch (e) {}
  console.log("ADAPTER " + JSON.stringify({ vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description }));
  const device = await adapter.requestDevice();

  const M = 3, N = 16, K = 64, G = 16;
  const nG = K / G;
  const rnd = mulberry32(1234);
  const Wvals = new Int32Array(N * K);
  for (let i = 0; i < N * K; i++) Wvals[i] = Math.floor(rnd() * 16);
  const scales = new Float32Array(N * nG), zeros = new Float32Array(N * nG);
  for (let i = 0; i < N * nG; i++) { scales[i] = 0.01 + rnd() * 0.2; zeros[i] = Math.floor(rnd() * 16); }
  const X = new Float32Array(M * K);
  for (let i = 0; i < M * K; i++) X[i] = rnd() * 2 - 1;

  // pack 4-bit weights -> u32, layout [N][K/8]
  const packed = new Uint32Array(N * (K / 8));
  for (let n = 0; n < N; n++)
    for (let k = 0; k < K; k++)
      packed[n * (K / 8) + (k >> 3)] |= (Wvals[n * K + k] & 15) << ((k & 7) * 4);

  // CPU reference
  const Yref = new Float32Array(M * N);
  for (let m = 0; m < M; m++)
    for (let n = 0; n < N; n++) {
      let acc = 0;
      for (let k = 0; k < K; k++) {
        const g = (k / G) | 0;
        acc += X[m * K + k] * ((Wvals[n * K + k] - zeros[n * nG + g]) * scales[n * nG + g]);
      }
      Yref[m * N + n] = acc;
    }

  const U = GPUBufferUsage;
  const mk = (arr, usage) => { const b = device.createBuffer({ size: arr.byteLength, usage }); device.queue.writeBuffer(b, 0, arr); return b; };
  const dimsBuf = mk(new Uint32Array([M, N, K, G]), U.UNIFORM | U.COPY_DST);
  const xBuf = mk(X, U.STORAGE | U.COPY_DST);
  const wBuf = mk(packed, U.STORAGE | U.COPY_DST);
  const sBuf = mk(scales, U.STORAGE | U.COPY_DST);
  const zBuf = mk(zeros, U.STORAGE | U.COPY_DST);
  const yBuf = device.createBuffer({ size: M * N * 4, usage: U.STORAGE | U.COPY_SRC });
  const readBuf = device.createBuffer({ size: M * N * 4, usage: U.COPY_DST | U.MAP_READ });

  const module = device.createShaderModule({ code: WGSL });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: dimsBuf } },
      { binding: 1, resource: { buffer: xBuf } },
      { binding: 2, resource: { buffer: wBuf } },
      { binding: 3, resource: { buffer: sBuf } },
      { binding: 4, resource: { buffer: zBuf } },
      { binding: 5, resource: { buffer: yBuf } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil((M * N) / 64));
  pass.end();
  enc.copyBufferToBuffer(yBuf, 0, readBuf, 0, M * N * 4);
  device.queue.submit([enc.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const Ygpu = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  let maxrel = 0, maxabs = 0;
  for (let i = 0; i < M * N; i++) {
    const a = Math.abs(Ygpu[i] - Yref[i]);
    const r = a / (Math.abs(Yref[i]) + 1e-6);
    if (r > maxrel) maxrel = r;
    if (a > maxabs) maxabs = a;
  }
  console.log("SAMPLE gpu[0]=" + Ygpu[0].toFixed(5) + " ref[0]=" + Yref[0].toFixed(5));
  console.log("MAXABS " + maxabs.toExponential(3) + " MAXREL " + maxrel.toExponential(3));
  console.log("RESULT " + (maxrel < 1e-3 ? "PASS" : "FAIL"));
}

await main().catch((e) => console.log("RESULT ERROR " + (e && e.message ? e.message : e)));
