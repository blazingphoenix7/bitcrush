// Autonomous logic check for the WGSL kernel (no GPU needed).
// Replicates the shader's exact 4-bit unpacking + indexing in JS and confirms it
// reproduces the reference matmul. Validates packing/indexing BEFORE we run the
// real kernel on the GPU, so the browser step only confirms hardware, not logic.
//   run:  node web/sim_check.mjs
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

const M=3,N=16,K=64,G=16,nG=K/G;
const rnd=mulberry32(1234);
const Wvals=new Int32Array(N*K); for(let i=0;i<N*K;i++)Wvals[i]=Math.floor(rnd()*16);
const scales=new Float32Array(N*nG),zeros=new Float32Array(N*nG);
for(let i=0;i<N*nG;i++){scales[i]=0.01+rnd()*0.2;zeros[i]=Math.floor(rnd()*16);}
const X=new Float32Array(M*K); for(let i=0;i<M*K;i++)X[i]=rnd()*2-1;

// pack 4-bit weights -> u32, layout [N][K/8]  (same as kernel_proof.mjs)
const packed=new Uint32Array(N*(K/8));
for(let n=0;n<N;n++)for(let k=0;k<K;k++)packed[n*(K/8)+(k>>3)]|=(Wvals[n*K+k]&15)<<((k&7)*4);

// shader simulation: reads ONLY from `packed`, exactly like the WGSL does
const Ysim=new Float32Array(M*N);
for(let idx=0;idx<M*N;idx++){
  const m=(idx/N)|0, n=idx%N, kd8=K/8, kdG=K/G; let acc=0;
  for(let k=0;k<K;k++){
    const word=packed[n*kd8+(k>>3)];
    const nib=(word>>>((k&7)*4))&15;
    const g=(k/G)|0;
    const w=(nib-zeros[n*kdG+g])*scales[n*kdG+g];
    acc+=X[m*K+k]*w;
  }
  Ysim[m*N+n]=acc;
}

// independent reference straight from the unpacked integer weights
const Yref=new Float32Array(M*N);
for(let m=0;m<M;m++)for(let n=0;n<N;n++){let acc=0;for(let k=0;k<K;k++){const g=(k/G)|0;acc+=X[m*K+k]*((Wvals[n*K+k]-zeros[n*nG+g])*scales[n*nG+g]);}Yref[m*N+n]=acc;}

let maxabs=0; for(let i=0;i<M*N;i++){const a=Math.abs(Ysim[i]-Yref[i]); if(a>maxabs)maxabs=a;}
console.log("sim vs ref  MAXABS", maxabs.toExponential(3));
console.log("packing / unpacking / indexing:", maxabs<1e-4 ? "CORRECT ✓" : "WRONG ✗");
