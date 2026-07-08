// Machine audio. Muted by default; armed by the SPEAKER switch (a user gesture, per autoplay
// policy). One quiet ambient drone runs through a real bitcrusher worklet whose bit depth follows
// the lever — the sound and the weights are being rounded to the same number of bits. UI one-shots
// (detent clicks, switch clunks, token blips) route through the same crusher.

export function createSound() {
  let ctx = null, master = null, crusher = null, bedGain = null, ready = false, armed = false;
  let lastBlip = 0, curBits = 16;

  async function init() {
    if (ctx) return;
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    master = ctx.createGain(); master.gain.value = 0;
    try {
      await ctx.audioWorklet.addModule("./crusher-worklet.js?v=1");
      crusher = new AudioWorkletNode(ctx, "crusher", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2] });
    } catch (e) { crusher = ctx.createGain(); }    // graceful: no crush, still sound
    crusher.connect(master); master.connect(ctx.destination);

    // ambient bed: warm fifth (A2+E3) + sub, slow-breathing lowpass
    bedGain = ctx.createGain(); bedGain.gain.value = 0.055;
    const lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 640; lp.Q.value = 0.8;
    const lfo = ctx.createOscillator(); lfo.frequency.value = 0.07;
    const lfoAmt = ctx.createGain(); lfoAmt.gain.value = 220;
    lfo.connect(lfoAmt); lfoAmt.connect(lp.frequency); lfo.start();
    for (const [type, freq, g] of [["triangle", 110, .5], ["triangle", 164.81, .35], ["sine", 55, .45]]) {
      const o = ctx.createOscillator(); o.type = type; o.frequency.value = freq;
      const og = ctx.createGain(); og.gain.value = g;
      o.connect(og); og.connect(lp); o.start();
    }
    lp.connect(bedGain); bedGain.connect(crusher);
    ready = true;
    setBits(curBits);
  }

  function setParam(name, v) {
    const p = crusher?.parameters?.get?.(name);
    if (p && ctx) p.setTargetAtTime(v, ctx.currentTime, 0.06);
  }
  function setBits(b) {
    curBits = b;
    if (!ready) return;
    setParam("bits", Math.max(2.5, Math.min(16, b)));
    setParam("hold", b < 6 ? Math.round(1 + (6 - b) * 3.2) : 1);   // sample-rate murder below 6 bits
  }

  function blip(freq, dur, vol, type = "square") {
    if (!armed || !ready) return;
    const t = ctx.currentTime;
    const o = ctx.createOscillator(); o.type = type; o.frequency.value = freq;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(vol, t + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g); g.connect(crusher);
    o.start(t); o.stop(t + dur + 0.03);
  }

  return {
    get armed() { return armed; },
    async arm() {
      await init();
      await ctx.resume().catch(() => {});
      armed = true;
      master.gain.setTargetAtTime(0.5, ctx.currentTime, 0.25);
    },
    disarm() {
      armed = false;
      if (ready) master.gain.setTargetAtTime(0, ctx.currentTime, 0.12);
    },
    setBits,
    detent() { blip(1500 + curBits * 60, 0.018, 0.10); },
    clunk() { blip(150, 0.05, 0.16, "square"); blip(72, 0.08, 0.13, "sine"); },
    button() { blip(420, 0.035, 0.12); blip(210, 0.06, 0.08, "sine"); },
    token(coh) {                                     // pitch sinks as the mind goes
      const now = performance.now();
      if (now - lastBlip < 70) return;
      lastBlip = now;
      blip(190 + coh * 540, 0.014, 0.05, "triangle");
    },
    alarm() { blip(96, 0.35, 0.16, "sawtooth"); },
  };
}
