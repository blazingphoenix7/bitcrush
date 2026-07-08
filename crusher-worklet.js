// AudioWorklet bitcrusher — the literal same operation as the weight quantizer:
// amplitude quantization (round to 2^(bits-1) levels) + sample-and-hold rate reduction.
class Crusher extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: "bits", defaultValue: 16, minValue: 1, maxValue: 16, automationRate: "k-rate" },
      { name: "hold", defaultValue: 1, minValue: 1, maxValue: 32, automationRate: "k-rate" },
    ];
  }
  constructor() { super(); this.phase = 0; this.held = [0, 0]; }
  process(inputs, outputs, params) {
    const inp = inputs[0], out = outputs[0];
    if (!inp || !inp.length) return true;
    const bits = Math.max(1, params.bits[0]);
    const hold = Math.max(1, Math.round(params.hold[0]));
    const L = Math.pow(2, bits - 1);
    for (let ch = 0; ch < out.length; ch++) {
      const i = inp[ch] || inp[0], o = out[ch];
      for (let n = 0; n < o.length; n++) {
        if ((this.phase + n) % hold === 0) this.held[ch] = i[n];
        o[n] = Math.round(this.held[ch] * L) / L;
      }
    }
    this.phase += out[0].length;
    return true;
  }
}
registerProcessor("crusher", Crusher);
