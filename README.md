# BITCRUSH BC-06

**A real language model runs in your browser. A lever lobotomizes it, live.**

<!-- demo GIF goes here before launch -->

**🔗 Try it: [blazingphoenix7.github.io/bitcrush](https://blazingphoenix7.github.io/bitcrush/)** — needs a WebGPU browser (Chrome/Edge desktop) and ~1.2 GB of patience on first load.

Slam the lever from **16 bits down to 2** and every one of the model's 196 weight matrices is re-quantized **in real time, on your GPU**, while it answers *your* prompt. At 16 bits it gives solid advice. At 4 it's fine — that's the twist. At 3 it slurs and loops. At 2, the lights are on but nobody's home.

Nothing is staged. The text is real greedy decoding from a real 596-million-parameter transformer (**Qwen3-0.6B**) running through a **hand-written WebGPU engine** — no PyTorch, no ONNX, no WebLLM, no server. The machine is the fiction; the model is the fact.

---

## The machine

| Instrument | What it actually measures |
|---|---|
| **THE LEVER** | Bits per weight, 16 → 2. The scale is warped: the bottom half of the throw is 4 → 2 bits, because that's where minds end. The red band is labeled LOBOTOMY. |
| **CRT** | The model answering you, token by token. Each token is tinted by its real softmax entropy — watch confidence cool from phosphor green to red. |
| **COHERENCE (VU)** | 1 − normalized mean entropy of the generation. A confused mind pins the needle left. |
| **CORTEX ACTIVITY (EKG)** | Per-token entropy as a strip chart. Calm wave = fluent; jagged spikes = word salad. |
| **BRAIN SIZE** | Actual bytes the weights would occupy at this precision. 1,192 MB at fp16 → 421 MB at 2-bit. |
| **WEIGHT SCOPE** | 8,192 real weights from layer 0's query projection, live-histogrammed onto the 2^bits representable values. Amber mass below the baseline = literally what rounding threw away. |
| **GHOST SCAN** | After a crushed answer, the fp16 model re-reads the *same emitted sequence* (teacher-forced) and every token where it disagrees gets underlined — hover to see the word the intact mind would have chosen. |
| **NAIVE / SMART** | Same bit budget, spent badly or well: one scale per output channel vs one per 64-weight group. At 3 bits, SMART is coherent and NAIVE says "and and and and". This is the entire intuition behind modern quantization, in one switch. |
| **SPEAKER** | An AudioWorklet bitcrusher — the *same operation* (amplitude quantization + sample-and-hold) applied to sound, driven by the same lever. |

Also: a warning tape that dares you to click it, a boot-up self-test, screws that spin, and a QA stamp. Everything on the panel is a real signal or a real toy. **The model's output is never faked, styled, or pre-rendered.**

## Under the hood

The point of Bitcrush is that *runtime* re-quantization — dragging precision like a fader — isn't something existing runtimes do. So the engine is built from scratch:

- **Custom WGSL compute kernels**: embedding, RMSNorm, matmul, fused **quantize-dequantize matmul** (`matmul_q` — group-wise affine RTN computed inline as weights are read, so changing precision costs one uniform, not a re-upload), per-group scale/zero precompute, per-head QK-norm + RoPE, grouped-query attention with a persistent KV cache, SwiGLU, tied-embedding LM head, and a row-range LM head for the ghost pass.
- **Correctness is proven, not vibed**: the forward pass matches PyTorch to `maxAbs 9e-5` on logits; KV-cache greedy decoding reproduces PyTorch's `generate()` token-for-token; 8-bit quantization is bit-for-bit indistinguishable from fp16 argmax. The harness ships in the repo (`web/engine-test-qwen3.html`, plus browser-free Node oracles in `quant/`).
- **Custom byte-level BPE tokenizer** in ~100 lines of JS, exactly matching Qwen's pre-tokenizer regex and merges.
- **Zero dependencies, zero build step.** The whole front-end is vanilla ES modules. View-source works.
- The **ghost** is teacher-forced — the fp16 pass replays the crushed model's actual tokens, so divergences are honest counterfactuals, not two runs drifting apart.
- Weights ship as one fp16 binary (~1.2 GB, cached by the browser after first load) with a JSON manifest.

## Run it locally

```bash
git clone https://github.com/blazingphoenix7/bitcrush
cd bitcrush/web
python serve.py        # threaded static server → http://localhost:8123/
```

Weights auto-fetch from Hugging Face on first load (or drop a local export into `web/weights-qwen3/`). It runs on an Intel iGPU — a discrete GPU is just faster.

Regenerating the model export (optional, needs Python + PyTorch):

```bash
cd quant && python export_qwen3i.py   # downloads Qwen3-0.6B, writes web/weights-qwen3/
```

## Honest notes

- **Is the degradation real?** Yes. Group-wise round-to-nearest quantization of all attention/MLP weights (embeddings and norms stay fp16, as in practice). Greedy decoding, so every run at the same settings reproduces exactly.
- **Why does 4-bit sound fine?** Because it is. That's the actual state of the art's dirty secret, and the reason the interesting part of the lever is 3 → 2.
- **"SMART" is not GPTQ.** It's group-wise RTN (the same family as `Q4_0`-style formats). Percentile-spaced levels in the scope illustrate the NF4-style intuition. No claims beyond what's implemented.
- **Coherence** is 1 − normalized mean token entropy: a live, honest confidence signal, not a language-quality benchmark.

## Roadmap

The engine is a platform. Next explorables, each its own release: sampling playground (temperature/top-k live), attention flow, speculative decoding races, logit lens.

## License

Code: MIT. Model weights: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), Apache-2.0, © Alibaba Cloud.
