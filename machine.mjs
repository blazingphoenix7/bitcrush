// BITCRUSH BC-06 — machine controller.
// One lever drives everything: the WGSL weight re-quantizer, the 7-seg, the scope, the wordmark
// pixels, the LEDs, the sound. The text on the CRT is REAL inference — never faked, never styled
// into fake degradation. The machine is the fiction; the model is the fact.
import { loadModel, generate, setQuant, quantNames, teacherForce, sampleWeights, argmax, argmaxMasked } from "./qwen3.mjs?v=18";
import { loadTokenizer } from "./qwen3-tok.mjs?v=1";
import { createSeg7 } from "./seg7.mjs?v=1";
import { createPixelmark } from "./pixelmark.mjs?v=2";
import { createScope } from "./scope.mjs?v=1";
import { createEkg } from "./ekg.mjs?v=1";
import { createSound } from "./sound.mjs?v=1";

const $ = (id) => document.getElementById(id);
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const lerp = (a, b, t) => a + (b - a) * t;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ── constants ────────────────────────────────────────────────────────────── */
const BMIN = 2, BMAX = 16, EXP = 2.6;            // lever gamma: the bottom half is 4→2 bits
const N_TOKENS = 44, MIN_NEW = 12;
const SMART_GROUP = 64;
const ENT_LO = 1.6, ENT_HI = 6.5;
// Qwen3 chat template (thinking off), with a system message anchoring English:
// <|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
// Composed at boot via the tokenizer (verified identical to transformers' apply_chat_template).
const SYS_PROMPT = "You are a helpful assistant. Always respond in English.";
let CHAT_PREFIX = [];                             // built once the tokenizer is up
const CHAT_SUFFIX = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271];
const EOS = [151645];
// weights: local single-file copy in dev; on the deployed site, same-origin 95 MB parts on the
// gh-pages branch (GitHub Pages caps files at 100 MB; release assets lack CORS for browser fetch)
const WEIGHTS_LOCAL = "./weights-qwen3/";
const WEIGHTS_REMOTE = "./weights-remote/";
async function weightsBase() {
  try { const r = await fetch(WEIGHTS_LOCAL + "manifest.json", { method: "HEAD" }); if (r.ok) return WEIGHTS_LOCAL; } catch {}
  return WEIGHTS_REMOTE;
}

const fracToBits = (f) => BMIN + (BMAX - BMIN) * Math.pow(1 - clamp(f, 0, 1), EXP);
const bitsToFrac = (b) => 1 - Math.pow((clamp(b, BMIN, BMAX) - BMIN) / (BMAX - BMIN), 1 / EXP);

/* ── state ────────────────────────────────────────────────────────────────── */
let model = null, tok = null, ready = false, busy = false;
let bits = BMAX;                                  // live lever value (float)
let targetDetent = BMAX;                          // integer the lever is parked on
let committedBits = BMAX;                         // precision of the last generation
let mode = "smart";
let promptText = "", promptIds = [];
let genCtl = null, commitTimer = 0, ghostTimer = 0;
let quantParams = 0, embedBytes = 0, size16 = 1;
let lastMsPerTok = 0, wasCrushedTo2 = false;
let engLock = true, banMask = null;               // ENG LOCK: decode-time ban on non-Latin tokens (a labeled switch, not a silent filter)
let touchedLever = false, firstAnswerDone = false; // for the one-time "grab me" pulse on the cap
const cache = new Map();                          // key → { tokens, ents }
const ghostCache = new Map();                     // key → [{i, word}]
const sound = createSound();

// Tokens allowed under ENG LOCK: ASCII + Latin-1/Extended letters + common typography.
// Anything else (CJK, Cyrillic, Arabic, kana, emoji, stray multi-byte fragments) is banned at argmax.
function buildLatinBanMask(vocabSize) {
  const banned = /[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF\u2000-\u206F]/u;
  const m = new Uint8Array(vocabSize);
  for (let id = 0; id < vocabSize; id++) {
    const s = tok.decodeOne(id);
    if (s && banned.test(s)) m[id] = 1;           // ids that decode to "" (special ids, incl. EOS) stay allowed
  }
  return m;
}

let seg, mark, scope, ekg;

/* ── entropy → color / coherence ──────────────────────────────────────────── */
function entropyOf(logits) {
  let mx = -Infinity; for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  let sum = 0; for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - mx);
  const inv = 1 / sum; let H = 0;
  for (let i = 0; i < logits.length; i++) { const p = Math.exp(logits[i] - mx) * inv; if (p > 1e-9) H -= p * Math.log(p); }
  return H;
}
const cohOf = (meanEnt) => clamp(1 - (meanEnt - ENT_LO) / (ENT_HI - ENT_LO), 0, 1);
const hexLerp = (a, b, t) => {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  return "#" + pa.map((v, i) => Math.round(lerp(v, pb[i], t)).toString(16).padStart(2, "0")).join("");
};
function tokenColor(ent) {                        // phosphor → amber → red as confidence dies
  const heat = clamp((ent - ENT_LO) / (ENT_HI - ENT_LO), 0, 1);
  return heat < 0.5 ? hexLerp("#7dffa0", "#ffd36e", heat * 2) : hexLerp("#ffd36e", "#ff5340", (heat - 0.5) * 2);
}
const STATES = [[.78, "lucid", "LUCID"], [.5, "slurring", "SLURRING"], [.28, "salad", "WORD SALAD"], [-1, "static", "STATIC"]];
const stateFor = (coh) => STATES.find(([t]) => coh >= t);

/* ── displays ─────────────────────────────────────────────────────────────── */
const sizeAtBits = (b) => quantParams * (b >= BMAX ? 16 : b) / 8 + embedBytes;

function setLever(v, opts = {}) {
  bits = clamp(v, BMIN, BMAX);
  const pct = (bitsToFrac(bits) * 100).toFixed(2) + "%";
  $("cap").style.setProperty("--p", pct);
  seg?.set(bits.toFixed(1).padStart(4, " "));
  scope?.update(bits, mode);
  updateScopeInfo();
  mark?.corrupt(clamp((4.5 - bits) / 2.5, 0, 1));  // pristine ≥4.5 bits; wrecked at 2 — damage tracks the story
  sound.setBits(bits);
  // brain size
  if (size16 > 1) {
    const mb = sizeAtBits(bits) / 1e6, ratio = sizeAtBits(bits) / size16;
    $("mbNum").textContent = String(Math.round(mb)).padStart(4, " ");
    const leds = $("ledbar").children, lit = clamp(Math.round(ratio * leds.length), 1, leds.length);
    for (let i = 0; i < leds.length; i++) leds[i].classList.toggle("on", i < lit);
  }
  $("crt").classList.toggle("flicker", !reducedMotion && bits <= 3.2);
  const lever = $("lever");
  lever.setAttribute("aria-valuenow", String(Math.round(bits)));
  lever.setAttribute("aria-valuetext", `${Math.round(bits)} bits per weight`);
  if (bits >= 4) restoreTape();
}
function updateScopeInfo() {
  if (!scope) return;
  const { levels, step } = scope.info();
  $("scopeInfo").textContent = levels > 4096 ? "65,536 LEVELS" : `${levels.toLocaleString()} LEVELS · STEP ${step.toPrecision(2)}`;
}
function setHealth(coh) {
  const [, key, label] = stateFor(coh);
  $("needle").style.transform = `rotate(${(-55 + clamp(coh, 0, 1) * 110).toFixed(1)}deg)`;
  $("cohNum").textContent = String(Math.round(coh * 100));
  for (const li of $("statusLeds").children) li.classList.toggle("on", li.dataset.s === key);
  $("stState").textContent = label;
}

/* ── generation ───────────────────────────────────────────────────────────── */
const promptKey = () => promptIds.join(",");
const curKey = () => `${promptKey()}|${committedBits}|${mode}|${engLock ? "L" : "U"}`;

function reencodePrompt() {
  promptText = $("prompt").value.trim() || "Give me a quick pep talk for my job interview tomorrow.";
  promptIds = [...CHAT_PREFIX, ...tok.encode(promptText), ...CHAT_SUFFIX];
  $("qEcho").textContent = "> " + promptText;
}

function spanFor(id, ent) {
  const s = document.createElement("span");
  s.className = "tok";
  s.textContent = tok.decodeOne(id);
  s.style.color = tokenColor(ent);
  return s;
}

async function commit(bInt) {
  if (!ready) return;
  bInt = clamp(Math.round(bInt), BMIN, BMAX);
  committedBits = bInt;
  const key = curKey();
  genCtl?.abort(); genCtl = new AbortController();
  const signal = genCtl.signal;
  clearTimeout(ghostTimer);
  $("ghostLine").textContent = "";

  // the 2-bit moment — one confident exclamation, once per descent
  if (bInt === 2 && !wasCrushedTo2) {
    wasCrushedTo2 = true;
    if (!reducedMotion) {
      $("crt").classList.add("tear"); setTimeout(() => $("crt").classList.remove("tear"), 520);
      $("device").classList.add("rattle"); setTimeout(() => $("device").classList.remove("rattle"), 520);
    }
    sound.alarm();
  } else if (bInt > 2) wasCrushedTo2 = false;

  if (cache.has(key)) {                            // greedy is deterministic → replay instantly
    const entry = cache.get(key);
    renderSequence(entry);
    $("stTok").textContent = "FROM CACHE";
    $("cursor").hidden = true;
    if (ghostCache.has(key)) applyGhost(key); else scheduleGhost(key, entry.tokens);
    return;
  }

  busy = true;
  setQuant(model, bInt, mode === "smart" ? SMART_GROUP : 0);
  const stream = $("outStream");
  stream.replaceChildren();
  ekg.reset();
  $("cursor").hidden = false;
  $("stTok").textContent = "THINKING";
  const tokens = [], ents = [];
  let entSum = 0, ghostArgs = null;
  const t0 = performance.now();
  try {
    await generate(model, promptIds, N_TOKENS, {
      signal, eosIds: EOS, minNew: MIN_NEW, banMask: engLock ? banMask : null,
      onToken: (id, gen, logits) => {
        if (signal.aborted) return;
        const ent = entropyOf(logits);
        tokens.push(id); ents.push(ent); entSum += ent;
        stream.appendChild(spanFor(id, ent));
        ekg.push(ent);
        const coh = cohOf(entSum / tokens.length);
        setHealth(coh);
        sound.token(coh);
        flickLed("ledThink");
        $("stTok").textContent = `THINKING · ${tokens.length} TOK`;
        $("crtInner").scrollTop = $("crtInner").scrollHeight;
      },
    });
    if (!signal.aborted) {
      const ms = performance.now() - t0;
      lastMsPerTok = Math.round(ms / Math.max(tokens.length, 1));
      cache.set(key, { tokens, ents });
      $("stTok").textContent = `${tokens.length} TOK · ${lastMsPerTok} MS/TOK`;
      $("cursor").hidden = true;
      $("perfNote").textContent = `measured live: ${lastMsPerTok} ms/token on your GPU`;
      $("srLive").textContent = `Answer at ${bInt} bits: ${tok.decode(tokens)}`;
      ghostArgs = [key, tokens];
      // one-time invitation: if the first answer lands and the lever's never been touched, pulse the cap
      if (!firstAnswerDone) {
        firstAnswerDone = true;
        if (!reducedMotion && !document.hidden) setTimeout(() => { if (!touchedLever) $("cap").classList.add("beckon"); }, 3200);
      }
    }
  } catch (e) {
    if (!signal.aborted) { $("stTok").textContent = "ERROR"; console.error(e); }
  } finally {
    if (genCtl.signal === signal) busy = false;   // an aborted run must not clear the successor's flag
  }
  // ghost scan happens AFTER busy clears — a direct call inside the try would veto itself
  if (ghostArgs && genCtl.signal === signal) scheduleGhost(...ghostArgs);
}

function renderSequence({ tokens, ents }) {
  const stream = $("outStream");
  stream.replaceChildren();
  ekg.reset();
  let entSum = 0;
  for (let i = 0; i < tokens.length; i++) { stream.appendChild(spanFor(tokens[i], ents[i])); ekg.push(ents[i]); entSum += ents[i]; }
  setHealth(cohOf(entSum / Math.max(tokens.length, 1)));
  $("crtInner").scrollTop = $("crtInner").scrollHeight;
}

function scheduleCommit(bInt) { clearTimeout(commitTimer); commitTimer = setTimeout(() => commit(bInt), 90); }

function flickLed(id) {
  const el = $(id);
  el.classList.add("lit");
  clearTimeout(el._t); el._t = setTimeout(() => el.classList.remove("lit"), 150);
}

/* ── the ghost: what the full-precision mind would have said ──────────────── */
async function runGhost(key, tokens) {
  if (busy || key !== curKey() || ghostCache.has(key)) return;
  $("ledGhost").classList.add("lit");
  try {
    const all = await teacherForce(model, promptIds.concat(tokens), promptIds.length - 1);
    const divs = [];
    const lock = key.endsWith("|L");               // the ghost plays by the same decode rules as the run it haunts
    for (let i = 0; i < tokens.length; i++) {
      const meant = lock ? argmaxMasked(all[i], banMask, null) : argmax(all[i]);
      if (meant !== tokens[i]) divs.push({ i, word: tok.decodeOne(meant) });
    }
    ghostCache.set(key, divs);
    if (key === curKey()) applyGhost(key);
  } catch (e) { console.warn("ghost scan failed", e); }
  $("ledGhost").classList.remove("lit");
}
function scheduleGhost(key, tokens) {
  if (committedBits >= BMAX) return;
  if (ghostCache.has(key)) { applyGhost(key); return; }
  clearTimeout(ghostTimer);
  // hidden tabs clamp even 0ms timers to ~1/min — run directly; the 1.7s courtesy is for visible users mid-play
  if (document.hidden) runGhost(key, tokens);
  else ghostTimer = setTimeout(() => runGhost(key, tokens), 1700);
}
function applyGhost(key) {
  const divs = ghostCache.get(key);
  if (!divs) return;
  const spans = $("outStream").children;
  for (const d of divs) {
    const s = spans[d.i]; if (!s) continue;
    s.classList.add("div");
    s.dataset.meant = (d.word.trim() || "␣").toUpperCase();
    s.tabIndex = 0;
  }
  $("ghostLine").textContent = divs.length
    ? `GHOST SCAN — the 16-bit mind disagreed at ${divs.length}/${spans.length} tokens. Hover the underlined ones.`
    : `GHOST SCAN — identical to the 16-bit mind. Nothing lost. Yet.`;
}

/* ── lever ────────────────────────────────────────────────────────────────── */
function buildScale() {
  const wrap = $("leverScale");
  wrap.replaceChildren();
  const majors = new Set([16, 8, 4, 3, 2]);
  for (let b = BMIN; b <= BMAX; b++) {
    const p = (bitsToFrac(b) * 100).toFixed(2) + "%";
    const t = document.createElement("div");
    t.className = "tick" + (majors.has(b) ? " major" : "");
    t.style.setProperty("--p", p);
    wrap.appendChild(t);
    if (majors.has(b)) {
      const n = document.createElement("span");
      n.className = "tick-num" + (b <= 3 ? " hot" : "");
      // clamp the endpoint numerals so 16 and 2 don't clip past the track ends
      n.style.setProperty("--p", clamp(bitsToFrac(b) * 100, 2.5, 97.5).toFixed(2) + "%");
      n.textContent = b;
      wrap.appendChild(n);
    }
  }
  $("lever").style.setProperty("--rz", (bitsToFrac(4) * 100).toFixed(2) + "%");
}
function leverFrac(e) {
  const r = $("lever").getBoundingClientRect();
  return r.width > r.height ? clamp((e.clientX - r.left) / r.width, 0, 1)
                            : clamp((e.clientY - r.top) / r.height, 0, 1);
}
let dragging = false, lastDetent = BMAX;
function onDown(e) {
  if (!ready) return;
  e.preventDefault();                              // don't start a text selection under the drag
  touchedLever = true; $("cap").classList.remove("beckon");
  document.body.classList.add("dragging");
  const lever = $("lever");
  lever.focus({ preventScroll: true });            // preventDefault suppressed click-focus; keep arrows working
  dragging = true;
  lever.classList.add("grabbing");
  try { lever.setPointerCapture(e.pointerId); } catch {}
  onMove(e);
}
function onMove(e) {
  if (!dragging) return;
  const v = fracToBits(leverFrac(e));
  setLever(v);
  const d = Math.round(v);
  if (d !== lastDetent) { lastDetent = d; sound.detent(); navigator.vibrate?.(5); }
}
function onUp() {
  document.body.classList.remove("dragging");
  if (!dragging) return;
  dragging = false;
  $("lever").classList.remove("grabbing");
  targetDetent = clamp(Math.round(bits), BMIN, BMAX);
  setLever(targetDetent);
  scheduleCommit(targetDetent);
}
function nudge(d) {
  if (!ready) return;
  touchedLever = true; $("cap").classList.remove("beckon");
  targetDetent = clamp(targetDetent + d, BMIN, BMAX);
  setLever(targetDetent);
  sound.detent(); navigator.vibrate?.(5);
  scheduleCommit(targetDetent);
}
function slamTo(b) {
  if (!ready) return;
  targetDetent = clamp(Math.round(b), BMIN, BMAX);
  setLever(targetDetent);
  clearTimeout(commitTimer);
  commit(targetDetent);                            // direct — never gate a slam on a throttleable timer
}

/* ── the dare ─────────────────────────────────────────────────────────────── */
let tapeTripped = false;
function setTape(line1, line2) {
  $("tape").replaceChildren(document.createTextNode(line1), document.createElement("br"), document.createTextNode(line2));
}
function restoreTape() {
  if (!tapeTripped) return;
  tapeTripped = false;
  setTape("DO NOT GO", "BELOW 3 ⚠");
}

/* ── boot ─────────────────────────────────────────────────────────────────── */
const bootQ = [];
let bootBusy = false;
function bootLine(text, cls) {
  return new Promise((res) => {
    bootQ.push({ text, cls, res });
    pumpBoot();
  });
}
function pumpBoot() {
  if (bootBusy || !bootQ.length) return;
  bootBusy = true;
  const { text, cls, res } = bootQ.shift();
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.textContent = text;
  $("bootlog").appendChild(div);
  $("crtInner").scrollTop = $("crtInner").scrollHeight;
  // stagger is cosmetic ONLY — hidden tabs throttle timers to ~1/min, so never let it gate progress
  if (reducedMotion || document.hidden) { bootBusy = false; res(div); queueMicrotask(pumpBoot); }
  else setTimeout(() => { bootBusy = false; res(div); pumpBoot(); }, 90);
}
const bar = (f) => "▓".repeat(Math.round(f * 12)).padEnd(12, "░");

async function boot() {
  seg = createSeg7($("seg"), { cells: 3, dpAfter: 2 });
  seg.set("--.-".replace(".", ""));               // dashes while booting
  mark = createPixelmark($("pixelmark"), "BITCRUSH", 4);
  ekg = createEkg($("ekg"));
  const lb = $("ledbar");
  for (let i = 0; i < 12; i++) lb.appendChild(document.createElement("i"));

  bootLine("BITCRUSH BC-06 · NEURAL COMPRESSION UNIT");
  bootLine("FIRMWARE 1.0 · POWER-ON SELF-TEST", "");
  if (new URLSearchParams(location.search).has("nomodel")) {   // layout-QA mode: full chrome, no 1.2 GB subject
    bootLine("  TEST MODE — SUBJECT NOT LOADED", "fail");
    setLever(3);                                   // park the cap somewhere meaningful for geometry checks
    setHealth(0.62);
    return;
  }
  if (!navigator.gpu) {
    bootLine("  WEBGPU ................. FAIL", "fail");
    bootLine("");
    bootLine("THIS MACHINE NEEDS WEBGPU.", "fail");
    bootLine("USE CHROME OR EDGE ON A DESKTOP — THE WHOLE POINT");
    bootLine("IS A REAL MODEL ON *YOUR* GPU. NO SERVER TO FALL BACK TO.");
    $("stState").textContent = "NO GPU";
    return;
  }
  bootLine("  WEBGPU ................. OK", "ok");
  // POST: sweep the status LEDs once, like real gear checking its lamps (cosmetic — never gates)
  if (!reducedMotion && !document.hidden) {
    const lis = [...$("statusLeds").children];
    lis.forEach((li, i) => {
      setTimeout(() => li.classList.add("on"), 240 + i * 130);
      setTimeout(() => li.classList.remove("on"), 240 + i * 130 + 160);
    });
  }
  const wBase = await weightsBase();
  let dlLine = null, dlPending = false;
  try {
    model = await loadModel(wBase, {
      log: () => {},
      onProgress: (f) => {
        if (!dlLine) {                             // guard: progress fires faster than the line lands
          if (!dlPending) { dlPending = true; bootLine("  DOWNLOADING SUBJECT .... " + bar(0)).then((d) => (dlLine = d)); }
          return;
        }
        dlLine.textContent = `  DOWNLOADING SUBJECT .... ${bar(f)} ${Math.round(f * 1192)}/1192 MB`;
      },
    });
  } catch (e) {
    bootLine("  SUBJECT DOWNLOAD ....... FAIL — " + (e.message || e), "fail");
    $("stState").textContent = "FAULT";
    return;
  }
  if (dlLine) dlLine.textContent = "  DOWNLOADING SUBJECT .... " + bar(1) + " 1192/1192 MB";
  bootLine("  MOUNTING 596,049,920 PARAMETERS ... OK", "ok");
  tok = await loadTokenizer("./tok-qwen3/");
  bootLine("  TOKENIZER · 151,643 ENTRIES ....... OK", "ok");
  // chat template with the English-anchoring system message (verified == transformers' output)
  CHAT_PREFIX = [151644, ...tok.encode("system\n" + SYS_PROMPT), 151645, 198, 151644, ...tok.encode("user\n")];
  banMask = buildLatinBanMask(model.cfg.vocab_size);
  bootLine("  LANGUAGE LOCK .......... ENGAGED", "ok");

  for (const n of quantNames(model.cfg)) { const [N, K] = model.W[n].shape; quantParams += N * K; }
  embedBytes = model.W["model.embed_tokens.weight"].shape.reduce((a, b) => a * b, 1) * 2;
  size16 = sizeAtBits(BMAX);

  const sample = await sampleWeights(model, "model.layers.0.self_attn.q_proj.weight", 8192);
  scope = createScope($("scope"), sample);
  bootLine("  WEIGHT SCOPE · 8,192 SAMPLES ...... OK", "ok");
  await bootLine("SUBJECT IS AWAKE.", "ok");

  // hash state: #b=3&m=naive&q=...
  try {
    const h = new URLSearchParams(location.hash.slice(1));
    if (h.get("q")) $("prompt").value = h.get("q").slice(0, 160);
    if (h.get("m") === "naive") setMode("naive", true);
    if (h.get("l") === "0") { engLock = false; $("swLock").setAttribute("aria-pressed", "false"); }
    const hb = parseInt(h.get("b") || "", 10);
    if (hb >= BMIN && hb <= BMAX) targetDetent = hb;
  } catch {}

  ready = true;
  $("lever").classList.remove("locked");
  reencodePrompt();
  setLever(targetDetent);
  setHealth(1);
  const awake = () => {
    $("bootlog").hidden = true;
    $("chat").hidden = false;
    commit(targetDetent);
  };
  if (reducedMotion || document.hidden) awake();
  else setTimeout(awake, 420);
}

/* ── mode / inputs / toys ─────────────────────────────────────────────────── */
function setMode(m, silent) {
  mode = m;
  const sw = $("modeSwitch");
  sw.dataset.mode = m;
  sw.setAttribute("aria-pressed", String(m === "smart"));
  sw.setAttribute("aria-label", "Quantization mode: " + m);
  scope?.update(bits, mode);
  if (!silent) { sound.clunk(); commit(committedBits); }
}
function ask() {
  if (!ready) return;
  reencodePrompt();
  sound.button();
  const btn = $("askBtn");
  btn.classList.add("pressed"); setTimeout(() => btn.classList.remove("pressed"), 130);
  commit(committedBits);
}

function wire() {
  const lever = $("lever");
  lever.classList.add("locked");
  lever.addEventListener("pointerdown", onDown);
  lever.addEventListener("pointermove", onMove);
  lever.addEventListener("pointerup", onUp);
  lever.addEventListener("pointercancel", onUp);
  lever.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowLeft") { e.preventDefault(); nudge(-1); }
    else if (e.key === "ArrowUp" || e.key === "ArrowRight") { e.preventDefault(); nudge(1); }
    else if (e.key === "Home") { e.preventDefault(); slamTo(BMAX); }
    else if (e.key === "End") { e.preventDefault(); slamTo(BMIN); }
  });

  $("modeSwitch").addEventListener("click", () => setMode(mode === "smart" ? "naive" : "smart"));
  $("askBtn").addEventListener("click", ask);
  $("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); ask(); } });
  for (const chip of document.querySelectorAll(".chip[data-q]"))
    chip.addEventListener("click", () => { $("prompt").value = chip.dataset.q; ask(); });

  $("tape").addEventListener("click", () => {
    if (!ready) return;
    sound.alarm();
    tapeTripped = true;
    setTape("TOLD", "YOU. ☠");
    slamTo(2);
  });

  $("swLock").addEventListener("click", (e) => {
    engLock = e.currentTarget.getAttribute("aria-pressed") !== "true";
    e.currentTarget.setAttribute("aria-pressed", String(engLock));
    sound.clunk();
    if (ready) commit(committedBits);              // decode rules changed → same settings, new run
  });
  $("swSound").addEventListener("click", async (e) => {
    const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
    e.currentTarget.setAttribute("aria-pressed", String(on));
    if (on) { await sound.arm(); sound.clunk(); } else sound.disarm();
  });
  $("swLights").addEventListener("click", (e) => {
    const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
    e.currentTarget.setAttribute("aria-pressed", String(on));
    document.body.classList.toggle("dark", !on);
    sound.clunk();
  });

  for (const s of document.querySelectorAll(".screw"))
    s.addEventListener("click", () => { s._r = (s._r || 0) + 90; s.style.transform = `rotate(${s._r}deg)`; });
  $("stamp").addEventListener("click", () => {
    const st = $("stamp"); st._r = (st._r || 0) + 360;
    st.style.transform = `rotate(${st._r - 11}deg)`;
  });

  $("copyLink").addEventListener("click", async () => {
    const h = `#b=${committedBits}&m=${mode}${engLock ? "" : "&l=0"}&q=${encodeURIComponent($("prompt").value.trim())}`;
    history.replaceState(null, "", h);              // the URL is shareable even if the clipboard says no
    const c = $("copyLink");
    try {
      await navigator.clipboard.writeText(location.origin + location.pathname + h);
      c.textContent = "COPIED ✓";
    } catch { c.textContent = "LINK IN URL BAR"; }
    setTimeout(() => (c.textContent = "COPY LINK"), 1400);
  });

  // idle heartbeat on the EKG (visible tab only; harmless if throttled)
  if (!reducedMotion) {
    let t = 0;
    setInterval(() => { if (document.visibilityState === "visible") ekg?.idleBeat((t = (t + 0.17) % 1)); }, 1900);
  }
}

/* ── go ───────────────────────────────────────────────────────────────────── */
buildScale();
wire();
boot();

// test hooks (used by the dev harness; harmless in prod)
window.__bc = {
  commit, slamTo, nudge, setMode,
  get bits() { return bits; }, get committed() { return committedBits; }, get mode() { return mode; },
  get ready() { return ready; }, get engLock() { return engLock; }, get banMask() { return banMask; },
  cache, ghostCache,
  get model() { return model; }, get promptIds() { return promptIds; },
  teacherForce, sampleWeights,
};
