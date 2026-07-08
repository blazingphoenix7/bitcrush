// Qwen3 byte-level BPE tokenizer (encode/decode) for the browser.
// Same byte-level scheme as GPT-2, but Qwen's pre-tokenizer regex: case-insensitive contractions,
// numbers split per-digit (\p{N}, not \p{N}+), no automatic prefix space. Base model → no BOS.
const SEP = " ";
// Translated from tok-qwen3/pretokenizer.json. Global `i` flag folds the (?i:) contraction group;
// it doesn't affect the \p{L}/\p{N} classes, so it's safe to apply to the whole pattern.
const PAT = /'s|'t|'re|'ve|'m|'ll|'d|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu;

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const enc = {}, dec = {};
  for (let i = 0; i < bs.length; i++) { const ch = String.fromCharCode(cs[i]); enc[bs[i]] = ch; dec[ch] = bs[i]; }
  return { enc, dec };
}

function getPairs(word) {
  const s = new Set();
  for (let i = 0; i < word.length - 1; i++) s.add(word[i] + SEP + word[i + 1]);
  return s;
}

// Pure constructor from already-loaded vocab (token→id map) + merges text. No I/O → unit-testable.
export function createTokenizer(vocab, mergesText) {
  const bpeRanks = {};
  mergesText.split(/\r?\n/).forEach((line, i) => {
    if (!line || line.startsWith("#")) return;         // skip a possible version header
    bpeRanks[line.replace(" ", SEP)] = i;
  });
  const decoder = {};
  for (const [tok, id] of Object.entries(vocab)) decoder[id] = tok;
  const { enc: byteEncoder, dec: byteDecoder } = bytesToUnicode();
  const textEnc = new TextEncoder(), textDec = new TextDecoder("utf-8");
  const cache = new Map();

  function bpe(token) {
    if (cache.has(token)) return cache.get(token);
    let word = Array.from(token);
    let pairs = getPairs(word);
    if (pairs.size === 0) { cache.set(token, token); return token; }
    while (true) {
      let minRank = Infinity, bigram = null;
      for (const p of pairs) { const r = bpeRanks[p]; if (r !== undefined && r < minRank) { minRank = r; bigram = p; } }
      if (bigram === null) break;
      const sepIdx = bigram.indexOf(SEP);
      const first = bigram.slice(0, sepIdx), second = bigram.slice(sepIdx + 1);
      const nw = [];
      let i = 0;
      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) { for (let k = i; k < word.length; k++) nw.push(word[k]); break; }
        for (let k = i; k < j; k++) nw.push(word[k]);
        i = j;
        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) { nw.push(first + second); i += 2; }
        else { nw.push(word[i]); i += 1; }
      }
      word = nw;
      if (word.length === 1) break;
      pairs = getPairs(word);
    }
    const out = word.join(" ");
    cache.set(token, out);
    return out;
  }

  function encode(text) {
    const ids = [];
    for (const m of text.matchAll(PAT)) {
      let token = "";
      for (const b of textEnc.encode(m[0])) token += byteEncoder[b];
      for (const t of bpe(token).split(" ")) { const id = vocab[t]; if (id !== undefined) ids.push(id); }
    }
    return ids;
  }

  function decode(ids) {
    let text = "";
    for (const id of ids) text += (decoder[id] ?? "");
    const bytes = [];
    for (const ch of text) { const b = byteDecoder[ch]; if (b !== undefined) bytes.push(b); }
    return textDec.decode(new Uint8Array(bytes));
  }

  // decode a single id to its display string (for streaming one token at a time)
  function decodeOne(id) { return decode([id]); }

  return { encode, decode, decodeOne, vocabSize: Object.keys(vocab).length };
}

export async function loadTokenizer(baseUrl) {
  const vocab = await (await fetch(baseUrl + "vocab.json")).json();
  const mergesText = await (await fetch(baseUrl + "merges.txt")).text();
  return createTokenizer(vocab, mergesText);
}
