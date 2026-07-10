// Pixel wordmark that literally sheds pixels as the bits drop. 5x7 bitmap glyphs rendered as SVG
// rects; corrupt(level) deterministically kills/reddens a fraction of pixels (seeded by level so it
// never shimmers — same lever position, same damage).

const FONT = {
  B: ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
  I: ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"],
  T: ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
  C: [".####", "#....", "#....", "#....", "#....", "#....", ".####"],
  R: ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
  U: ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
  S: [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
  H: ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
};

export function createPixelmark(el, text = "BITCRUSH", px = 4) {
  const NS = "http://www.w3.org/2000/svg";
  const cols = text.length * 6 - 1, rows = 7;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${cols * px} ${rows * px}`);
  svg.setAttribute("shape-rendering", "crispEdges");   // pixels are the point — no antialiased mush
  svg.classList.add("pixelmark-svg");
  const pixels = [];
  let cx = 0;
  for (const ch of text) {
    const glyph = FONT[ch];
    if (!glyph) { cx += 6; continue; }
    for (let r = 0; r < 7; r++) for (let c = 0; c < 5; c++) {
      if (glyph[r][c] !== "#") continue;
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", (cx + c) * px); rect.setAttribute("y", r * px);
      rect.setAttribute("width", px - 0.6); rect.setAttribute("height", px - 0.6);
      rect.classList.add("pxl");
      svg.appendChild(rect); pixels.push(rect);
    }
    cx += 6;
  }
  el.appendChild(svg);

  // xorshift PRNG so damage is deterministic per level
  function corrupt(level) {                        // 0 = pristine … 1 = wrecked
    let seed = 0x9e37 ^ Math.round(level * 97) * 2654435761;
    const rnd = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return ((seed >>> 0) / 4294967296); };
    const kill = Math.min(level * 0.55, 0.5), red = Math.min(level * 0.35, 0.3);
    for (const p of pixels) {
      const r = rnd();
      p.classList.toggle("dead", r < kill);
      p.classList.toggle("hot", r >= kill && r < kill + red);
    }
  }
  return { corrupt };
}
