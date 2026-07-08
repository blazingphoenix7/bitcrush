// 7-segment LED display (SVG). Ghost segments stay faintly visible when off — the hardware tell.
// createSeg7(el, {cells: 3, dpAfter: 2}) -> { set("16.0") }   (digits right-aligned, one decimal)

const SEGS = {
  //        A  B  C  D  E  F  G
  "0": ["A","B","C","D","E","F"],
  "1": ["B","C"],
  "2": ["A","B","G","E","D"],
  "3": ["A","B","G","C","D"],
  "4": ["F","G","B","C"],
  "5": ["A","F","G","C","D"],
  "6": ["A","F","G","E","D","C"],
  "7": ["A","B","C"],
  "8": ["A","B","C","D","E","F","G"],
  "9": ["A","B","C","D","F","G"],
  "-": ["G"],
  " ": [],
};
// segment rects within a 27x45 digit cell
const GEO = {
  A: [4.5, 0, 18, 5], G: [4.5, 20, 18, 5], D: [4.5, 40, 18, 5],
  F: [0, 4.5, 5, 16], B: [22, 4.5, 5, 16],
  E: [0, 24.5, 5, 16], C: [22, 24.5, 5, 16],
};
const CELL_W = 27, CELL_H = 45, GAP = 9, DP_W = 9;

export function createSeg7(el, opts = {}) {
  const cells = opts.cells ?? 3;
  const dpAfter = opts.dpAfter ?? 2;              // decimal point drawn after this cell (1-based)
  const W = cells * CELL_W + (cells - 1) * GAP + DP_W;
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${CELL_H}`);
  svg.classList.add("seg7");
  const digitSegs = [];                            // digitSegs[i][segName] = rect
  let x = 0;
  for (let i = 0; i < cells; i++) {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("transform", `translate(${x},0) skewX(-4)`);
    const map = {};
    for (const [name, [rx, ry, rw, rh]] of Object.entries(GEO)) {
      const r = document.createElementNS(NS, "rect");
      r.setAttribute("x", rx); r.setAttribute("y", ry);
      r.setAttribute("width", rw); r.setAttribute("height", rh);
      r.setAttribute("rx", 2.2);
      r.classList.add("seg");
      g.appendChild(r); map[name] = r;
    }
    svg.appendChild(g); digitSegs.push(map);
    x += CELL_W;
    if (i + 1 === dpAfter) {                       // decimal point
      const dp = document.createElementNS(NS, "circle");
      dp.setAttribute("cx", x + 4); dp.setAttribute("cy", CELL_H - 3.5);
      dp.setAttribute("r", 3); dp.classList.add("seg", "dp", "on");
      svg.appendChild(dp);
      x += DP_W;
    } else if (i < cells - 1) x += GAP - 4;
  }
  el.appendChild(svg);

  function set(str) {                              // e.g. "16.0" / " 8.4" / " 2.0"
    const chars = str.replace(".", "").padStart(cells, " ").slice(-cells);
    for (let i = 0; i < cells; i++) {
      const on = new Set(SEGS[chars[i]] ?? []);
      for (const [name, rect] of Object.entries(digitSegs[i]))
        rect.classList.toggle("on", on.has(name));
    }
  }
  return { set };
}
