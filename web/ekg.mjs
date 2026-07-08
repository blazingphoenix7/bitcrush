// EKG strip — the subject's "brain activity". y = per-token entropy (real, from the live logits).
// Calm confident generation = low gentle trace; a dying mind = jagged spikes. Event-driven redraws
// (per token), so it works even when rAF is throttled.

const PHOS = "#45ff78";

export function createEkg(canvas) {
  const ctx = canvas.getContext("2d");
  const MAXP = 72;
  let pts = [];                                    // entropy values, newest last
  let W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    W = canvas.clientWidth || 300; H = canvas.clientHeight || 40;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  const yOf = (ent) => H - 5 - Math.min(ent / 9, 1) * (H - 12);

  function draw(beatX = -1) {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(69,255,120,.14)";
    ctx.beginPath(); ctx.moveTo(0, yOf(0.4)); ctx.lineTo(W, yOf(0.4)); ctx.stroke();
    if (!pts.length) {                             // idle flatline + roaming blip
      const y = yOf(0.4);
      ctx.strokeStyle = PHOS; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(0, y);
      if (beatX >= 0) {
        ctx.lineTo(beatX - 8, y); ctx.lineTo(beatX - 4, y - 7); ctx.lineTo(beatX, y + 5); ctx.lineTo(beatX + 4, y);
      }
      ctx.lineTo(W, y); ctx.stroke();
      return;
    }
    const step = W / (MAXP - 1), off = Math.max(0, pts.length - MAXP);
    const draw1 = (width, alpha) => {
      ctx.strokeStyle = PHOS; ctx.globalAlpha = alpha; ctx.lineWidth = width;
      ctx.lineJoin = "round"; ctx.beginPath();
      for (let i = off; i < pts.length; i++) {
        const x = (i - off) * step, y = yOf(pts[i]);
        i === off ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke(); ctx.globalAlpha = 1;
    };
    draw1(3.2, 0.22);                              // glow pass
    draw1(1.4, 1);
    const lx = ((pts.length - 1 - off) * step), ly = yOf(pts[pts.length - 1]);
    ctx.fillStyle = PHOS; ctx.beginPath(); ctx.arc(lx, ly, 2.4, 0, 7); ctx.fill();
  }

  new ResizeObserver(resize).observe(canvas);
  resize();
  return {
    push(ent) { pts.push(ent); draw(); },
    reset() { pts = []; draw(); },
    idleBeat(x) { if (!pts.length) draw(x * W); },  // x: 0..1 roaming heartbeat position
    hasData() { return pts.length > 0; },
  };
}
