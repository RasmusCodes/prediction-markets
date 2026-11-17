"use strict";
// Prediction Market Desk - vanilla JS client. Polls the simulator API and draws canvas charts with hover tooltips.
const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const S = { sel: null, state: null, showTruth: true, real: null };

async function get(url) { const r = await fetch(url, { cache: "no-store" }); return r.json(); }
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}
const fmt = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
const money = (x) => (x === null || x === undefined) ? "–" : (x < 0 ? "−$" : "$") + Math.abs(x).toFixed(2);
const mmss = (t) => { t = Math.max(0, Math.floor(t)); const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(s).padStart(2, "0"); };

// ------------------------------------------------------------------------------------------- chart helper
// series: [{name, color, x:[], y:[], dash, width, band:{lo:[],hi:[]}}]; opts: {ymin,ymax,ylab,vlines:[]}
function lineChart(canvas, series, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  if (!canvas._h) { canvas._h = canvas.height; canvas.style.height = canvas._h + "px"; }
  const W = canvas.clientWidth;
  if (!W) { requestAnimationFrame(() => lineChart(canvas, series, opts)); return; }   // tab not laid out yet
  canvas.width = W * dpr; canvas.height = canvas._h * dpr; canvas._dpr = dpr;
  const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const h = canvas._h, pad = { l: 44, r: 12, t: 8, b: 24 };
  g.clearRect(0, 0, W, h);
  let xs = [], ys = [];
  series.forEach((s) => { s.x.forEach((x, i) => { if (x !== null && s.y[i] !== null && s.y[i] !== undefined) { xs.push(x); ys.push(s.y[i]); } });
    if (s.band) { s.band.lo.forEach((v) => v !== null && ys.push(v)); s.band.hi.forEach((v) => v !== null && ys.push(v)); } });
  if (!xs.length) { g.fillStyle = css("--muted"); g.font = "12px system-ui"; g.fillText("No data yet", pad.l + 8, h / 2); canvas._geom = null; return; }
  let x0 = Math.min(...xs), x1 = Math.max(...xs); if (x1 === x0) x1 = x0 + 1;
  let y0 = opts.ymin ?? Math.min(...ys), y1 = opts.ymax ?? Math.max(...ys);
  if (opts.ymin === undefined || opts.ymax === undefined) { const span = (y1 - y0) || 1; y0 -= span * 0.08; y1 += span * 0.08; }
  const X = (x) => pad.l + (x - x0) / (x1 - x0) * (W - pad.l - pad.r);
  const Y = (y) => pad.t + (1 - (y - y0) / (y1 - y0)) * (h - pad.t - pad.b);
  // grid + axes
  g.strokeStyle = css("--grid"); g.lineWidth = 1; g.fillStyle = css("--muted"); g.font = "11px system-ui";
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) { const v = y0 + (y1 - y0) * i / ticks, yy = Y(v);
    g.beginPath(); g.moveTo(pad.l, yy); g.lineTo(W - pad.r, yy); g.stroke(); g.fillText(fmt(v, Math.abs(y1 - y0) < 10 ? 1 : 0), 4, yy + 4); }
  for (let i = 0; i <= 4; i++) { const v = x0 + (x1 - x0) * i / 4; g.fillText(opts.xfmt ? opts.xfmt(v) : mmss(v), X(v) - 14, h - 6); }
  (opts.vlines || []).forEach((v) => { if (v >= x0 && v <= x1) { g.strokeStyle = css("--axis"); g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(v), pad.t); g.lineTo(X(v), h - pad.b); g.stroke(); g.setLineDash([]); } });
  // bands then lines
  series.forEach((s) => { if (!s.band) return; g.fillStyle = s.bandColor || css("--band"); g.beginPath(); let started = false;
    s.x.forEach((x, i) => { const v = s.band.hi[i]; if (v === null) return; if (!started) { g.moveTo(X(x), Y(v)); started = true; } else g.lineTo(X(x), Y(v)); });
    for (let i = s.x.length - 1; i >= 0; i--) { const v = s.band.lo[i]; if (v !== null) g.lineTo(X(s.x[i]), Y(v)); } g.closePath(); g.fill(); });
  series.forEach((s) => { if (!s.y) return; g.strokeStyle = s.color; g.lineWidth = s.width || 2; g.setLineDash(s.dash || []); g.beginPath(); let pen = false;
    s.x.forEach((x, i) => { const v = s.y[i]; if (v === null || v === undefined) { pen = false; return; } if (!pen) { g.moveTo(X(x), Y(v)); pen = true; } else g.lineTo(X(x), Y(v)); });
    g.stroke(); g.setLineDash([]); });
  canvas._geom = { X, Y, x0, x1, pad, W, h, series, xfmt: opts.xfmt, ylab: opts.ylab || "" };
}

function attachHover(canvas) {
  const tip = $("#tip");
  canvas.addEventListener("mousemove", (e) => {
    const G = canvas._geom; if (!G) return;
    const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left;
    if (mx < G.pad.l || mx > G.W - G.pad.r) { tip.style.display = "none"; return; }
    const xv = G.x0 + (mx - G.pad.l) / (G.W - G.pad.l - G.pad.r) * (G.x1 - G.x0);
    let lines = [`<b>${G.xfmt ? G.xfmt(xv) : "t " + mmss(xv)}</b>`];
    G.series.forEach((s) => { if (!s.x.length || !s.y) return; let best = 0, bd = Infinity;
      s.x.forEach((x, i) => { const d = Math.abs(x - xv); if (d < bd && s.y[i] !== null) { bd = d; best = i; } });
      if (bd < Infinity) lines.push(`<span style="color:${s.color}">■</span> ${s.name}: ${fmt(s.y[best], 2)}${G.ylab}`); });
    tip.innerHTML = lines.join("<br>"); tip.style.display = "block";
    tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 10) + "px";
  });
  canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; });
}
function legend(el, items) { el.innerHTML = items.map(([n, c, dash]) => `<span><i style="background:${c};${dash ? "opacity:.6" : ""}"></i>${n}</span>`).join(""); }

// ------------------------------------------------------------------------------------------- simulator views
function renderState(st) {
  S.state = st;
  $("#prog").style.width = (100 * st.t / st.T) + "%";
  $("#clock").textContent = `t = ${mmss(st.t)} / ${mmss(st.T)}  ·  seed ${st.seed}  ·  ${st.events.toLocaleString()} events  ·  ${st.trades.toLocaleString()} trades` + (st.settled ? "  ·  SETTLED" : "");
  $("#btnPause").textContent = st.running ? "Pause" : "Resume";
  const vs = $("#venueSel");
  if (vs.options.length === 1) [...new Set(st.markets.map((m) => m.venue))].forEach((v) => vs.add(new Option(v, v)));
  const f = $("#filter").value.toLowerCase(), vf = vs.value;
  const tb = $("#mktTable tbody");
  const rows = st.markets.filter((m) => (!vf || m.venue === vf) && (!f || (m.market + " " + m.title).toLowerCase().includes(f)));
  tb.innerHTML = rows.map((m) => {
    const key = m.venue + "|" + m.market, sel = S.sel && S.sel.key === key ? "sel" : "";
    const settled = m.outcome !== null && m.outcome !== undefined ? (m.outcome ? " ✓YES" : " ✗NO") : "";
    return `<tr class="${sel}" data-key="${key}"><td>${m.venue}</td><td><span class="mk mono">${m.market}${settled}</span><span class="ttl">${m.title}</span></td>
      <td class="r">${fmt(m.bid)}</td><td class="r">${fmt(m.ask)}</td><td class="r">${fmt(m.mid, 1)}</td>
      <td class="r truthcol">${fmt(m.truth, 1)}</td><td class="r">${fmt(m.fair, 1)}</td><td class="r">${m.volume.toLocaleString()}</td></tr>`;
  }).join("");
  if (!S.sel && st.markets.length) selectMarket(st.markets[0].venue + "|" + st.markets[0].market);
  renderStrategies(st);
  renderRisk(st.risk);
  renderArb(st);
}

function renderStrategies(st) {
  const names = { mm: "Automated market maker", arb: "LP arbitrageur", mispricing: "Mispricing (copula)", you: "You (manual)" };
  $("#strats").innerHTML = Object.entries(st.strategies).map(([k, s]) => {
    const inv = Object.entries(s.inventory).slice(0, 6).map(([m, q]) => `${m.split(":")[1]} ${q > 0 ? "+" : ""}${q}`).join(", ") || "flat";
    const extra = k === "mm" ? `<span>messages</span><span>${s.messages.toLocaleString()}</span><span>event requotes</span><span>${s.event_requotes}</span><span>news pulls</span><span>${s.pulls}</span>`
      : k === "arb" ? `<span>opportunities taken</span><span>${s.opportunities_taken}</span>` : "";
    return `<div class="strat"><div class="h"><span>${names[k] || k}</span><span class="${s.pnl_truth >= 0 ? "pos" : "neg"}">${money(s.pnl_truth)}</span></div>
      <div class="kv"><span>fills / volume</span><span>${s.fills} / ${s.volume.toLocaleString()}</span><span>fees</span><span>${money(s.fees)}</span>
      <span>open orders</span><span>${s.open_orders}</span>${extra}<span>inventory</span><span class="mono">${inv}</span></div></div>`;
  }).join("");
}

function renderRisk(r) {
  if (!r || !r.families) { $("#risk").innerHTML = ""; return; }
  const status = r.killed ? `<span class="status neg">⛔ Kill switch engaged</span> ${r.kill_reason}` : `<span class="status pos">✓ Within limits</span>`;
  $("#risk").innerHTML = `<div>${status}</div><div class="sub">Worst-case settlement loss per family (exact over joint outcome states)</div>` +
    r.families.map((f) => `<div class="fam"><span>${f.family}</span><span>${money(-f.worst_case_loss)}</span></div>`).join("") +
    `<div class="sub mt">Limits: max position ${r.limits.max_pos}, family loss $${(r.limits.max_family_loss / 100).toFixed(0)}, collar ${r.limits.collar_ticks}c, drawdown $${(r.limits.max_drawdown / 100).toFixed(0)}</div>`;
}

function renderArb(st) {
  $("#arbSub").textContent = `${st.scans} scans · ${st.arb_count} with an opportunity · LP over joint outcome states, after fees, at displayed depth`;
  $("#arbTable tbody").innerHTML = st.arb.length ? st.arb.map((o) => `<tr><td>${o.family}</td><td>${o.kind}</td><td class="r">${fmt(o.profit, 2)}</td>
    <td class="r">${fmt(o.capital, 2)}</td><td class="mono">${o.legs.map((l) => `${l.side} ${l.qty} ${l.venue}:${l.market}@${l.limit}`).join(" · ")}</td></tr>`).join("")
    : `<tr><td colspan="5" class="sub">No executable arbitrage in the latest scan.</td></tr>`;
}

function selectMarket(key) {
  const [venue, market] = key.split("|");
  S.sel = { key, venue, market };
  const m = S.state && S.state.markets.find((x) => x.venue === venue && x.market === market);
  $("#chartTitle").textContent = `${venue} · ${market}`;
  $("#chartSub").textContent = m ? m.title + " · fees " + m.fee : "";
  $("#tkMarket").textContent = `${venue} · ${market}`;
  if (m && m.mid) $("#tkPx").value = Math.round(m.mid);
  refreshMarket();
}

async function refreshMarket() {
  if (!S.sel) return;
  const { venue, market } = S.sel;
  const [bk, se] = await Promise.all([get(`/api/book?venue=${venue}&market=${encodeURIComponent(market)}`),
    get(`/api/series?venue=${venue}&market=${encodeURIComponent(market)}`)]);
  // book ladder
  const levels = [];
  const asks = bk.asks.slice(0, 8).reverse(), bids = bk.bids.slice(0, 8);
  const maxq = Math.max(1, ...bk.asks.map((x) => x[1]), ...bk.bids.map((x) => x[1]));
  const own = (k) => (bk.own[k] || []).map((o) => ` · ${o.who} ${o.qty}`).join("");
  asks.forEach(([p, q]) => levels.push(`<div class="lvl"><div class="b"></div><div class="p ${bk.own["a" + p] ? "own" : ""}">${p}</div><div class="a"><span style="width:${100 * q / maxq}%"></span><em>${q}${own("a" + p)}</em></div></div>`));
  levels.push(`<div class="lvl mid"><div></div><div class="p sub">${bk.truth !== null && S.showTruth ? "truth " + fmt(bk.truth, 1) : "·"}</div><div></div></div>`);
  bids.forEach(([p, q]) => levels.push(`<div class="lvl"><div class="b"><span style="width:${100 * q / maxq}%"></span><em>${q}${own("b" + p)}</em></div><div class="p ${bk.own["b" + p] ? "own" : ""}">${p}</div><div class="a"></div></div>`));
  $("#book").innerHTML = levels.join("");
  $("#bookSub").textContent = bk.bids.length && bk.asks.length ? `spread ${bk.asks[0][0] - bk.bids[0][0]}c` : "one-sided";
  $("#tape tbody").innerHTML = bk.trades.map((t) => `<tr><td>${mmss(t.t)}</td><td class="r">${t.price}</td><td class="r">${t.qty}</td>
    <td class="${t.side === "buy" ? "pos" : "neg"}">${t.side}</td><td class="sub">${t.taker} / ${t.maker}</td></tr>`).join("");
  // probability chart
  const mid = se.bid.map((b, i) => (b !== null && se.ask[i] !== null) ? (b + se.ask[i]) / 2 : null);
  const series = [{ name: "Mid", color: css("--s1"), x: se.t, y: mid, band: { lo: se.bid, hi: se.ask } },
    { name: "Our fair value", color: css("--s2"), x: se.fair_t, y: se.fair, width: 1.6 }];
  const leg = [["Mid (bid/ask band)", css("--s1")], ["Our fair value", css("--s2")]];
  if (S.showTruth) { series.push({ name: "True probability", color: css("--s3"), x: se.t, y: se.truth, width: 1.6, dash: [4, 3] }); leg.push(["True probability (simulation only)", css("--s3"), true]); }
  legend($("#probLegend"), leg);
  lineChart($("#probChart"), series, { ymin: 0, ymax: 100, vlines: se.news, ylab: "c" });
}

async function refreshPnl() {
  const p = await get("/api/pnl");
  const colors = { our_mm: css("--s1"), our_arb: css("--s2"), our_mispricing: css("--s3"), you: css("--s4") };
  const names = { our_mm: "Market maker", our_arb: "LP arbitrageur", our_mispricing: "Mispricing", you: "You" };
  const series = Object.entries(p.pnl).map(([k, v]) => ({ name: names[k] || k, color: colors[k] || css("--muted"), x: v.t, y: v.pnl }));
  legend($("#pnlLegend"), series.map((s) => [s.name, s.color]));
  lineChart($("#pnlChart"), series, { ylab: "$", vlines: S.state ? [S.state.close_t] : [] });
}

// ------------------------------------------------------------------------------------------- real-data + research tabs
async function loadReal() {
  if (S.real) return;
  S.real = await get("/api/real");
  const sel = $("#winSel");
  S.real.windows.forEach((w, i) => sel.add(new Option(`${w.ticker} (${w.result ? "YES" : "NO"})`, i)));
  sel.onchange = drawReal;
  if (!S.real.windows.length) {   // raw live capture not present (it is not part of the public repository)
    sel.style.display = "none";
    $("#realChart").style.display = "none";
    $("#realLegend").textContent = "No recorded windows: this chart needs data/processed_speed_live.pkl, which " +
      "research/real_speed.py builds from the raw live capture (not included in the repository; see data/DATA.md).";
  }
  drawReal();
  const tabs = S.real.tables || {};
  const titles = { calibration_summary: "Calibration by venue and horizon", flb_returns: "Return to buying YES by price bucket",
    real_arb_summary: "Executable arbitrage in the live capture", real_replay_crypto15m_summary: "Replay backtests, 15-minute crypto markets",
    speed_event_summary: "Catch-up after large spot moves", sim_e5_mm_design: "Simulation E5: market-maker design",
    sim_e4_arbitrage: "Simulation E4: arbitrage incidence" };
  $("#realTables").innerHTML = Object.entries(tabs).map(([k, t]) => `<h3>${titles[k] || k}</h3><div class="tablewrap short"><table class="grid small">
    <thead><tr>${(t.index && typeof t.index[0] === "string" ? ["<th></th>"] : []).join("")}${t.columns.map((c) => `<th>${c}</th>`).join("")}</tr></thead>
    <tbody>${t.data.map((r, i) => `<tr>${t.index && typeof t.index[0] === "string" ? `<td>${t.index[i]}</td>` : ""}${r.map((v) => `<td>${typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3)) : (v ?? "–")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`).join("");
  const caps = { fig_calibration_reliability: "Reliability diagrams, 24 h before close", fig_calibration_horizons: "Calibration slope by horizon",
    fig_flb_returns: "Favourite-longshot returns", fig_liquidity_efficiency: "Calibration by liquidity tercile",
    fig_speed_impulse: "Kalshi catch-up after spot moves", fig_real_overround: "Overround of mutually exclusive sets",
    fig_sim_e1_liquidity_speed: "Simulation: liquidity and information speed", fig_sim_e2_calibration: "Simulation: longshot-biased flow",
    fig_sim_e4_arbitrage: "Simulation: arbitrage incidence", fig_sim_e5_mm_design: "Simulation: market-maker design" };
  $("#figs").innerHTML = (S.real.figures || []).map((f) => `<figure><img src="/figures/${f}" alt="${caps[f.replace(".png", "")] || f}" loading="lazy">
    <figcaption>${caps[f.replace(".png", "")] || f}</figcaption></figure>`).join("");
}
function drawReal() {
  const w = S.real.windows[$("#winSel").value || 0]; if (!w) return;
  const series = [{ name: "Kalshi price", color: css("--s1"), x: w.t, y: w.kalshi }, { name: "Spot-implied p*", color: css("--s2"), x: w.t, y: w.pstar, width: 1.6 }];
  legend($("#realLegend"), series.map((s) => [s.name, s.color]));
  lineChart($("#realChart"), series, { ymin: 0, ymax: 100, ylab: "c", xfmt: (v) => mmss(v) + " into window" });
}

// ------------------------------------------------------------------------------------------- wiring
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab !== "sim") loadReal();
}));
$("#mktTable tbody").addEventListener("click", (e) => { const tr = e.target.closest("tr"); if (tr) selectMarket(tr.dataset.key); });
$("#filter").addEventListener("input", () => S.state && renderState(S.state));
$("#venueSel").addEventListener("change", () => S.state && renderState(S.state));
$("#showTruth").addEventListener("change", (e) => { S.showTruth = e.target.checked; document.body.classList.toggle("hide-truth", !S.showTruth); refreshMarket(); });
$("#btnPause").addEventListener("click", () => post("/api/control", { action: S.state && S.state.running ? "pause" : "resume" }));
$("#speed").addEventListener("change", (e) => post("/api/control", { action: "speed", value: e.target.value }));
$("#btnReset").addEventListener("click", async () => { await post("/api/control", { action: "reset" }); S.sel = null; $("#venueSel").length = 1; });
$("#ticket").addEventListener("submit", async (e) => {
  e.preventDefault(); if (!S.sel) return;
  const r = await post("/api/order", { venue: S.sel.venue, market: S.sel.market, side: $("#tkSide").value, price: +$("#tkPx").value, qty: +$("#tkQty").value, tif: $("#tkTif").value });
  $("#tkMsg").textContent = r.ok ? `order ${r.oid} sent (arrives after 50 ms latency)` : "rejected: check price 1-99 and quantity";
});
[$("#probChart"), $("#pnlChart"), $("#realChart")].forEach(attachHover);
window.addEventListener("resize", () => { refreshMarket(); refreshPnl(); if (S.real) drawReal(); });

async function tick() { try { renderState(await get("/api/state")); } catch (e) { /* server restarting */ } }
tick(); setInterval(tick, 1000);
setInterval(() => refreshMarket().catch(() => {}), 1500);
setInterval(() => refreshPnl().catch(() => {}), 3000);
