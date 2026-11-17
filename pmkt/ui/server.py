"""Dashboard server: runs a live simulation in a background thread and serves a single-page UI.

    python -m pmkt.ui.server [--port 8797] [--seed 7] [--speed 30]

Standard library only (http.server). Endpoints (JSON):
  GET  /api/state                     clock, markets table (bid/ask/mid/truth/fair), strategies, risk, arbitrage
  GET  /api/book?venue=&market=       depth ladder, our resting orders, recent trades
  GET  /api/series?venue=&market=     probability history: bid, ask, mid, truth, our fair value
  GET  /api/pnl                       equity and inventory paths of our strategies and the manual account
  GET  /api/real                      recorded real-market examples (Kalshi 15-min crypto vs spot) + research tables
  POST /api/control {action: pause|resume|speed|reset, value}
  POST /api/order   {venue, market, side: buy|sell, price, qty, tif}   manual order from the "you" account
  POST /api/cancel  {venue, oid}
Static files: pmkt/ui/static/, research figures under /figures/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..core.orderbook import BUY, MAX_TICK, SELL
from ..sim.agents import Agent
from ..sim.scenario import build

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGS = os.path.join(ROOT, "paper", "figures")


class Human(Agent):
    """Manual trading account driven from the dashboard."""
    kind = "human"

    def __init__(self, capital):
        super().__init__("you", capital, [], latency=0.05, jitter=0.0, rate=0.0)
        self.events = []

    def on_order_result(self, venue, market, res):
        super().on_order_result(venue, market, res)
        self.events.append({"t": self.sim.now, "oid": res.oid, "venue": venue, "market": market, "status": res.status,
                            "reason": res.reason, "filled": sum(f.qty for f in res.fills)})
        self.events = self.events[-50:]


def _f(x, nd=4):
    if x is None:
        return None
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


class SimHost:
    def __init__(self, seed: int = 7, speed: float = 30.0, T: float = 7200.0):
        self.lock = threading.RLock()
        self.speed = speed
        self.T = T
        self.running = True
        self.reset(seed)
        threading.Thread(target=self._loop, daemon=True).start()

    def reset(self, seed: int | None = None):
        with self.lock:
            self.seed = seed if seed is not None else random.randint(1, 10_000)
            over = {"seed": self.seed, "T": self.T, "record_dt": 2.0, "monitor": {"enabled": True, "period": 2.0}}
            self.sim, self.ctx = build(overrides=over)
            self.human = Human({v: 1_000_000 for v in self.sim.venues})   # $10k per venue
            self.sim.add_agent(self.human)
            self.fair_hist: dict[str, list] = {}
            self.last_fair_t = -1e9
            self.started = time.time()

    def _loop(self):
        last = time.time()
        while True:
            time.sleep(0.1)
            now = time.time()
            dt = now - last
            last = now
            with self.lock:
                if not self.running or self.sim.settled:
                    continue
                target = min(self.sim.now + dt * self.speed, self.sim.T)
                t_budget = time.time() + 0.25
                while self.sim.now < target and time.time() < t_budget:
                    self.sim.run(until=min(self.sim.now + 2.0, target))
                    self._sample_fair()

    def _sample_fair(self):
        mm = self.ctx["ours"].get("mm")
        if mm is None or self.sim.now - self.last_fair_t < 2.0:
            return
        self.last_fair_t = self.sim.now
        for m in mm.markets:
            h = self.fair_hist.setdefault(m, [])
            h.append((self.sim.now, mm.fair(m)))
            if len(h) > 6000:
                del h[: len(h) - 6000]

    # ------------------------------------------------------------------ views
    def state(self):
        with self.lock:
            sim, ctx = self.sim, self.ctx
            w = sim.world
            mm = ctx["ours"].get("mm")
            mon = ctx["ours"].get("monitor")
            rows = []
            for v, ex in sim.venues.items():
                for m, mk in ex.markets.items():
                    b = mk.book
                    bb, bq, ba, aq = b.bbo()
                    d = w.defs[m]
                    rows.append({"venue": v, "market": m, "title": d.title, "family": d.family, "kind": d.kind,
                                 "bid": bb or None, "ask": ba if ba <= MAX_TICK else None, "bid_sz": bq, "ask_sz": aq,
                                 "mid": _f(b.mid(), 2), "last": b.last_price, "volume": b.volume,
                                 "truth": _f(100 * w.prob(m, sim.now), 2),
                                 "fair": _f(100 * mm.fair(m), 2) if mm and m in mm.fv else None,
                                 "status": mk.status, "outcome": mk.outcome,
                                 "fee": ex.fee.describe()})
            strategies = {}
            for key, ag in ctx["ours"].items():
                if key == "monitor":
                    continue
                strategies[key] = self._agent_summary(ag)
            strategies["you"] = self._agent_summary(self.human)
            risk = {}
            if mm is not None:
                pos = mm.net_positions()
                marks = {m: 100 * mm.fair(m) for m in mm.markets}
                risk = {"killed": mm.risk.killed, "kill_reason": mm.risk.kill_reason,
                        "peak": _f(mm.risk.peak / 100 if mm.risk.peak else None, 2),
                        "rejects": dict(mm.risk.n_rejects), "limits": vars(mm.risk.lim),
                        "families": [{"family": f, "worst_case_loss": _f(mm.risk.worst_case_loss(f, pos, marks) / 100, 2),
                                      "positions": {m: int(pos.get(m, 0)) for m in s.markets if pos.get(m, 0)}}
                                     for f, s in mm.risk.struct.items()]}
            arb = []
            if mon is not None:
                for o in mon.latest[:12]:
                    arb.append({"family": o.family, "kind": o.kind, "profit": _f(o.profit / 100, 2),
                                "capital": _f(o.capital / 100, 2),
                                "legs": [{"venue": v, "market": m, "side": "buy" if s == BUY else "sell", "limit": p,
                                          "qty": q} for v, m, s, p, q in o.orders]})
            news = [{"t": n.t, "family": n.family} for n in w.scheduled_news() if n.t < w.T - 1]
            arb_hist = mon.log[-400:] if mon else []
            return {"t": _f(sim.now, 1), "T": sim.T, "close_t": sim.close_t, "running": self.running,
                    "settled": sim.settled, "speed": self.speed, "seed": self.seed, "markets": rows,
                    "strategies": strategies, "risk": risk, "arb": arb,
                    "arb_count": len(set(x[0] for x in arb_hist)), "scans": mon.n_scans if mon else 0,
                    "news": news, "events": sim.n_events, "trades": len(sim.recorder.trades),
                    "human_events": self.human.events[-8:]}

    def _agent_summary(self, ag):
        sim = self.sim
        eq = 0.0
        cash0 = 0
        fees = 0
        vol = 0
        inv = {}
        mm = self.ctx["ours"].get("mm")
        for v, ex in sim.venues.items():
            a = ex.accounts.get(ag.id)
            if a is None:
                continue
            marks = {m: (100 * sim.world.prob(m, sim.now)) for m in ex.markets}
            eq += a.equity(marks)
            cash0 += a.cash0
            fees += a.fees_paid
            vol += a.volume
            for m, q in a.pos.items():
                if q:
                    inv[f"{v}:{m}"] = int(q)
        out = {"kind": ag.kind, "pnl_truth": _f((eq - cash0) / 100, 2), "fees": _f(fees / 100, 2), "volume": vol,
               "fills": ag.n_fills, "inventory": inv,
               "open_orders": sum(1 for st in ag.orders.values() if st.status != "pending")}
        if ag.kind == "mm":
            out.update({"messages": ag.qm.n_msgs, "event_requotes": ag.n_event_requotes, "pulls": ag.n_pulls})
        if ag.kind == "arb":
            out.update({"opportunities_taken": len(ag.found), "orders": ag.n_sent})
        return out

    def book(self, venue, market):
        with self.lock:
            ex = self.sim.venues[venue]
            b = ex.markets[market].book
            bids, asks = b.depth(12)
            own = {}
            for aid, ag in (("mm", self.ctx["ours"].get("mm")), ("you", self.human)):
                if ag is None:
                    continue
                for oid, st in ag.orders.items():
                    if st.venue == venue and st.market == market and st.status == "live":
                        own.setdefault(f"{'b' if st.side == BUY else 'a'}{st.price}", []).append(
                            {"who": aid, "qty": st.rem, "oid": oid})
            trades = [{"t": _f(t[0], 1), "price": t[3], "qty": t[4], "side": "buy" if t[5] == BUY else "sell",
                       "taker": t[8], "maker": t[9]}
                      for t in reversed(self.sim.recorder.trades[-400:]) if t[1] == venue and t[2] == market][:30]
            return {"venue": venue, "market": market, "bids": bids, "asks": asks, "own": own, "trades": trades,
                    "truth": _f(100 * self.sim.world.prob(market, self.sim.now), 2)}

    def series(self, venue, market):
        with self.lock:
            rows = [r for r in self.sim.recorder.book_rows if r[1] == venue and r[2] == market]
            fh = list(self.fair_hist.get(market, []))
            step = max(1, len(rows) // 1500)
            rows = rows[::step]
            return {"t": [_f(r[0], 1) for r in rows], "bid": [r[3] for r in rows], "ask": [r[4] for r in rows],
                    "truth": [_f(100 * r[9], 2) if r[9] is not None else None for r in rows],
                    "fair_t": [_f(x[0], 1) for x in fh[::max(1, len(fh) // 1500)]],
                    "fair": [_f(100 * x[1], 2) for x in fh[::max(1, len(fh) // 1500)]],
                    "news": [n.t for n in self.sim.world.news if self.sim.world.fam_of.get(market) == n.family]}

    def pnl(self):
        with self.lock:
            out = {}
            ids = [a.id for a in self.ctx["ours"].values() if a.kind != "monitor"] + ["you"]
            by = {}
            for r in self.sim.recorder.agent_rows:
                if r[1] in ids:
                    d = by.setdefault(r[1], {})
                    d[r[0]] = d.get(r[0], 0.0) + r[4]
            for aid, d in by.items():
                ts = sorted(d)
                cash0 = sum(ex.accounts[aid].cash0 for ex in self.sim.venues.values() if aid in ex.accounts)
                step = max(1, len(ts) // 1200)
                out[aid] = {"t": [_f(t, 1) for t in ts[::step]], "pnl": [_f((d[t] - cash0) / 100, 2) for t in ts[::step]]}
            inv = {}
            mm = self.ctx["ours"].get("mm")
            if mm is not None:
                for r in self.sim.recorder.pos_rows[-20000:]:
                    if r[1] == mm.id:
                        inv.setdefault(f"{r[2]}:{r[3]}", []).append((_f(r[0], 1), r[4]))
            return {"pnl": out, "inventory": {k: {"t": [x[0] for x in v][-600:], "q": [x[1] for x in v][-600:]}
                                              for k, v in inv.items()}}

    def control(self, action, value=None):
        with self.lock:
            if action == "pause":
                self.running = False
            elif action == "resume":
                self.running = True
            elif action == "speed":
                self.speed = max(0.5, min(float(value), 600.0))
            elif action == "reset":
                self.reset(int(value) if value else None)
                self.running = True
        return {"ok": True}

    def order(self, d):
        with self.lock:
            side = BUY if d.get("side") == "buy" else SELL
            oid = self.human.submit(d["venue"], d["market"], side, int(d["price"]), int(d["qty"]), d.get("tif", "GTC"))
            return {"ok": oid is not None, "oid": oid}

    def cancel(self, d):
        with self.lock:
            self.human.cancel(int(d["oid"]))
            return {"ok": True}


REAL_CACHE = {}


def real_payload():
    """Small recorded examples for the Real data tab (built once from research outputs, if present)."""
    if "v" in REAL_CACHE:
        return REAL_CACHE["v"]
    out = {"windows": [], "tables": {}}
    try:
        import pandas as pd
        p = os.path.join(ROOT, "data", "processed_speed_live.pkl")
        if os.path.exists(p):
            d = pd.read_pickle(p)
            for tk, g in list(d.groupby("ticker"))[:40]:
                g = g.iloc[::5]
                out["windows"].append({"ticker": tk, "series": g.series.iloc[0], "result": int(g.result.iloc[0]),
                                       "t": [float(x - g.t.iloc[0]) for x in g.t], "kalshi": [_f(100 * x, 2) for x in g.k],
                                       "pstar": [_f(100 * x, 2) for x in g.pstar]})
        tabs = os.path.join(ROOT, "paper", "tables")
        for name in ("calibration_summary", "flb_returns", "real_arb_summary", "real_replay_crypto15m_summary",
                     "speed_event_summary", "sim_e5_mm_design", "sim_e4_arbitrage"):
            f = os.path.join(tabs, name + ".csv")
            if os.path.exists(f):
                t = pd.read_csv(f)
                out["tables"][name] = json.loads(t.round(4).to_json(orient="split"))
        out["figures"] = sorted(x for x in os.listdir(FIGS) if x.endswith(".png")) if os.path.isdir(FIGS) else []
    except Exception as e:  # the tab is optional; never break the server
        out["error"] = str(e)
    REAL_CACHE["v"] = out
    return out


def make_handler(host: SimHost):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, default=lambda o: None).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path, ctype):
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self._file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")
                if u.path.startswith("/static/"):
                    name = os.path.basename(u.path)
                    ctype = {"js": "application/javascript", "css": "text/css"}.get(name.rsplit(".", 1)[-1], "text/plain")
                    return self._file(os.path.join(STATIC, name), ctype + "; charset=utf-8")
                if u.path.startswith("/figures/"):
                    return self._file(os.path.join(FIGS, os.path.basename(u.path)), "image/png")
                if u.path == "/api/state":
                    return self._json(host.state())
                if u.path == "/api/book":
                    return self._json(host.book(q["venue"], q["market"]))
                if u.path == "/api/series":
                    return self._json(host.series(q["venue"], q["market"]))
                if u.path == "/api/pnl":
                    return self._json(host.pnl())
                if u.path == "/api/real":
                    return self._json(real_payload())
                self.send_error(404)
            except KeyError as e:
                self._json({"error": f"missing {e}"}, 400)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            u = urlparse(self.path).path
            try:
                if u == "/api/control":
                    return self._json(host.control(d.get("action"), d.get("value")))
                if u == "/api/order":
                    return self._json(host.order(d))
                if u == "/api/cancel":
                    return self._json(host.cancel(d))
            except (KeyError, ValueError) as e:
                return self._json({"error": str(e)}, 400)
            self.send_error(404)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8797)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--speed", type=float, default=30.0)
    ap.add_argument("--T", type=float, default=7200.0)
    a = ap.parse_args()
    host = SimHost(a.seed, a.speed, a.T)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(host))
    print(f"dashboard on http://127.0.0.1:{a.port}  (seed {a.seed}, speed {a.speed}x)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
