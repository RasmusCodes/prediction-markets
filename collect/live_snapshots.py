"""Read-only live market-data collector for Kalshi and Polymarket (public, unauthenticated endpoints).

What it records (SQLite, WAL mode, single writer):
  quotes   top-of-book per market, written only when it changes (Kalshi batch GET /markets?tickers=...,
           Polymarket batch POST /books)
  depth    top-10 levels per side, written only when changed (Kalshi GET /markets/{t}/orderbook for the
           most active tickers; Polymarket from the same /books payload)
  trades   the complete Kalshi trade tape (GET /markets/trades, cursor paging back to the last seen trade)
  cycles   every snapshot time, so unchanged quotes can be forward-filled exactly
  universe which markets were being tracked, re-selected every --reselect-min minutes

Etiquette: a token bucket per host (defaults 4 req/s Kalshi, 3 req/s Polymarket, far below published
limits), exponential backoff on 429/5xx honouring Retry-After, generic User-Agent, no auth, no orders.
Resumable: re-running appends to the same day file; INSERT OR IGNORE on trade ids.

Usage:  python collect/live_snapshots.py --hours 10 [--db data/live/live.sqlite]
        python collect/live_snapshots.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone

import requests

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
UA = "pmkt-research/0.1 (read-only market data)"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}", flush=True)


class Bucket:
    """Token bucket: at most `rate` requests per second on average, burst `burst`."""

    def __init__(self, rate: float, burst: float = 2.0):
        self.rate, self.burst = rate, burst
        self.tokens, self.t = burst, time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(wait)


class Client:
    def __init__(self, k_rate: float, p_rate: float):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/json"})
        self.buckets = {"kalshi": Bucket(k_rate), "poly": Bucket(p_rate), "coinbase": Bucket(3.0)}
        self.n = Counter()

    def call(self, host: str, method: str, url: str, **kw):
        backoff = 1.0
        for attempt in range(8):
            self.buckets[host].take()
            try:
                r = self.s.request(method, url, timeout=20, **kw)
            except requests.RequestException as e:  # network blip / DNS outage: back off and retry
                log(f"net error {host}: {type(e).__name__}; sleeping {backoff:.0f}s")
                time.sleep(backoff); backoff = min(backoff * 2, 120); continue
            self.n[host] += 1
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else backoff
                log(f"{host} HTTP {r.status_code}; backing off {wait:.0f}s")
                time.sleep(wait + random.random()); backoff = min(backoff * 2, 120); continue
            log(f"{host} HTTP {r.status_code} on {url[:120]}: {r.text[:120]!r}")
            return None
        return None


def f(x):
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


def iso_to_ts(s: str | None):
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    if "." in s:  # trim sub-microsecond digits that fromisoformat rejects on 3.10
        head, tail = s.split(".", 1)
        frac, _, tz = tail.partition("+")
        s = f"{head}.{frac[:6]}+{tz}" if tz else f"{head}.{frac[:6]}"
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS cycles(ts REAL, venue TEXT, n INTEGER);
CREATE TABLE IF NOT EXISTS quotes(ts REAL, venue TEXT, market TEXT, bid REAL, ask REAL, bid_sz REAL, ask_sz REAL,
                                  last REAL, volume REAL, oi REAL);
CREATE INDEX IF NOT EXISTS ix_quotes ON quotes(venue, market, ts);
CREATE TABLE IF NOT EXISTS depth(ts REAL, venue TEXT, market TEXT, bids TEXT, asks TEXT);
CREATE INDEX IF NOT EXISTS ix_depth ON depth(venue, market, ts);
CREATE TABLE IF NOT EXISTS trades(venue TEXT, trade_id TEXT, market TEXT, ts REAL, price REAL, size REAL,
                                  taker_side TEXT, PRIMARY KEY(venue, trade_id));
CREATE INDEX IF NOT EXISTS ix_trades ON trades(venue, market, ts);
CREATE INDEX IF NOT EXISTS ix_trades_ts ON trades(ts);
CREATE TABLE IF NOT EXISTS spot(ts REAL, symbol TEXT, price REAL, bid REAL, ask REAL, server_ts REAL);
CREATE INDEX IF NOT EXISTS ix_spot ON spot(symbol, ts);
CREATE TABLE IF NOT EXISTS universe(ts REAL, venue TEXT, market TEXT, event TEXT, title TEXT, group_key TEXT,
                                    mutually_exclusive INTEGER, close_time TEXT, extra TEXT);
CREATE TABLE IF NOT EXISTS meta(venue TEXT, market TEXT, event TEXT, title TEXT, close_time TEXT, extra TEXT,
                                PRIMARY KEY(venue, market));
"""


class Store:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.last_q: dict = {}
        self.last_d: dict = {}

    def quotes(self, ts, venue, rows):
        """rows: list of (market, bid, ask, bid_sz, ask_sz, last, volume, oi); store only changes."""
        out = []
        for r in rows:
            key = (venue, r[0])
            if self.last_q.get(key) != r[1:]:
                self.last_q[key] = r[1:]
                out.append((ts, venue) + tuple(r))
        with self.lock:
            self.db.executemany("INSERT INTO quotes VALUES(?,?,?,?,?,?,?,?,?,?)", out)
            self.db.execute("INSERT INTO cycles VALUES(?,?,?)", (ts, venue, len(rows)))
            self.db.commit()
        return len(out)

    def depth(self, ts, venue, market, bids, asks):
        key = (venue, market)
        val = (json.dumps(bids), json.dumps(asks))
        if self.last_d.get(key) == val:
            return 0
        self.last_d[key] = val
        with self.lock:
            self.db.execute("INSERT INTO depth VALUES(?,?,?,?,?)", (ts, venue, market) + val)
            self.db.commit()
        return 1

    def trades(self, rows):
        with self.lock:
            cur = self.db.executemany("INSERT OR IGNORE INTO trades VALUES(?,?,?,?,?,?,?)", rows)
            self.db.commit()
            return cur.rowcount

    def spot(self, rows):
        with self.lock:
            self.db.executemany("INSERT INTO spot VALUES(?,?,?,?,?,?)", rows)
            self.db.commit()

    def hot_tickers(self, venue: str, since: float, limit: int):
        with self.lock:
            cur = self.db.execute(
                "SELECT market, COUNT(*) c FROM trades WHERE ts>=? AND venue=? GROUP BY market ORDER BY c DESC LIMIT ?",
                (since, venue, limit))
            return [r[0] for r in cur.fetchall()]

    def universe(self, ts, venue, rows):
        with self.lock:
            self.db.executemany("INSERT INTO universe VALUES(?,?,?,?,?,?,?,?,?)", [(ts, venue) + r for r in rows])
            self.db.executemany("INSERT OR REPLACE INTO meta VALUES(?,?,?,?,?,?)",
                                [(venue, r[0], r[1], r[2], r[5], r[6]) for r in rows])
            self.db.commit()

    def active_tickers(self, venue: str, since: float, limit: int):
        with self.lock:
            cur = self.db.execute(
                "SELECT market, COUNT(*) c FROM trades WHERE venue=? AND ts>=? GROUP BY market ORDER BY c DESC LIMIT ?",
                (venue, since, limit))
            return [r[0] for r in cur.fetchall()]


# ----------------------------------------------------------------------------------------------- Kalshi
class Kalshi:
    def __init__(self, cli: Client, store: Store, args):
        self.cli, self.store, self.args = cli, store, args
        self.tickers: list[str] = []
        self.depth_tickers: list[str] = []
        self.hot: list[str] = []          # most-traded tickers of the last few minutes (15-min markets roll)
        self.last_trade_ts = time.time() - args.bootstrap_min * 60
        self.seen = deque(maxlen=400_000)
        self.seen_set: set = set()

    def poll_trades(self, max_pages: int = 10) -> int:
        params = {"limit": 1000, "min_ts": int(self.last_trade_ts) - 2}
        rows, newest = [], self.last_trade_ts
        for _ in range(max_pages):
            js = self.cli.call("kalshi", "GET", f"{KALSHI}/markets/trades", params=params)
            if not js:
                break
            page = js.get("trades", [])
            for t in page:
                tid = t.get("trade_id")
                if tid in self.seen_set:
                    continue
                ts = iso_to_ts(t.get("created_time"))
                if ts is None:
                    continue
                if len(self.seen) == self.seen.maxlen:
                    self.seen_set.discard(self.seen[0])
                self.seen.append(tid); self.seen_set.add(tid)
                newest = max(newest, ts)
                if (t.get("ticker") or "").startswith("KXMVE"):  # multivariate combos: not studied, skip storage
                    continue
                rows.append(("kalshi", tid, t.get("ticker"), ts, f(t.get("yes_price_dollars")),
                             f(t.get("count_fp") or t.get("count")), t.get("taker_side")))
            cur = js.get("cursor")
            if not cur or len(page) < 1000:
                break
            params["cursor"] = cur
        self.last_trade_ts = newest
        return self.store.trades(rows) if rows else 0

    def reselect(self) -> None:
        a = self.args
        hot = [t for t in self.store.active_tickers("kalshi", time.time() - 3600, 400) if not t.startswith("KXMVE")]
        if not hot:
            log("kalshi: no recent trades yet; universe unchanged"); return
        # map hot tickers to events (batched), rank events by trade activity
        ev_of, rank = {}, Counter()
        for i in range(0, min(len(hot), 300), 100):
            js = self.cli.call("kalshi", "GET", f"{KALSHI}/markets", params={"tickers": ",".join(hot[i:i + 100])})
            for m in (js or {}).get("markets", []):
                ev_of[m["ticker"]] = m.get("event_ticker")
        for pos, t in enumerate(hot):
            if t in ev_of:
                rank[ev_of[t]] += 1.0 / (1 + pos / 50)
        rows, chosen = [], []
        for ev, _ in rank.most_common(a.k_events):
            js = self.cli.call("kalshi", "GET", f"{KALSHI}/events/{ev}", params={"with_nested_markets": "true"})
            if not js:
                continue
            e = js.get("event", js)
            mkts = [m for m in (e.get("markets") or js.get("markets") or []) if m.get("status") in ("active", "open")]
            if len(mkts) > a.k_max_per_event:  # very wide events (e.g. 80-way): keep the most active outcomes
                mkts.sort(key=lambda m: f(m.get("volume_24h_fp")) or 0, reverse=True)
                mkts = mkts[: a.k_max_per_event]
            for m in mkts:
                if m["ticker"] in chosen:
                    continue
                chosen.append(m["ticker"])
                rows.append((m["ticker"], ev, (m.get("title") or "")[:200], e.get("series_ticker"),
                             int(bool(e.get("mutually_exclusive"))), m.get("close_time"),
                             json.dumps({"yes_sub_title": m.get("yes_sub_title"), "strike_type": m.get("strike_type"),
                                         "floor_strike": m.get("floor_strike"), "cap_strike": m.get("cap_strike"),
                                         "category": e.get("category")})))
        for t in hot[: a.k_top_markets]:
            if t not in chosen and len(chosen) < a.k_cap:
                chosen.append(t)
                rows.append((t, ev_of.get(t), "", None, 0, None, "{}"))
        self.tickers = chosen[: a.k_cap]
        self.depth_tickers = [t for t in hot if t in set(self.tickers)][: a.k_depth]
        self.store.universe(time.time(), "kalshi", rows[: a.k_cap])
        log(f"kalshi universe: {len(self.tickers)} markets in {len(rank)} active events; depth on {len(self.depth_tickers)}")

    def refresh_hot(self) -> None:
        self.hot = [t for t in self.store.hot_tickers("kalshi", time.time() - 180, 60) if not t.startswith("KXMVE")][:40]

    def poll_fast_depth(self) -> int:
        """Top few hot tickers (e.g. the live 15-minute BTC market) get depth every few seconds."""
        n = 0
        for t in self.hot[: self.args.k_fast_depth]:
            n += self._depth_one(t)
        return n

    def poll_quotes(self) -> int:
        rows = []
        tickers = list(dict.fromkeys(self.hot + self.tickers))
        for i in range(0, len(tickers), 100):
            js = self.cli.call("kalshi", "GET", f"{KALSHI}/markets", params={"tickers": ",".join(tickers[i:i + 100])})
            for m in (js or {}).get("markets", []):
                rows.append((m["ticker"], f(m.get("yes_bid_dollars")), f(m.get("yes_ask_dollars")),
                             f(m.get("yes_bid_size_fp")), f(m.get("yes_ask_size_fp")), f(m.get("last_price_dollars")),
                             f(m.get("volume_fp")), f(m.get("open_interest_fp"))))
        return self.store.quotes(time.time(), "kalshi", rows) if rows else 0

    def _depth_one(self, t: str) -> int:
        js = self.cli.call("kalshi", "GET", f"{KALSHI}/markets/{t}/orderbook", params={"depth": 10})
        if not js:
            return 0
        ob = js.get("orderbook_fp") or js.get("orderbook") or {}
        yes = [[f(p), f(s)] for p, s in (ob.get("yes_dollars") or [])]
        no = [[f(p), f(s)] for p, s in (ob.get("no_dollars") or [])]
        # Kalshi books are bids on both sides: YES bids, and NO bids == YES asks at 1 - p.
        bids = sorted(yes, key=lambda x: -x[0])[:10]
        asks = sorted([[round(1 - p, 4), s] for p, s in no], key=lambda x: x[0])[:10]
        return self.store.depth(time.time(), "kalshi", t, bids, asks)

    def poll_depth(self) -> int:
        fast = set(self.hot[: self.args.k_fast_depth])
        return sum(self._depth_one(t) for t in self.depth_tickers if t not in fast)


# ----------------------------------------------------------------------------------------------- spot (Coinbase)
class Spot:
    """BTC/ETH spot from Coinbase Exchange's public ticker: the information driving Kalshi's 15-minute crypto markets."""

    def __init__(self, cli: Client, store: Store, symbols):
        self.cli, self.store, self.symbols = cli, store, symbols

    def poll(self) -> None:
        rows = []
        for s in self.symbols:
            js = self.cli.call("coinbase", "GET", f"https://api.exchange.coinbase.com/products/{s}/ticker")
            if js:
                rows.append((time.time(), s, f(js.get("price")), f(js.get("bid")), f(js.get("ask")),
                             iso_to_ts(js.get("time"))))
        if rows:
            self.store.spot(rows)


# ----------------------------------------------------------------------------------------------- Polymarket
class Poly:
    def __init__(self, cli: Client, store: Store, args):
        self.cli, self.store, self.args = cli, store, args
        self.tokens: list[str] = []

    def reselect(self) -> None:
        a = self.args
        rows, toks = [], []
        for off in (0, 50):
            js = self.cli.call("poly", "GET", f"{GAMMA}/events", params={
                "limit": 50, "offset": off, "active": "true", "closed": "false",
                "order": "volume24hr", "ascending": "false"})
            for e in js or []:
                if len(rows) >= a.p_cap:
                    break
                mk = [m for m in e.get("markets", []) if m.get("active") and not m.get("closed") and m.get("clobTokenIds")]
                mk.sort(key=lambda m: f(m.get("volume24hr")) or 0, reverse=True)
                for m in mk[: a.p_max_per_event]:
                    try:
                        tok = json.loads(m["clobTokenIds"])[0]
                        outcomes = json.loads(m.get("outcomes") or "[]")
                    except (ValueError, TypeError, IndexError):
                        continue
                    if tok in toks:
                        continue
                    toks.append(tok)
                    rows.append((tok, e.get("slug"), (m.get("question") or "")[:200], e.get("id"),
                                 int(bool(e.get("negRisk"))), m.get("endDate"),
                                 json.dumps({"condition_id": m.get("conditionId"), "slug": m.get("slug"),
                                             "outcome0": outcomes[0] if outcomes else None,
                                             "group_item": m.get("groupItemTitle"), "neg_risk": m.get("negRisk"),
                                             "tick": m.get("orderPriceMinTickSize"), "fees": m.get("feesEnabled")})))
        if toks:
            self.tokens = toks[: a.p_cap]
            self.store.universe(time.time(), "poly", rows[: a.p_cap])
            log(f"polymarket universe: {len(self.tokens)} tokens")

    def poll_books(self) -> tuple[int, int]:
        q_rows, nd = [], 0
        ts = time.time()
        for i in range(0, len(self.tokens), 50):
            js = self.cli.call("poly", "POST", f"{CLOB}/books", json=[{"token_id": t} for t in self.tokens[i:i + 50]])
            for b in js or []:
                bids = sorted([[f(x["price"]), f(x["size"])] for x in b.get("bids", [])], key=lambda x: -x[0])[:10]
                asks = sorted([[f(x["price"]), f(x["size"])] for x in b.get("asks", [])], key=lambda x: x[0])[:10]
                tok = b.get("asset_id")
                q_rows.append((tok, bids[0][0] if bids else None, asks[0][0] if asks else None,
                               bids[0][1] if bids else None, asks[0][1] if asks else None,
                               f(b.get("last_trade_price")), None, None))
                nd += self.store.depth(ts, "poly", tok, bids, asks)
        nq = self.store.quotes(ts, "poly", q_rows) if q_rows else 0
        return nq, nd


def status(db_path: str) -> None:
    db = sqlite3.connect(db_path)
    for tbl in ("cycles", "quotes", "depth", "trades", "universe", "spot"):
        n = db.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"{tbl:9s} {n:>10,d}")
    for v, lo, hi in db.execute("SELECT venue, MIN(ts), MAX(ts) FROM cycles GROUP BY venue"):
        print(f"{v}: {datetime.fromtimestamp(lo, timezone.utc):%Y-%m-%d %H:%M} -> "
              f"{datetime.fromtimestamp(hi, timezone.utc):%Y-%m-%d %H:%M} UTC ({(hi - lo) / 3600:.2f} h)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "live", "live.sqlite"))
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--status", action="store_true")
    # Kalshi answered one HTTP 429 to a ~5 req/s burst unauthenticated, so the default stays near 3 req/s average.
    ap.add_argument("--k-rate", type=float, default=4.0)
    ap.add_argument("--p-rate", type=float, default=3.0)
    ap.add_argument("--quote-sec", type=float, default=4.0)
    ap.add_argument("--depth-sec", type=float, default=30.0)
    ap.add_argument("--fast-depth-sec", type=float, default=4.0)
    ap.add_argument("--k-fast-depth", type=int, default=2)
    ap.add_argument("--trade-sec", type=float, default=4.0)
    ap.add_argument("--spot-sec", type=float, default=1.0)
    ap.add_argument("--spot", default="BTC-USD,ETH-USD")
    ap.add_argument("--reselect-min", type=float, default=20.0)
    ap.add_argument("--bootstrap-min", type=float, default=60.0)
    ap.add_argument("--k-events", type=int, default=30)
    ap.add_argument("--k-max-per-event", type=int, default=40)
    ap.add_argument("--k-top-markets", type=int, default=150)
    ap.add_argument("--k-cap", type=int, default=400)
    ap.add_argument("--k-depth", type=int, default=20)
    ap.add_argument("--p-cap", type=int, default=300)
    ap.add_argument("--p-max-per-event", type=int, default=25)
    args = ap.parse_args()
    if args.status:
        status(args.db); return

    store = Store(args.db)
    cli = Client(args.k_rate, args.p_rate)
    k, p = Kalshi(cli, store, args), Poly(cli, store, args)
    sp = Spot(cli, store, [s for s in args.spot.split(",") if s])
    stop_at = time.time() + args.hours * 3600
    log(f"collector start -> {args.db} for {args.hours} h")
    n0 = k.poll_trades(max_pages=60)
    log(f"bootstrap: {n0} kalshi trades")
    k.refresh_hot(); k.reselect(); p.reselect()

    def loop(name, period, fn):
        while time.time() < stop_at:
            t0 = time.time()
            try:
                fn()
            except Exception as e:  # never let one bad payload kill the collector
                log(f"{name} error: {type(e).__name__}: {e}")
            time.sleep(max(0.0, period - (time.time() - t0)))

    def reselect_both():
        k.reselect(); p.reselect()

    threads = [threading.Thread(target=loop, args=a, daemon=True) for a in (
        ("k_trades", args.trade_sec, k.poll_trades),
        ("k_quotes", args.quote_sec, k.poll_quotes),
        ("k_depth", args.depth_sec, k.poll_depth),
        ("k_fast_depth", args.fast_depth_sec, k.poll_fast_depth),
        ("k_hot", 30.0, k.refresh_hot),
        ("p_books", args.quote_sec, p.poll_books),
        ("spot", args.spot_sec, sp.poll),
    )]
    for t in threads:
        t.start()
    last_rs, last_log = time.time(), time.time()
    while time.time() < stop_at:
        time.sleep(5)
        if time.time() - last_rs > args.reselect_min * 60:
            try:
                reselect_both()
            except Exception as e:
                log(f"reselect error: {e}")
            last_rs = time.time()
        if time.time() - last_log > 300:
            log(f"requests so far: {dict(cli.n)}")
            last_log = time.time()
    log("collector done")


if __name__ == "__main__":
    sys.exit(main())
