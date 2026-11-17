"""Kalshi stage 2: choose series/events from event_index and fetch all markets of sampled events. Resumable.
Sampling procedure (deterministic, seed=20261113):
  1. Series = all series with >=1 settled non-KXMVE event in event_index. Event date = YYMONDD token in event ticker (None if absent).
     Eligible events: date >= 2024-01-01 or date unknown (verified later via market close_time).
  2. Series are shuffled within category; categories are visited round-robin (one series per category per round) so small categories
     are not swamped by sports/crypto. Each visited series contributes up to --per-series (15) events.
  3. Within a series eligible events are sorted by date (unknown last); K evenly spaced positions are taken; if a position yields no
     qualifying event (no binary settled markets / >80 markets / close<2024) the nearest unused neighbour (+-1..4) is tried.
  4. Every qualifying market of a sampled event is kept (binary, result yes/no, close_time>=2024-01-01).
  5. Stop at --target-events events or --max-markets markets."""
import argparse, json, os, re, random, time, collections
from datetime import datetime, timezone
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--max-minutes", type=float, default=9); ap.add_argument("--per-series", type=int, default=15)
ap.add_argument("--target-events", type=int, default=4500); ap.add_argument("--max-markets", type=int, default=25000)
ap.add_argument("--max-series", type=int, default=0); ap.add_argument("--db", default="kalshi_hist.sqlite")
a = ap.parse_args(); clk = Clock(a.max_minutes)
c = db(os.path.join(DATA, a.db))
c.executescript("""
CREATE TABLE IF NOT EXISTS events(event_ticker TEXT PRIMARY KEY, series_ticker TEXT, category TEXT, title TEXT, mutually_exclusive INTEGER, raw_json TEXT);
CREATE TABLE IF NOT EXISTS markets(ticker TEXT PRIMARY KEY, event_ticker TEXT, series_ticker TEXT, title TEXT, yes_sub_title TEXT, strike_type TEXT, floor_strike REAL, cap_strike REAL,
 open_ts REAL, close_ts REAL, expiration_ts REAL, result TEXT, settlement_value REAL, volume REAL, open_interest REAL, last_price REAL, partition TEXT, raw_json TEXT);
""")
MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
def edate(t):
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", t)
    if not m or m.group(2) not in MON: return None
    try: return datetime(2000 + int(m.group(1)), MON[m.group(2)], int(m.group(3)), tzinfo=timezone.utc).timestamp()
    except ValueError: return None
T2024 = iso2ts("2024-01-01T00:00:00Z"); CUT = iso2ts("2026-09-13T00:00:00Z")
rows = c.execute("SELECT event_ticker, series_ticker, category FROM event_index").fetchall()
byser = collections.defaultdict(list); sercat = {}
for et, st, ca in rows:
    d = edate(et)
    if d is not None and d < T2024: continue
    byser[st].append((d if d is not None else 9e12, et)); sercat[st] = ca
for st in byser: byser[st].sort()
rnd = random.Random(20261113)
cats = collections.defaultdict(list)
for st in sorted(byser): cats[sercat[st]].append(st)
for ca in cats: rnd.shuffle(cats[ca])
order = []; i = 0
while any(i < len(v) for v in cats.values()):
    for ca in sorted(cats, key=lambda x: str(x)):
        if i < len(cats[ca]): order.append(cats[ca][i])
    i += 1
if a.max_series: order = order[:a.max_series]
print("series in play", len(order), "categories", len(cats), flush=True)
def fetch_markets(et, d):
    part = ["historical", "live"] if (d is not None and d < CUT - 86400) else ["live", "historical"]
    for pt in part:
        url = KBASE + ("/historical/markets" if pt == "historical" else "/markets")
        r = get(url, {"event_ticker": et, "limit": 200}, allow404=True)
        if r and r.get("markets"): return pt, r["markets"]
    return None, []
def qual(m):
    if m.get("market_type") != "binary" or m.get("result") not in ("yes", "no"): return False
    if m["ticker"].startswith("KXMVE"): return False
    ct = iso2ts(m.get("close_time"))
    return ct is not None and ct >= T2024
nev = c.execute("SELECT count(*) FROM events").fetchone()[0]; nmk = c.execute("SELECT count(*) FROM markets").fetchone()[0]
for st in order:
    if clk.over() or nev >= a.target_events or nmk >= a.max_markets: break
    if done(c, "ser_" + st): continue
    ev = byser[st]; n = len(ev); k = min(a.per_series, n); used = set(); got = 0
    for j in range(k):
        pos = round(j * (n - 1) / (k - 1)) if k > 1 else n // 2
        for off in (0, 1, -1, 2, -2, 3, -3, 4, -4):
            q = pos + off
            if q < 0 or q >= n or q in used: continue
            used.add(q)
            d, et = ev[q]
            key = "ev_" + et
            if done(c, key):
                if c.execute("SELECT 1 FROM events WHERE event_ticker=?", (et,)).fetchone(): got += 1; break
                continue
            pt, ms = fetch_markets(et, None if d == 9e12 else d)
            good = [m for m in ms if qual(m)] if pt else []
            ok = bool(good) and len(ms) <= 80 and len(good) >= 1 and nmk + len(good) <= a.max_markets
            if ok:
                er = c.execute("SELECT raw_json FROM event_index WHERE event_ticker=?", (et,)).fetchone()
                e = json.loads(er[0]); e.pop("markets", None)
                c.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?)", (et, st, e.get("category"), e.get("title"), int(bool(e.get("mutually_exclusive"))), json.dumps(e)))
                for m in good:
                    c.execute("INSERT OR IGNORE INTO markets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        m["ticker"], et, st, m.get("title"), m.get("yes_sub_title"), m.get("strike_type"), fnum(m.get("floor_strike")), fnum(m.get("cap_strike")),
                        iso2ts(m.get("open_time")), iso2ts(m.get("close_time")), iso2ts(m.get("expiration_time")), m.get("result"),
                        fnum(m.get("settlement_value_dollars")), fnum(m.get("volume_fp")), fnum(m.get("open_interest_fp")), fnum(m.get("last_price_dollars")), pt, json.dumps(m)))
                nev += 1; nmk += len(good); got += 1
            mark(c, key, "ok" if ok else "reject n=%d good=%d" % (len(ms), len(good)))
            if ok: break
        if nev >= a.target_events: break
    mark(c, "ser_" + st, str(got))
    if len(done_s := c.execute("SELECT 1 FROM fetch_log WHERE key LIKE 'ser_%'").fetchall()) % 25 == 0:
        print("series done", len(done_s), "events", nev, "markets", nmk, flush=True)
print("events", nev, "markets", nmk, "series done", c.execute("SELECT count(*) FROM fetch_log WHERE key LIKE 'ser_%'").fetchone()[0], "of", len(order))
