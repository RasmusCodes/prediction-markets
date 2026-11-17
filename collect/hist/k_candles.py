"""Kalshi stage 3: hourly candlesticks for the sampled calibration markets (volume >= 10). Resumable.

Window per market: [max(open, close - 30 days), close + 1 h], period_interval=60.
Primary source: batch GET /markets/candlesticks?market_tickers=... (works for both the live and the historical
partition, `_dollars`/`_fp` field names). Markets that come back empty are retried one by one on
GET /historical/markets/{ticker}/candlesticks (plain field names). Markets are batched by event so the shared
time window stays tight. fetch_log key: kh_<ticker>, note = number of candles stored.
"""
import argparse, os, collections
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--max-minutes", type=float, default=9); ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--days", type=float, default=30); ap.add_argument("--min-volume", type=float, default=10)
ap.add_argument("--db", default="kalshi_hist.sqlite")
a = ap.parse_args(); clk = Clock(a.max_minutes)
c = db(os.path.join(DATA, a.db))
c.executescript("""
CREATE TABLE IF NOT EXISTS candles(ticker TEXT, period INTEGER, end_ts INTEGER, bid_close REAL, ask_close REAL,
  price_close REAL, price_mean REAL, volume REAL, open_interest REAL, PRIMARY KEY(ticker, period, end_ts)) WITHOUT ROWID;
""")


def num(d, *keys):
    d = d or {}
    for k in keys:
        if d.get(k) is not None:
            return fnum(d.get(k))
    return None


def parse(k):
    return (int(k["end_period_ts"]),
            num(k.get("yes_bid"), "close_dollars", "close"), num(k.get("yes_ask"), "close_dollars", "close"),
            num(k.get("price"), "close_dollars", "close"), num(k.get("price"), "mean_dollars", "mean"),
            num(k, "volume_fp", "volume"), num(k, "open_interest_fp", "open_interest"))


rows = c.execute("SELECT ticker, event_ticker, open_ts, close_ts FROM markets WHERE volume >= ? AND close_ts IS NOT NULL "
                 "ORDER BY event_ticker, ticker", (a.min_volume,)).fetchall()
todo = [r for r in rows if not done(c, "kh_" + r[0])]
print("markets eligible", len(rows), "todo", len(todo), flush=True)
byev = collections.defaultdict(list)
for r in todo:
    byev[r[1]].append(r)
groups = []
for ev, rs in byev.items():
    for i in range(0, len(rs), a.batch):
        groups.append(rs[i:i + a.batch])
nb = 0
for grp in groups:
    if clk.over():
        break
    win = {r[0]: (int(max(r[2] or 0, r[3] - a.days * 86400)), int(r[3] + 3600)) for r in grp}
    s, e = min(w[0] for w in win.values()), max(w[1] for w in win.values())
    d = get(KBASE + "/markets/candlesticks", {"market_tickers": ",".join(win), "start_ts": s, "end_ts": e,
                                              "period_interval": 60}, allow404=True) or {}
    got = {m.get("market_ticker"): m.get("candlesticks") or [] for m in d.get("markets", [])}
    for t, (lo, hi) in win.items():
        ks = got.get(t) or []
        if not ks:  # fallback: per-market historical route
            dd = get(KBASE + f"/historical/markets/{t}/candlesticks", {"start_ts": lo, "end_ts": hi, "period_interval": 60},
                     allow404=True) or {}
            ks = dd.get("candlesticks") or []
        out = [(t, 60) + parse(k) for k in ks if lo - 3600 <= int(k["end_period_ts"]) <= hi]
        c.executemany("INSERT OR IGNORE INTO candles VALUES(?,?,?,?,?,?,?,?,?)", out)
        c.execute("INSERT OR REPLACE INTO fetch_log VALUES(?,?,?)", ("kh_" + t, time.time(), str(len(out))))
    c.commit()
    nb += 1
    if nb % 50 == 0:
        print("batches", nb, "of", len(groups), flush=True)
left = sum(1 for r in rows if not done(c, "kh_" + r[0]))
print("remaining", left, "candles", c.execute("SELECT COUNT(*) FROM candles").fetchone()[0], flush=True)
