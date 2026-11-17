"""Dataset 3: Kalshi KXBTC15M/KXETH15M (last 30d) + Coinbase 1-minute spot candles. Resumable.
stages: coinbase | markets | candles"""
import argparse, json, os, time
from datetime import datetime, timezone
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["coinbase", "markets", "candles"]); ap.add_argument("--max-minutes", type=float, default=9)
ap.add_argument("--days", type=int, default=30); ap.add_argument("--db", default="crypto15m.sqlite"); ap.add_argument("--batch", type=int, default=10)
a = ap.parse_args(); clk = Clock(a.max_minutes)
c = db(os.path.join(DATA, a.db))
c.executescript("""
CREATE TABLE IF NOT EXISTS k_markets(ticker TEXT PRIMARY KEY, series_ticker TEXT, event_ticker TEXT, open_ts REAL, close_ts REAL, expiration_ts REAL,
 floor_strike REAL, strike_type TEXT, result TEXT, settlement_value REAL, expiration_value TEXT, volume REAL, open_interest REAL, last_price REAL, partition TEXT, raw_json TEXT);
CREATE TABLE IF NOT EXISTS k_candles(ticker TEXT, end_ts INTEGER, bid_close REAL, ask_close REAL, price_close REAL, price_mean REAL, volume REAL, open_interest REAL, PRIMARY KEY(ticker,end_ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS cb_candles(product TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL, PRIMARY KEY(product,ts)) WITHOUT ROWID;
""")
NOW = int(time.time()); START = (NOW - a.days * 86400) // 60 * 60
if a.stage == "coinbase":
    for prod in ("BTC-USD", "ETH-USD"):
        s = START
        while s < NOW and not clk.over():
            e = min(s + 300 * 60, NOW)
            if e - 60 <= s: break
            key = "cb_%s_%d" % (prod, s)
            if not done(c, key):
                iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                d = get("https://api.exchange.coinbase.com/products/%s/candles" % prod, {"granularity": 60, "start": iso(s), "end": iso(e - 60)})
                c.executemany("INSERT OR IGNORE INTO cb_candles VALUES(?,?,?,?,?,?,?)", [(prod, int(x[0]), x[3], x[2], x[1], x[4], x[5]) for x in d])
                mark(c, key, str(len(d)))
            s = e
    print("cb rows", c.execute("SELECT product,count(*),min(ts),max(ts) FROM cb_candles GROUP BY 1").fetchall())
elif a.stage == "markets":
    for ser in ("KXBTC15M", "KXETH15M"):
        key = "mk_" + ser
        if done(c, key): continue
        cur = None
        while not clk.over():
            p = {"series_ticker": ser, "limit": 1000, "min_close_ts": START}
            if cur: p["cursor"] = cur
            d = get(KBASE + "/markets", p)
            for m in d["markets"]:
                c.execute("INSERT OR REPLACE INTO k_markets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    m["ticker"], ser, m["event_ticker"], iso2ts(m.get("open_time")), iso2ts(m.get("close_time")), iso2ts(m.get("expiration_time")),
                    fnum(m.get("floor_strike")), m.get("strike_type"), m.get("result"), fnum(m.get("settlement_value_dollars")), m.get("expiration_value"),
                    fnum(m.get("volume_fp")), fnum(m.get("open_interest_fp")), fnum(m.get("last_price_dollars")), "live", json.dumps(m)))
            c.commit()
            cur = d.get("cursor")
            if not cur or not d["markets"]:
                mark(c, key); break
    print(c.execute("SELECT series_ticker,count(*),min(close_ts),max(close_ts),sum(result='yes'),sum(result='no') FROM k_markets GROUP BY 1").fetchall())
else:
    rows = c.execute("SELECT ticker, series_ticker, open_ts, close_ts FROM k_markets WHERE result IN ('yes','no') ORDER BY series_ticker, close_ts").fetchall()
    todo = [r for r in rows if not done(c, "kc_" + r[0])]
    print("todo markets", len(todo))
    i = 0
    while i < len(todo) and not clk.over():
        grp = todo[i:i + a.batch]; i += a.batch
        # batch must be same-series? candlestick batch is series-agnostic; window covers all members
        s = int(min(r[2] for r in grp)) - 60; e = int(max(r[3] for r in grp)) + 120
        d = get(KBASE + "/markets/candlesticks", {"market_tickers": ",".join(r[0] for r in grp), "start_ts": s, "end_ts": e, "period_interval": 1})
        got = {m["market_ticker"]: m.get("candlesticks", []) for m in d["markets"]}
        for r in grp:
            lo, hi = r[2], r[3] + 60
            rowsout = []
            for k in got.get(r[0], []):
                t = k["end_period_ts"]
                if t < lo or t > hi + 60: continue
                pr = k.get("price") or {}
                rowsout.append((r[0], t, fnum((k.get("yes_bid") or {}).get("close_dollars")), fnum((k.get("yes_ask") or {}).get("close_dollars")),
                                fnum(pr.get("close_dollars")), fnum(pr.get("mean_dollars")), fnum(k.get("volume_fp")), fnum(k.get("open_interest_fp"))))
            c.executemany("INSERT OR IGNORE INTO k_candles VALUES(?,?,?,?,?,?,?,?)", rowsout)
            c.execute("INSERT OR REPLACE INTO fetch_log VALUES(?,?,?)", ("kc_" + r[0], time.time(), str(len(rowsout))))
        c.commit()
        if (i // a.batch) % 20 == 0: print("candle batches", i // a.batch, flush=True)
    print("remaining", sum(1 for r in rows if not done(c, "kc_" + r[0])), "candles", c.execute("SELECT count(*) FROM k_candles").fetchone()[0])
