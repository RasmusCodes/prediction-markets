"""Kalshi stage 1: series list + global scan of settled events (index). Resumable."""
import argparse, json, re, time, os
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--max-minutes", type=float, default=9); a = ap.parse_args()
clk = Clock(a.max_minutes)
c = db(os.path.join(DATA, "kalshi_hist.sqlite"))
c.executescript("""
CREATE TABLE IF NOT EXISTS series(ticker TEXT PRIMARY KEY, title TEXT, category TEXT, frequency TEXT, fee_type TEXT, fee_multiplier REAL, raw_json TEXT);
CREATE TABLE IF NOT EXISTS event_index(event_ticker TEXT PRIMARY KEY, series_ticker TEXT, category TEXT, title TEXT, mutually_exclusive INTEGER, raw_json TEXT);
""")
if not done(c, "series_list"):
    d = get(KBASE + "/series")
    for s in d["series"]:
        c.execute("INSERT OR REPLACE INTO series VALUES(?,?,?,?,?,?,?)", (s["ticker"], s.get("title"), s.get("category"), s.get("frequency"), s.get("fee_type"), fnum(s.get("fee_multiplier")), json.dumps(s)))
    c.commit(); mark(c, "series_list", str(len(d["series"]))); print("series", len(d["series"]))
if done(c, "eventscan_done"):
    print("event scan already complete"); raise SystemExit
r = c.execute("SELECT note FROM fetch_log WHERE key='eventscan_cursor'").fetchone()
cur = r[0] if r and r[0] else None
pages = 0
while not clk.over():
    p = {"limit": 200, "status": "settled"}
    if cur: p["cursor"] = cur
    d = get(KBASE + "/events", p)
    for e in d["events"]:
        st = e.get("series_ticker") or e["event_ticker"].split("-")[0]
        if st.startswith("KXMVE") or e["event_ticker"].startswith("KXMVE"): continue
        c.execute("INSERT OR IGNORE INTO event_index VALUES(?,?,?,?,?,?)", (e["event_ticker"], st, e.get("category"), e.get("title"), int(bool(e.get("mutually_exclusive"))), json.dumps(e)))
    cur = d.get("cursor")
    pages += 1
    if not cur or not d["events"]:
        mark(c, "eventscan_done"); print("scan complete"); break
    c.execute("INSERT OR REPLACE INTO fetch_log VALUES('eventscan_cursor',?,?)", (time.time(), cur)); c.commit()
    if pages % 50 == 0:
        print("pages", pages, c.execute("SELECT count(*) FROM event_index").fetchone()[0], flush=True)
print("pages this run", pages, "events", c.execute("SELECT count(*) FROM event_index").fetchone()[0])
