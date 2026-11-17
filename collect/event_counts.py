"""Fetch the true number of markets (and the mutually-exclusive flag) for every Kalshi event in the live universe,
so the arbitrage study can tell complete outcome sets from truncated ones. Read-only, ~2 req/s, cached.

    python collect/event_counts.py
"""
import os
import sqlite3
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = os.path.join(ROOT, "data", "live", "live.sqlite")
OUT = os.path.join(ROOT, "data", "live", "kalshi_event_counts.csv")
K = "https://api.elections.kalshi.com/trade-api/v2"


def main():
    db = sqlite3.connect(f"file:{LIVE}?mode=ro", uri=True)
    evs = sorted({r[0] for r in db.execute("SELECT DISTINCT event FROM universe WHERE venue='kalshi' AND mutually_exclusive=1")})
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["event"])
    s = requests.Session()
    s.headers.update({"User-Agent": "pmkt-research/0.1 (read-only market data)", "Accept": "application/json"})
    rows = done.to_dict("records")
    for ev in evs:
        if ev in set(done.event):
            continue
        for attempt in range(4):
            r = s.get(f"{K}/events/{ev}", params={"with_nested_markets": "true"}, timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            break
        if r.status_code != 200:
            rows.append({"event": ev, "n_markets": None, "mutually_exclusive": None, "status": r.status_code})
        else:
            js = r.json()
            e = js.get("event", js)
            mk = e.get("markets") or js.get("markets") or []
            rows.append({"event": ev, "n_markets": len(mk), "mutually_exclusive": e.get("mutually_exclusive"),
                         "status": 200})
        time.sleep(0.5)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(len(rows), "events ->", OUT)


if __name__ == "__main__":
    main()
