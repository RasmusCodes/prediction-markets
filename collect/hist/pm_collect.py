"""Polymarket calibration sample. Stages: cands (enumerate+sample), prices (hourly token0 history). Resumable."""
import argparse, json, re, os, random, time
from datetime import datetime, timezone, timedelta
from common import *
G = "https://gamma-api.polymarket.com"; CL = "https://clob.polymarket.com/prices-history"
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["cands", "prices"]); ap.add_argument("--max-minutes", type=float, default=9)
ap.add_argument("--top", type=int, default=6000); ap.add_argument("--rand-per-day", type=int, default=3)
ap.add_argument("--limit-days", type=int, default=0, help="test: only first N random days"); ap.add_argument("--max-items", type=int, default=0)
ap.add_argument("--db", default="polymarket_hist.sqlite")
a = ap.parse_args(); clk = Clock(a.max_minutes)
c = db(os.path.join(DATA, a.db))
c.executescript("""
CREATE TABLE IF NOT EXISTS markets(condition_id TEXT PRIMARY KEY, question TEXT, slug TEXT, event_slug TEXT, event_id TEXT, neg_risk INTEGER, neg_risk_market_id TEXT,
 outcomes TEXT, final_prices TEXT, winner_index INTEGER, token0 TEXT, token1 TEXT, start_ts REAL, end_ts REAL, closed_ts REAL, volume REAL, liquidity REAL,
 fees_enabled INTEGER, category TEXT, tags TEXT, sample_group TEXT, series_key TEXT, raw_json TEXT);
CREATE TABLE IF NOT EXISTS prices(token_id TEXT, t INTEGER, p REAL, PRIMARY KEY(token_id,t)) WITHOUT ROWID;
""")
def pts(s):
    if not s: return None
    s = str(s).strip().replace(" ", "T")
    if re.search(r"[+-]\d\d$", s): s += ":00"
    return iso2ts(s)
def jl(x):
    if isinstance(x, list): return x
    try: return json.loads(x)
    except Exception: return None
def series_key(m):
    ev = (m.get("events") or [{}])[0]
    slug = ev.get("slug") or m.get("slug") or ""
    if re.search(r"up or down|updown", (m.get("question") or "").lower() + " " + slug.lower()):
        return "updown:" + re.sub(r"\d.*$", "", slug)
    return re.sub(r"[-_]?\d.*$", "", slug)
def qualify(m):
    oc = jl(m.get("outcomes")); pr = jl(m.get("outcomePrices")); tk = jl(m.get("clobTokenIds"))
    if not (oc and pr and tk and len(oc) == 2 and len(pr) == 2 and len(tk) == 2): return None
    try: pf = [float(x) for x in pr]
    except Exception: return None
    if not (max(pf) >= 0.999 and min(pf) <= 0.001): return None
    cl = pts(m.get("closedTime")) or pts(m.get("endDate"))
    if not cl or cl < iso2ts("2023-01-01T00:00:00Z"): return None
    return dict(oc=oc, pf=pf, tk=tk, cl=cl)
counts = {}
for k, n in c.execute("SELECT series_key,count(*) FROM markets GROUP BY 1"): counts[k] = n
CAP = 300
def insert(m, q, group):
    sk = series_key(m)
    if counts.get(sk, 0) >= CAP: return False
    ev = (m.get("events") or [{}])[0]
    tags = ev.get("tags")
    cur = c.execute("INSERT OR IGNORE INTO markets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        m["conditionId"], m.get("question"), m.get("slug"), ev.get("slug"), str(ev.get("id")) if ev.get("id") is not None else None,
        int(bool(m.get("negRisk"))), m.get("negRiskMarketID"), json.dumps(q["oc"]), json.dumps(q["pf"]), q["pf"].index(max(q["pf"])), q["tk"][0], q["tk"][1],
        pts(m.get("startDate")), pts(m.get("endDate")), q["cl"], fnum(m.get("volumeNum")), fnum(m.get("liquidityNum")),
        int(bool(m.get("feesEnabled"))), m.get("category") or ev.get("category"),
        json.dumps([t.get("label") for t in tags]) if isinstance(tags, list) else None,
        group, sk, json.dumps(m)))
    if cur.rowcount:
        counts[sk] = counts.get(sk, 0) + 1
        return True
    return False
def keyset(params):
    cur = None
    while True:
        p = dict(params); p["limit"] = 100
        if cur: p["after_cursor"] = cur
        d = get(G + "/markets/keyset", p)
        yield d.get("markets", [])
        cur = d.get("next_cursor")
        if not cur or not d.get("markets"): return
if a.stage == "cands":
    if not done(c, "top_done"):
        n = c.execute("SELECT count(*) FROM markets WHERE sample_group='top'").fetchone()[0]
        r = c.execute("SELECT note FROM fetch_log WHERE key='top_cursor'").fetchone()
        cur = r[0] if r and r[0] else None
        while n < a.top and not clk.over():
            p = {"limit": 100, "closed": "true", "order": "volumeNum", "ascending": "false", "end_date_min": "2023-01-01T00:00:00Z"}
            if cur: p["after_cursor"] = cur
            d = get(G + "/markets/keyset", p)
            for m in d.get("markets", []):
                q = qualify(m)
                if q and n < a.top and insert(m, q, "top"): n += 1
            cur = d.get("next_cursor")
            c.execute("INSERT OR REPLACE INTO fetch_log VALUES('top_cursor',?,?)", (time.time(), cur or "")); c.commit()
            print("top n", n, flush=True)
            if not cur or not d.get("markets"): break
        if n >= a.top or not cur: mark(c, "top_done", str(n))
    if done(c, "top_done"):
        random.seed(12345)
        day = datetime(2023, 1, 1, tzinfo=timezone.utc); end = datetime.now(timezone.utc); nd = 0
        while day < end and not clk.over():
            nd += 1
            if a.limit_days and nd > a.limit_days: break
            key = "rand_day_" + day.strftime("%Y%m%d")
            if not done(c, key):
                pool = []; pages = 0
                for ms in keyset({"closed": "true", "volume_num_min": 1000, "volume_num_max": 50000,
                                  "end_date_min": day.strftime("%Y-%m-%dT00:00:00Z"), "end_date_max": (day + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")}):
                    for m in ms:
                        q = qualify(m)
                        if q and not c.execute("SELECT 1 FROM markets WHERE condition_id=?", (m["conditionId"],)).fetchone(): pool.append((m, q))
                    pages += 1
                    if pages >= 5: break
                random.shuffle(pool); k = 0
                for m, q in pool:
                    if k >= a.rand_per_day: break
                    if insert(m, q, "random"): k += 1
                mark(c, key, "pool=%d picked=%d pages=%d" % (len(pool), k, pages))
            day += timedelta(days=1)
        print("days processed", nd)
    print(dict(c.execute("SELECT sample_group,count(*) FROM markets GROUP BY 1").fetchall()))
else:
    rows = c.execute("SELECT condition_id, token0, closed_ts FROM markets ORDER BY (sample_group='top') DESC, volume DESC").fetchall()
    nproc = 0
    for cid, tok, cl in rows:
        if clk.over() or (a.max_items and nproc >= a.max_items): break
        key = "px_" + cid
        if done(c, key): continue
        total = 0; hi = int(cl)
        lo_lim = int(cl - 30 * 86400)
        # newest chunk first: [hi-14d, hi]; older chunk only if history reaches back to chunk start
        mid = max(lo_lim, hi - 15 * 86400)
        chunks = [(mid, hi), (lo_lim, mid - 1)]
        err = None
        for ci, (s, e) in enumerate(chunks):
            if e - s < 3600: break
            try:
                r = get(CL, {"market": tok, "startTs": s, "endTs": e, "fidelity": 60}, allow404=True)
            except Exception as ex:
                err = str(ex)[:150]; break
            h = (r or {}).get("history") or []
            c.executemany("INSERT OR IGNORE INTO prices VALUES(?,?,?)", [(tok, int(x["t"]), float(x["p"])) for x in h])
            total += len(h)
            if not h or min(x["t"] for x in h) > s + 3 * 3600: break
        if err and total == 0 and "400" not in err:
            print("ERR", cid[:12], err); continue
        mark(c, key, ("n=%d" % total) + (" err=" + err if err else "")); nproc += 1
        if nproc % 100 == 0: print("px done", nproc, flush=True)
    print("processed", nproc, "remaining", sum(1 for r in rows if not done(c, "px_" + r[0])))
