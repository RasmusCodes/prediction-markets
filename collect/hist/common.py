import time, json, sqlite3, threading, requests, random, os, sys
UA = {"User-Agent": "pmkt-research/0.1 (read-only market data)"}
RATES = {"api.elections.kalshi.com": 1.5, "gamma-api.polymarket.com": 2.0,
         "clob.polymarket.com": 2.0, "api.exchange.coinbase.com": 2.0}
_last = {}
_lock = threading.Lock()
_sess = requests.Session()
_sess.headers.update(UA)
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "hist"))
KBASE = "https://api.elections.kalshi.com/trade-api/v2"

def _wait(host):
    r = RATES.get(host, 1.0)
    with _lock:
        now = time.time()
        t = max(now, _last.get(host, 0) + 1.0 / r)
        _last[host] = t
    if t > now:
        time.sleep(t - now)

def get(url, params=None, tries=8, allow404=False):
    host = url.split("/")[2]
    delay = 2.0
    for i in range(tries):
        _wait(host)
        try:
            r = _sess.get(url, params=params, timeout=40)
        except requests.RequestException as e:
            time.sleep(delay); delay = min(delay * 2, 60); continue
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                time.sleep(delay); delay = min(delay * 2, 60); continue
        if r.status_code == 404 and allow404:
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            ra = r.headers.get("Retry-After")
            try: w = float(ra)
            except Exception: w = delay
            time.sleep(max(w, delay)); delay = min(delay * 2, 60); continue
        if r.status_code == 400 and allow404:
            return None
        raise RuntimeError(f"HTTP {r.status_code} {url} {params} {r.text[:200]}")
    raise RuntimeError(f"gave up {url} {params}")

def db(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)   # data/hist/ is not in the repository
    c = sqlite3.connect(path, timeout=60)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("CREATE TABLE IF NOT EXISTS fetch_log(key TEXT PRIMARY KEY, done_ts REAL, note TEXT)")
    return c

def done(c, key):
    return c.execute("SELECT 1 FROM fetch_log WHERE key=?", (key,)).fetchone() is not None

def mark(c, key, note=""):
    c.execute("INSERT OR REPLACE INTO fetch_log VALUES(?,?,?)", (key, time.time(), note))
    c.commit()

def iso2ts(s):
    if not s: return None
    from datetime import datetime, timezone
    s = s.replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    if d.tzinfo is None: d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

class Clock:
    def __init__(self, minutes): self.end = time.time() + minutes * 60
    def over(self): return time.time() > self.end
