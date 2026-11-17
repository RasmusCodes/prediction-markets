"""Reproducible experiment runner.

Each experiment writes runs/<UTC timestamp>_<name>/ with
  manifest.json   config overrides, seeds, config hash, code hash (sha256 of every pmkt/*.py), python and
                  package versions, start/end time, wall-clock seconds
  results.csv     one row per (arm, seed) with the requested metrics
Runs fan out over a process pool; each (arm, seed) is independent and fully determined by its config.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def code_hash() -> str:
    h = hashlib.sha256()
    pk = os.path.join(ROOT, "pmkt")
    for d, _, files in sorted(os.walk(pk)):
        for f in sorted(files):
            if f.endswith(".py"):
                with open(os.path.join(d, f), "rb") as fh:
                    h.update(f.encode())
                    h.update(fh.read())
    return h.hexdigest()[:16]


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def summarize_sim(sim, ctx) -> dict:
    """Default metric bundle for one finished simulation."""
    from .metrics import agent_report, frames, info_halflife, market_quality
    fr = frames(sim)
    rep = agent_report(sim, fr)
    mq = market_quality(sim, fr)
    hl = info_halflife(sim, fr)
    out = {"events": sim.n_events, "trades": len(fr["trades"]), "mae": float(mq.mae.mean()),
           "rmse": float(mq.rmse.mean()), "spread_c": float(mq.spread_c.mean()), "two_sided": float(mq.two_sided.mean()),
           "depth_top": float(mq.depth_top.mean()),
           "t50_med": float(hl.t50.median()) if len(hl) else float("nan"),
           "t90_med": float(hl.t90.median()) if len(hl) else float("nan"),
           "t50_frac_reached": float(hl.t50.notna().mean()) if len(hl) else float("nan")}
    for kind, g in rep.groupby("kind"):
        out[f"pnl_{kind}"] = float(g.pnl.sum())
        out[f"vol_{kind}"] = float(g.volume.sum())
        if kind in ("mm", "naive_mm", "amm"):
            out[f"mo5_{kind}"] = float(g.mo_5s.mean())
            out[f"sc_{kind}"] = float(g.spread_capture_c.mean())
            out[f"dd_{kind}"] = float(g.max_dd.max())
    return out


def _one(job):
    arm, seed, overrides, fn_name = job
    from ..sim.scenario import build, merge
    t0 = time.time()
    ov = merge(overrides, {"seed": seed})
    sim, ctx = build(overrides=ov)
    sim.run()
    fn = summarize_sim if fn_name is None else _resolve(fn_name)
    res = fn(sim, ctx)
    res.update({"arm": arm, "seed": seed, "wall_s": round(time.time() - t0, 1), "config_hash": ctx["hash"]})
    return res


def _resolve(path: str):
    mod, _, name = path.rpartition(".")
    return getattr(__import__(mod, fromlist=[name]), name)


def _pin_blas_threads() -> None:
    """One BLAS thread per worker: N processes x default OpenBLAS pools exhausted memory on a 16-core laptop."""
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"


def run_experiment(name: str, arms: dict[str, dict], seeds: list[int], metric_fn: str | None = None,
                   workers: int | None = None, out_root: str | None = None,
                   out_dir: str | None = None) -> tuple[pd.DataFrame, str]:
    """arms: {arm_name: config overrides}. metric_fn: dotted path of fn(sim, ctx) -> dict (default summarize_sim).

    Results stream to results.jsonl as each run finishes; passing an existing out_dir resumes, skipping the
    (arm, seed) pairs already recorded. List-valued metrics (e.g. per-market calibration rows) are written to
    <key>.csv with arm and seed columns instead of results.csv."""
    _pin_blas_threads()
    out_root = out_root or os.path.join(ROOT, "runs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_dir or os.path.join(out_root, f"{stamp}_{name}")
    os.makedirs(out_dir, exist_ok=True)
    jl = os.path.join(out_dir, "results.jsonl")
    done = set()
    if os.path.exists(jl):
        with open(jl) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["arm"], r["seed"]))
                except (ValueError, KeyError):
                    pass
    jobs = [(arm, s, ov, metric_fn) for arm, ov in arms.items() for s in seeds if (arm, s) not in done]
    t0 = time.time()
    manifest = {"name": name, "started_utc": stamp, "arms": arms, "seeds": seeds, "metric_fn": metric_fn,
                "code_hash": code_hash(), "versions": versions(), "n_jobs": len(jobs) + len(done),
                "resumed_from": len(done)}
    workers = workers or min(max(len(jobs), 1), max(1, (os.cpu_count() or 2) - 1))
    with open(jl, "a") as fh:
        if workers > 1 and len(jobs) > 1:
            with Pool(workers, maxtasksperchild=4) as p:
                for r in p.imap_unordered(_one, jobs):
                    fh.write(json.dumps(r, default=float) + "\n")
                    fh.flush()
        else:
            for j in jobs:
                fh.write(json.dumps(_one(j), default=float) + "\n")
                fh.flush()
    rows = []
    with open(jl) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    extras: dict[str, list] = {}
    flat = []
    for r in rows:
        f = {}
        for k, v in r.items():
            if isinstance(v, list):
                extras.setdefault(k, []).extend({**x, "arm": r["arm"], "seed": r["seed"]} for x in v)
            else:
                f[k] = v
        flat.append(f)
    df = pd.DataFrame(flat)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    for k, v in extras.items():
        pd.DataFrame(v).to_csv(os.path.join(out_dir, f"{k}.csv"), index=False)
    manifest["wall_s"] = round(time.time() - t0, 1)
    manifest["finished_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return df, out_dir
