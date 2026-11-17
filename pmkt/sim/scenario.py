"""Build a complete simulation (world, venues, competing traders, our strategies) from one config dict.

Every random stream is derived from the config seed with numpy SeedSequence: the world path uses its own child
seeds, and each agent gets its own, so changing one strategy's parameters leaves the information path and the
other agents' random draws unchanged (common random numbers across experiment arms).
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from ..core.exchange import Exchange, MarketSpec
from ..core.fees import fee_from_spec
from ..pricing.liquidity import QuoteParams
from ..risk.risk import RiskLimits, RiskManager, structures_from_world
from ..world.truth import BinaryFamily, LadderFamily, PairFamily, World, make_grid
from . import agents as A
from .engine import Simulator

CENT = 100  # cents per dollar

DEFAULT_CONFIG = {
    "name": "showcase",
    "seed": 7,
    "T": 2 * 3600.0,
    "dt": 1.0,
    "record_dt": 5.0,
    # Variance budget per family (shares of total_var): diffusion = 1 - jump_share - sum(sched shares) - terminal.
    # sched entries are [time as a fraction of T, share of total_var revealed then]; terminal_share is revealed at
    # T, after trading closes (the event itself).
    "terminal_share": 0.5,
    "families": [
        {"type": "binary", "id": "ELECT", "p0": 0.56, "total_var": 1.0, "jump_share": 0.25, "n_jumps": 5,
         "sched": [[0.5, 0.15]], "title": "Will candidate X win the election?"},
        {"type": "ladder", "id": "CPI", "strikes": [-0.6, -0.2, 0.2, 0.6], "x0": 0.05, "total_var": 0.8,
         "jump_share": 0.2, "n_jumps": 4, "sched": [[0.75, 0.2]], "title": "Core CPI surprise (s.d.)"},
        {"type": "pair", "id": "FED", "pA": 0.62, "pB": 0.45, "rho": 0.7, "total_var": 1.0, "sched": [[0.75, 0.2]],
         "titles": ["Fed cuts in December", "Fed cuts in January"]},
        {"type": "binary", "id": "BTC", "p0": 0.5, "total_var": 1.0, "jump_share": 0.2, "n_jumps": 8,
         "terminal_share": 0.3, "title": "BTC above $70k at the close?"},
    ],
    "venues": {
        "KX": {"fee": {"kind": "quadratic", "taker_rate": 0.07, "maker_rate": 0.0}, "msg_rate": 10, "conversion": False},
        "PM": {"fee": {"kind": "zero"}, "msg_rate": 20, "conversion": True},
    },
    "listings": {"KX": ["ELECT", "CPI", "FED"], "PM": ["ELECT", "BTC", "CPI"]},
    "agents": {
        "noise": {"n": 10, "rate": 0.25, "mean_size": 8, "longshot_bias": 0.15, "limit_prob": 0.35,
                  "capital": 10_000, "latency": 0.25, "jitter": 0.2},
        "informed": {"n": 6, "rate": 0.05, "noise_sd": 0.25, "bias_sd": 0.05, "threshold": 0.03, "max_pos": 300,
                     "max_order": 60, "capital": 25_000, "latency": 0.15, "jitter": 0.1, "react_delay": [0.5, 4.0]},
        "momentum": {"n": 2, "rate": 0.05, "lookback": 60, "threshold": 3, "size": 10, "capital": 10_000,
                     "latency": 0.3},
        "naive_mm": {"per_venue": 1, "rate": 0.3, "half_spread": 3, "size": 15, "max_inv": 300, "capital": 50_000,
                     "latency": 0.2},
        "amm": {"venues": ["PM"], "b": 120.0, "alpha": None, "levels": 4, "rate": 0.5, "capital": 50_000,
                "latency": 0.1},
        "arbitrageur": {"n": 1, "rate": 0.3, "min_edge": 1.0, "max_qty": 50, "capital": 50_000, "latency": 0.15},
    },
    "strategies": {
        # portfolio skew off by default: in experiment E5 it lowered P&L and raised drawdown (noisy covariances)
        "mm": {"enabled": True, "fair_mode": "model", "portfolio": False, "rate": 1.0, "latency": 0.03,
               "capital": 100_000, "params": {}, "risk": {}},
        "arb": {"enabled": True, "rate": 2.0, "latency": 0.02, "min_profit": 10.0, "capital": 100_000},
        "mispricing": {"enabled": True, "model": "copula", "threshold": 0.03, "rate": 0.5, "capital": 50_000},
    },
}


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _family(spec: dict, grid: np.ndarray, rng: np.random.Generator, terminal_share: float = 0.0):
    """Variance budget: total_var = diffusion + jumps + scheduled news + terminal resolution.

    `terminal_share` of the budget is revealed at T, after trading has closed (the event itself happening), which
    keeps intraday probability volatility realistic: without it every market would have to travel to 0 or 1
    during the session."""
    T = grid[-1]
    tv = float(spec.get("total_var", 1.0))
    ts = float(spec.get("terminal_share", terminal_share))
    sched = [(float(f) * T, float(v) * tv) for f, v in spec.get("sched", [])]
    if ts > 0:
        sched.append((T, ts * tv))
    sv = sum(v for _, v in sched)
    kind = spec["type"]
    if kind == "pair":
        sigma = np.sqrt(max(tv - sv, 1e-6) / T)
        return PairFamily(spec["id"], grid, spec["pA"], spec["pB"], rho=spec.get("rho", 0.6), sigma=sigma, sched=sched,
                          rng=rng, titles=tuple(spec.get("titles", ("A", "B"))))
    js = float(spec.get("jump_share", 0.0))
    nj = float(spec.get("n_jumps", 0.0))
    sigma = np.sqrt(max(tv * (1 - js) - sv, 1e-6) / T)
    lam = nj / T if nj > 0 and js > 0 else 0.0
    jsd = np.sqrt(tv * js / nj) if lam > 0 else 0.0
    if kind == "binary":
        p0 = float(rng.uniform(*spec["p0_range"])) if "p0_range" in spec else spec.get("p0", 0.5)
        return BinaryFamily(spec["id"], grid, p0, sigma=sigma, lam=lam, jump_sd=jsd, sched=sched,
                            rng=rng, title=spec.get("title", ""))
    if kind == "ladder":
        return LadderFamily(spec["id"], grid, spec["strikes"], x0=spec.get("x0", 0.0), sigma=sigma, lam=lam,
                            jump_sd=jsd, sched=sched, rng=rng, title=spec.get("title", ""),
                            buckets=spec.get("buckets", True), ladder=spec.get("ladder", True), unit=spec.get("unit", ""))
    raise ValueError(kind)


def build(config: dict | None = None, overrides: dict | None = None):
    """Returns (sim, ctx) where ctx holds world, structures, listings and handles to our strategies."""
    cfg = merge(DEFAULT_CONFIG, config or {}) if config is not None else copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        cfg = merge(cfg, overrides)
    ss = np.random.SeedSequence(int(cfg["seed"]))
    world_ss, agent_ss = ss.spawn(2)
    grid = make_grid(cfg["T"], cfg["dt"])
    fam_rngs = [np.random.default_rng(s) for s in world_ss.spawn(len(cfg["families"]))]
    fams = [_family(spec, grid, r, float(cfg.get("terminal_share", 0.0))) for spec, r in zip(cfg["families"], fam_rngs)]
    world = World(fams, grid)

    venues = {}
    listings = []
    for vid, vs in cfg["venues"].items():
        ex = Exchange(vid, fee_from_spec(vs.get("fee")), msg_rate=vs.get("msg_rate"),
                      allow_conversion=bool(vs.get("conversion", False)))
        for fid in cfg["listings"].get(vid, []):
            fam = world.families[fid]
            for md in fam.markets():
                ex.add_market(MarketSpec(md.id, md.title, md.kind, fid, md.event, world.T, dict(md.meta)))
                listings.append((vid, md.id))
        for ev in ex.events:
            ex.set_event_convertible(ev, True)
        venues[vid] = ex

    sim = Simulator(world, venues, seed=int(cfg["seed"]), record_dt=cfg["record_dt"])
    structures = structures_from_world(world)
    p0 = {m: float(world.paths[m][0]) for m in world.paths}
    seeds = iter(int(s.generate_state(1)[0]) for s in agent_ss.spawn(400))
    ag = cfg["agents"]

    def cap(d, default):
        return {v: int(d.get("capital", default) * CENT) for v in venues}

    def kw(d, keys=("rate", "latency", "jitter")):
        return {k: d[k] for k in keys if k in d}

    if ag.get("noise", {}).get("n"):
        d = ag["noise"]
        for i in range(d["n"]):
            bias = d.get("longshot_bias", 0.0)
            sim.add_agent(A.NoiseTrader(f"noise{i}", cap(d, 10_000), listings, mean_size=d.get("mean_size", 8),
                                        longshot_bias=bias, limit_prob=d.get("limit_prob", 0.3),
                                        limit_life=d.get("limit_life", 90.0), seed=next(seeds), **kw(d)))
    if ag.get("informed", {}).get("n"):
        d = ag["informed"]
        for i in range(d["n"]):
            sim.add_agent(A.InformedTrader(f"informed{i}", cap(d, 25_000), listings, noise_sd=d.get("noise_sd", 0.25),
                                           bias_sd=d.get("bias_sd", 0.05), threshold=d.get("threshold", 0.03),
                                           max_pos=d.get("max_pos", 300), max_order=d.get("max_order", 60),
                                           react_prob=d.get("react_prob", 1.0),
                                           react_delay=tuple(d.get("react_delay", (0.5, 3.0))), seed=next(seeds), **kw(d)))
    if ag.get("momentum", {}).get("n"):
        d = ag["momentum"]
        for i in range(d["n"]):
            sim.add_agent(A.MomentumTrader(f"momentum{i}", cap(d, 10_000), listings, lookback=d.get("lookback", 60),
                                           threshold=d.get("threshold", 3), size=d.get("size", 10), seed=next(seeds),
                                           **kw(d)))
    if ag.get("naive_mm", {}).get("per_venue"):
        d = ag["naive_mm"]
        for v in venues:
            for i in range(d["per_venue"]):
                ls = [x for x in listings if x[0] == v]
                sim.add_agent(A.NaiveMarketMaker(f"naive_mm_{v}{i}", cap(d, 50_000), ls, half_spread=d.get("half_spread", 3),
                                                 size=d.get("size", 15), max_inv=d.get("max_inv", 300), p0=p0,
                                                 seed=next(seeds), **kw(d)))
    if ag.get("amm", {}).get("venues"):
        d = ag["amm"]
        for v in d["venues"]:
            if v not in venues:
                continue
            ls = [x for x in listings if x[0] == v]
            sim.add_agent(A.AMMAgent(f"amm_{v}", cap(d, 50_000), ls, b=d.get("b", 120.0), alpha=d.get("alpha"),
                                     levels=d.get("levels", 4), p0=p0, seed=next(seeds), **kw(d)))
    groups, ladders, dups = [], [], []
    for v, ex in venues.items():
        for ev, mids in ex.events.items():
            groups.append((v, list(mids)))
        for fid, fam in world.families.items():
            if isinstance(fam, LadderFamily) and fam.with_ladder:
                ms = [fam._ladder_id(K) for K in fam.strikes]
                if all(m in ex.markets for m in ms):
                    ladders.append((v, ms))
    vids = list(venues)
    for i in range(len(vids)):
        for j in range(i + 1, len(vids)):
            for m in venues[vids[i]].markets:
                if m in venues[vids[j]].markets:
                    dups.append(((vids[i], m), (vids[j], m)))
    if ag.get("arbitrageur", {}).get("n"):
        d = ag["arbitrageur"]
        for i in range(d["n"]):
            sim.add_agent(A.SimpleArbitrageur(f"arbitrageur{i}", cap(d, 50_000), groups, ladders, dups,
                                              min_edge=d.get("min_edge", 1.0), max_qty=d.get("max_qty", 50),
                                              seed=next(seeds), **kw(d)))

    from ..strategies.arbitrage import ArbitrageStrategy
    from ..strategies.market_maker import MarketMaker
    from ..strategies.mispricing import MispricingStrategy
    st = cfg["strategies"]
    ours = {}
    if st.get("mm", {}).get("enabled"):
        d = st["mm"]
        qp = QuoteParams(**d.get("params", {}))
        rl = RiskLimits(**d.get("risk", {}))
        mm = MarketMaker("our_mm", cap(d, 100_000), d.get("listings") or listings, params=qp,
                         fair_mode=d.get("fair_mode", "model"), risk=RiskManager(rl, structures),
                         portfolio=d.get("portfolio", False), pull_s=d.get("pull_s", 20.0),
                         seed=next(seeds), **kw(d))
        for m, fv in mm.fv.items():
            fv.kf.x = float(np.log(p0[m] / (1 - p0[m]))) if 0 < p0[m] < 1 else 0.0
            fv.kf.P = 0.05
        sim.add_agent(mm)
        ours["mm"] = mm
    if st.get("arb", {}).get("enabled"):
        d = st["arb"]
        arb = ArbitrageStrategy("our_arb", cap(d, 100_000), structures, min_profit=d.get("min_profit", 10.0),
                                max_capital=d.get("max_capital", 3_000_00), seed=next(seeds), **kw(d))
        sim.add_agent(arb)
        ours["arb"] = arb
    if st.get("mispricing", {}).get("enabled"):
        d = st["mispricing"]
        model = d.get("model", "copula")
        if model == "copula":
            ls = [x for x in listings if world.defs[x[1]].kind in ("binary", "combo") and world.fam_of[x[1]] in
                  [f for f, fam in world.families.items() if isinstance(fam, PairFamily)]]
        elif model == "consistency":
            ls = [x for x in listings if world.defs[x[1]].kind in ("bucket", "ladder")]
        else:
            ls = listings
        mp = MispricingStrategy("our_mispricing", cap(d, 50_000), ls, model=model, threshold=d.get("threshold", 0.03),
                                kelly=d.get("kelly", 0.25), rho_prior=d.get("rho_prior"), calib_map=d.get("calib_map"),
                                seed=next(seeds), **kw(d))
        sim.add_agent(mp)
        ours["mispricing"] = mp
    if cfg.get("monitor", {}).get("enabled"):
        from ..strategies.arbitrage import ArbMonitor
        mon = ArbMonitor(structures, period=cfg["monitor"].get("period", 5.0),
                         min_profit=cfg["monitor"].get("min_profit", 1.0))
        sim.add_agent(mon)
        ours["monitor"] = mon
    ctx = {"config": cfg, "world": world, "structures": structures, "listings": listings, "ours": ours,
           "groups": groups, "ladders": ladders, "duplicates": dups, "hash": config_hash(
               {k: v for k, v in cfg.items() if k != "strategies"} | {"strategies": {
                   k: {kk: vv for kk, vv in v.items() if kk != "calib_map"} for k, v in cfg["strategies"].items()}})}
    return sim, ctx
