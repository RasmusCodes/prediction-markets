import numpy as np
import pandas as pd

from pmkt.backtest.replay import BUY, SELL, ReplayBacktester
from pmkt.core.fees import QuadraticFee
from pmkt.sim import build
from pmkt.strategies.arbitrage import ArbScanner


def test_short_simulation_conserves_cash_and_is_reproducible():
    over = {"T": 300.0, "seed": 3}
    out = []
    for _ in range(2):
        sim, ctx = build(overrides=over)
        tot0 = sum(a.cash0 for ex in sim.venues.values() for a in ex.accounts.values())
        sim.run()
        tot1 = sum(a.cash + a.reserved for ex in sim.venues.values() for a in ex.accounts.values())
        fees = sum(ex.fees_collected for ex in sim.venues.values())
        assert tot0 == tot1 + fees
        assert all(q == 0 for ex in sim.venues.values() for a in ex.accounts.values() for q in a.pos.values())
        out.append((len(sim.recorder.trades), round(sum(t[3] * t[4] for t in sim.recorder.trades))))
    assert out[0] == out[1]          # same seed -> identical run


def test_lp_scanner_finds_planted_bucket_arbitrage():
    sim, ctx = build(overrides={"T": 120.0, "seed": 1, "agents": {"noise": {"n": 0}, "informed": {"n": 0},
                                                                     "momentum": {"n": 0}, "naive_mm": {"per_venue": 0},
                                                                     "amm": {"venues": []}, "arbitrageur": {"n": 0}},
                                     "strategies": {"mm": {"enabled": False}, "arb": {"enabled": False},
                                                    "mispricing": {"enabled": False}}})
    ex = sim.venues["PM"]
    ex.open_account("planter", 10_000_000)
    buckets = ex.events["CPI:buckets"]
    for m in buckets:                       # asks summing to 5 x 15 = 75c < 100c: buy-all arbitrage
        ex.submit("planter", m, SELL, 15, 10)
    opps = ArbScanner(sim, ctx["structures"], min_profit=1.0).scan()
    cpi = [o for o in opps if o.family == "CPI"]
    assert cpi and cpi[0].profit >= 10 * (100 - 75) * 0.99


def test_replay_queue_model():
    quotes = pd.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "bid": [40, 40, 40, 40], "ask": [42, 42, 42, 42],
                           "bid_sz": [30, 30, 10, 10], "ask_sz": [20, 20, 20, 20]})
    trades = pd.DataFrame({"t": [1.5, 2.5, 2.6], "price": [40, 40, 39], "qty": [15, 12, 5], "taker_side": [SELL, SELL, SELL]})

    class Joiner:
        done = False
        def on_quote(self, bt, t, q):
            if not self.done:
                bt.quote(BUY, 40, 10)
                self.done = True
        def on_trade(self, bt, t, tr):
            pass

    bt = ReplayBacktester(quotes, trades, fee=QuadraticFee(0.07), latency=0.1, result=1)
    r = bt.run(Joiner())
    # joined behind 30 at t=0.1; 15 traded (queue 15), displayed drops to 10 (queue 10), 12 traded -> we get 2,
    # then a trade-through at 39 fills 5 more
    assert list(r.fills.qty) == [2, 5]
    assert r.position == 7 and abs(r.pnl - 7 * 0.60) < 1e-9
