import random

import pytest

from pmkt.core.exchange import Exchange, MarketSpec
from pmkt.core.fees import QuadraticFee, ZeroFee
from pmkt.core.orderbook import BUY, SELL, Order, OrderBook


def test_quadratic_fee_matches_published_formula():
    f = QuadraticFee(0.07)
    assert f.taker(50, 100) == 175      # $1.75 on 100 contracts at 50c
    assert f.taker(50, 1) == 2          # 1.75c rounds up to 2c per single-contract fill
    assert f.taker(10, 10) == 7         # 0.07*10*0.09 = $0.063 -> 7c
    assert f.maker(50, 100) == 0
    assert QuadraticFee(0.07, 0.0175).maker(50, 100) == 44  # 43.75c -> 44c


def test_price_time_priority_and_price_improvement():
    b = OrderBook("X")
    b.add(Order(1, "A", "X", SELL, 55, 5))
    b.add(Order(2, "B", "X", SELL, 55, 5))
    b.add(Order(3, "C", "X", SELL, 54, 2))
    r = b.add(Order(4, "D", "X", BUY, 60, 4))
    assert [(f.maker, f.price, f.qty) for f in r.fills] == [("C", 54, 2), ("A", 55, 2)]
    assert b.queue_ahead(2) == 3 and b.best_ask == 55
    b.check()


def test_tif_variants_and_post_only():
    b = OrderBook("X")
    b.add(Order(1, "A", "X", SELL, 50, 3))
    r = b.add(Order(2, "B", "X", BUY, 50, 5, tif="IOC"))
    assert sum(f.qty for f in r.fills) == 3 and not r.resting
    b.add(Order(3, "A", "X", SELL, 52, 3))
    assert b.add(Order(4, "B", "X", BUY, 52, 5, tif="FOK")).rejected == "fok_insufficient"
    assert b.add(Order(5, "B", "X", BUY, 52, 1, post_only=True)).rejected == "post_only_would_cross"
    assert b.add(Order(6, "B", "X", BUY, 51, 1, post_only=True)).resting
    b.check()


def test_self_trade_prevention_cancels_resting():
    b = OrderBook("X")
    b.add(Order(1, "A", "X", SELL, 50, 3))
    b.add(Order(2, "B", "X", SELL, 51, 3))
    r = b.add(Order(3, "A", "X", BUY, 51, 2))
    assert [c.oid for c in r.stp_cancelled] == [1]
    assert [(f.maker, f.qty) for f in r.fills] == [("B", 2)]
    b.check()


def test_netting_and_fees():
    ex = Exchange("V", QuadraticFee(0.07))
    ex.add_market(MarketSpec("M"))
    for t in ("A", "B"):
        ex.open_account(t, 100_000)
    ex.submit("B", "M", SELL, 40, 10)               # B rests: buy NO at 60
    r = ex.submit("A", "M", BUY, 45, 10)            # A lifts at 40 (price improvement)
    assert r.status == "filled" and r.fills[0].price == 40
    a = ex.accounts["A"]
    assert a.pos["M"] == 10 and a.reserved == 0
    assert a.cash == 100_000 - 400 - QuadraticFee(0.07).taker(40, 10)
    ex.submit("B", "M", BUY, 45, 10)                # B rests a bid (would close its NO)
    ex.submit("A", "M", SELL, 45, 10)               # A sells into it: A flat, B flat
    assert ex.accounts["A"].pos["M"] == 0 and ex.accounts["B"].pos["M"] == 0
    assert ex.accounts["A"].cash == 100_000 + 50 - ex.accounts["A"].fees_paid
    assert ex.total_cash() + ex.fees_collected == 200_000


def test_mint_and_redeem_full_sets():
    ex = Exchange("V", ZeroFee(), allow_conversion=True)
    for k in "abc":
        ex.add_market(MarketSpec(f"E-{k}", event="E"))
    ex.set_event_convertible("E")
    ex.open_account("A", 10_000)
    assert ex.mint_set("A", "E", 5)
    assert ex.accounts["A"].cash == 9_500 and all(ex.accounts["A"].pos[m] == 5 for m in ex.events["E"])
    assert ex.redeem_set("A", "E", 5) and ex.accounts["A"].cash == 10_000


@pytest.mark.parametrize("seed", range(6))
def test_random_flow_conserves_cash_and_keeps_books_consistent(seed):
    rng = random.Random(seed)
    ex = Exchange("V", QuadraticFee(0.07, 0.0175), allow_conversion=True)
    mids = ["S"] + [f"E-{k}" for k in range(3)]
    ex.add_market(MarketSpec("S"))
    for k in range(3):
        ex.add_market(MarketSpec(f"E-{k}", event="E"))
    ex.set_event_convertible("E")
    traders = [f"T{i}" for i in range(6)]
    for tr in traders:
        ex.open_account(tr, rng.randint(2_000, 50_000))
    total0 = ex.total_cash()
    live = []
    for step in range(3_000):
        tr = rng.choice(traders)
        u = rng.random()
        if u < 0.70:
            m = rng.choice(mids)
            side = rng.choice((BUY, SELL))
            px = max(1, min(99, int(rng.gauss(50, 15))))
            tif = rng.choice(("GTC", "GTC", "GTC", "IOC", "FOK"))
            r = ex.submit(tr, m, side, px, rng.randint(1, 40), tif=tif, post_only=rng.random() < 0.1, t=step)
            if r.resting:
                live.append((tr, r.oid))
        elif u < 0.90 and live:
            tr2, oid = live.pop(rng.randrange(len(live)))
            ex.cancel(tr2, oid, t=step)
        elif u < 0.95:
            ex.mint_set(tr, "E", rng.randint(1, 5))
        else:
            ex.redeem_set(tr, "E", rng.randint(1, 5))
        for a in ex.accounts.values():
            assert a.cash >= 0 and a.reserved >= 0
        if step % 250 == 0:
            for m in mids:
                ex.book(m).check()
    winner = rng.randrange(3)
    ex.settle("S", rng.random() < 0.5)
    for k in range(3):
        ex.settle(f"E-{k}", k == winner)
    assert all(a.reserved == 0 for a in ex.accounts.values())
    assert all(q == 0 for a in ex.accounts.values() for q in a.pos.values())
    assert ex.total_cash() + ex.fees_collected == total0
