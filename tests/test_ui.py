import time

from pmkt.ui.server import SimHost


def test_dashboard_backend_runs_and_accepts_orders():
    host = SimHost(seed=3, speed=600.0, T=300.0)
    time.sleep(1.5)
    st = host.state()
    assert st["t"] > 0 and len(st["markets"]) > 10
    assert {"mm", "arb", "you"} <= set(st["strategies"])
    m = st["markets"][0]
    bk = host.book(m["venue"], m["market"])
    assert "bids" in bk and "asks" in bk
    se = host.series(m["venue"], m["market"])
    assert len(se["t"]) == len(se["bid"])
    r = host.order({"venue": m["venue"], "market": m["market"], "side": "buy", "price": 99, "qty": 5, "tif": "IOC"})
    assert r["ok"]
    host.control("pause")
    assert host.state()["running"] is False
