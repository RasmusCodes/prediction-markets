# Data dictionary

**The data files are not in the repository** (about 2.8 GB of SQLite stores). The collectors in `collect/` rebuild
the resolved-market datasets from the same public endpoints; the live capture of 13 November 2026 is a one-time
recording and cannot be re-collected. The simulator, the simulation experiments, the tests and the dashboard's
simulator need none of these files. Check each venue's terms before sharing data you collect.

All data were collected read-only from public endpoints (Kalshi `api.elections.kalshi.com/trade-api/v2`, Polymarket
`gamma-api.polymarket.com` and `clob.polymarket.com`, Coinbase Exchange public market data) on 13-14 November 2026.
Timestamps are UTC epoch seconds. Prices are probabilities in dollars (0-1) unless a column says cents.

## live/live.sqlite (collect/live_snapshots.py)

Capture window 2026-11-13 05:32-19:24 UTC (13.9 h, one gap of about 40 minutes).

| Table | Rows | Columns |
|---|---|---|
| `quotes` | 323,525 | ts, venue (kalshi/poly), market, bid, ask, bid_sz, ask_sz, last, volume, oi. Written only when the top of book changed. Kalshi empty sides are bid 0 / ask 1 with size 0 |
| `cycles` | 15,085 | ts, venue, n: every polling time, so unchanged quotes can be forward-filled exactly |
| `depth` | 356,176 | ts, venue, market, bids, asks (JSON top-10 [[price, size]]); Kalshi asks derived from NO bids as 1 - p |
| `trades` | 4,725,594 | venue, trade_id, market, ts, price (YES), size (fractional contracts exist), taker_side (yes/no). Complete Kalshi tape, multivariate combos excluded |
| `spot` | 60,094 | ts (receipt), symbol (BTC-USD/ETH-USD), price, bid, ask, server_ts. Polled once per second |
| `universe` | 15,186 | ts, venue, market, event, title, group_key, mutually_exclusive, close_time, extra (JSON: strike type/value, category, Polymarket condition id). Kalshi re-selected every 20 min (most active events + their siblings); Kalshi's most-traded tickers are added every 30 s (15-minute markets roll) |
| `meta` | | latest metadata per market |

`live/kalshi_event_counts.csv`: true number of markets per mutually exclusive Kalshi event (collect/event_counts.py),
used to tell complete outcome sets from truncated ones.

## hist/kalshi_hist.sqlite (collect/hist/k_scan.py, k_sample.py, k_candles.py)

| Table | Rows | Notes |
|---|---|---|
| `series` | 16,341 | ticker, title, category, frequency, fee_type, fee_multiplier |
| `event_index` | 1,008,752 | every settled event (stage 1 scan) |
| `events` | 1,795 | sampled events (mutually_exclusive, category) |
| `markets` | 20,000 | settled binary markets from 462 series, Jan 2024 - Nov 2026; partition = live/historical (cutoff 2026-09-13) |
| `candles` | 1,880,957 | hourly (period 60): end_ts, bid_close, ask_close, price_close, price_mean, volume, open_interest; 30 days before close; 11,622 markets (405 of 12,041 markets with volume >= 10 not fetched) |

Sampling: series shuffled within category and visited round-robin across categories; up to 15 events per series at
evenly spaced dates (nearest neighbour if an event has no qualifying market or more than 80 markets); every binary
yes/no market of a sampled event kept; stop at 20,000 markets.

## hist/polymarket_hist.sqlite (collect/hist/pm_collect.py)

| Table | Rows | Notes |
|---|---|---|
| `markets` | 9,703 | condition_id, question, event_slug, neg_risk, outcomes, final_prices, winner_index (0 = first outcome won), token0/1, start/end/closed ts, volume, sample_group (top / random), series_key |
| `prices` | 3,325,342 | token_id (token0), t, p: CLOB price history at hourly fidelity, 30 days before close |

Sample: top ~6,000 closed binary markets by volume plus a random ~4,000 with $1k-$50k volume; recurring series capped.

## hist/crypto15m.sqlite (collect/hist/crypto15m.py)

| Table | Rows | Notes |
|---|---|---|
| `k_markets` | 5,784 | every KXBTC15M / KXETH15M market 2026-10-14 to 11-13: open/close, floor_strike (target), result |
| `k_candles` | 90,881 | 1-minute bid/ask/price closes per market |
| `cb_candles` | 86,395 | Coinbase BTC-USD / ETH-USD 1-minute OHLCV (ts = minute start) |

## Derived files

`processed_calibration_panel.pkl` (market x horizon probabilities and outcomes, both venues) and
`processed_speed_live.pkl` (1-second Kalshi VWAP, spot and p* for every 15-minute window in the live capture) are
written by the research scripts.
