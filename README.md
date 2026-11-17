# Prediction-market laboratory

An automated prediction-market maker and quantitative trading system: a simulated exchange, a pricing engine, an
automated market maker, an LP arbitrage engine, risk and execution layers, a backtester, a live dashboard, and an
empirical research paper built on Kalshi and Polymarket market data.

**Real platforms are read-only by design.** The collectors use only public, documented market-data endpoints, under
token-bucket rate limits well below the published ones, with a generic user agent. Nothing in this repository places
orders on a real venue; every strategy trades only the simulated exchange or a historical replay.

## Quick start

Python 3.12 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
python -m pytest tests                      # 25 tests: matching, accounting conservation, world model, pricing, sim, replay, UI
python -m pmkt.ui.server --port 8797        # dashboard at http://127.0.0.1:8797
```

Read the paper: `paper/paper.pdf` or `paper/paper.html` (self-contained; download it to view); source
`paper/paper.md`, tables in `paper/tables/`, figures in `paper/figures/`.

## Data

The raw market data (about 2.8 GB of SQLite stores) are **not included**. Everything simulated runs without them:
the simulator, the experiments in `runs/`, the tests, the dashboard's simulator and research tabs, and the paper
build. The real-data scripts (`research/real_*.py`) need the stores that the collectors in `collect/` build from the
venues' public endpoints; `data/DATA.md` documents every table. The live capture of 13 November 2026 behind the
speed, microstructure and arbitrage studies was a one-time recording and cannot be re-collected. Check each venue's
terms before sharing data you collect.

## What is where

| Path | Contents |
|---|---|
| `pmkt/core/` | `orderbook.py` price-time-priority book (GTC/IOC/FOK/post-only, self-trade prevention); `exchange.py` venue with fully collateralised integer-cent accounts, netting, mint/redeem of full sets, settlement, message rate limits; `fees.py` Kalshi quadratic, linear and zero fee schedules |
| `pmkt/world/truth.py` | Ground truth: latent diffusion + Poisson jumps + scheduled news + terminal resolution; exact martingale probabilities for binaries, strike ladders with buckets, and correlated pairs with a conjunction contract |
| `pmkt/pricing/` | `implied` (mid, microprice, de-vigging), `bayes` (beta-binomial, log-odds fusion, Glosten-Milgrom, log-odds Kalman filter), `logistic` (IRLS, cluster-robust SEs), `calibration` (Brier/Murphy, ECE, Platt, isotonic, beta), `events` (scheduled-risk, post-event distribution, jump detector), `correlation` (copula, implied rho, Fréchet, tetrachoric, EWMA covariance, lead-lag), `liquidity` (Avellaneda-Stoikov in log-odds, toxicity, Kyle's lambda), `amm` (LMSR, LS-LMSR, curve-to-ladder) |
| `pmkt/sim/` | `engine.py` discrete-event simulator with per-agent latency and a recorder; `agents.py` noise, informed, momentum, naive MM, AMM and rule-based arbitrageur; `scenario.py` builds everything from one config + seed |
| `pmkt/strategies/` | `market_maker.py` inventory-aware, liquidity-sensitive MM with event-driven requotes; `arbitrage.py` LP Dutch-book scanner, strategy and read-only monitor; `mispricing.py` calibration / copula / consistency / lead-lag models with Kelly sizing |
| `pmkt/risk/`, `pmkt/execution/` | Exact scenario risk per family, pre-trade checks, kill switch; quote reconciliation and fee-aware routing |
| `pmkt/backtest/` | `metrics.py` (P&L, markouts, spread capture, pricing error, information half-life), `runner.py` (parallel, resumable experiments with manifests), `replay.py` (historical replay with queue-position fills) |
| `pmkt/ui/` | Dashboard server (standard library) and single-page client: market explorer, live order book, probability charts, trade ticket, strategy/P&L/inventory tracking, risk, arbitrage scanner, recorded real-market examples |
| `collect/` | Read-only collectors: `live_snapshots.py` (quotes, depth, Kalshi tape, Coinbase spot), `hist/` (Kalshi and Polymarket resolved markets, 15-minute crypto), `event_counts.py` |
| `research/` | Simulation experiments E1-E5, analysis and paired contrasts; real-data studies of calibration, speed, microstructure and arbitrage |
| `data/` | `DATA.md` data dictionary; the SQLite stores themselves are built locally by `collect/` and not committed |
| `runs/` | Experiment outputs: `results.jsonl`, `results.csv`, `manifest.json` (config, seeds, code hash, versions) |

## Reproducing the research

Simulation (no external data needed; `runs/` already holds the outputs the paper uses, and the runner resumes from them):

```
python -m research.sim_experiments e1 e2 e3 e4 --seeds 8 --T 3600      # ~30 min on 8 workers, resumable
python -m research.sim_experiments e5 --seeds 32 --T 3600 --tag seeds32 # market-maker ablations, ~90 min on 3 workers
python -m research.sim_analysis
python -m research.sim_contrasts
```

Real data (collect first; read-only and rate-limited). The `collect/hist/` scripts stop after `--max-minutes`
(default 9) and resume where they left off, so run each until it reports nothing left to fetch:

```
python collect/live_snapshots.py --hours 14                # live capture (the paper used 13 Nov 2026)
python collect/hist/k_scan.py                              # Kalshi: index of settled events
python collect/hist/k_sample.py                            # Kalshi: stratified market sample
python collect/hist/k_candles.py                           # Kalshi: hourly candles
python collect/hist/pm_collect.py cands                    # Polymarket: market sample
python collect/hist/pm_collect.py prices                   # Polymarket: price histories
python collect/hist/crypto15m.py coinbase|markets|candles  # 15-minute crypto markets and Coinbase candles
python collect/event_counts.py                             # outcome counts for the arbitrage study
python -m research.real_calibration
python -m research.real_speed
python -m research.real_microstructure
python -m research.real_arbitrage
python paper/build.py                                      # paper.html, and paper.pdf if Edge/Chrome is installed
```

Set `OPENBLAS_NUM_THREADS=1` when running many workers (the runner does this itself); without it, parallel workers
can fail with OpenBLAS memory errors.

## Transaction-cost realism

One-cent ticks; Kalshi's published quadratic fee per fill (rounded up per fill, so small orders pay proportionally
more); per-agent latency on every order and cancel; price-time queue priority and queue-position fill modelling in
replays; full collateral for resting orders; per-trader message rate limits; capital carry on locked collateral in
the metrics; settlement at the official outcome.

## Headline results (details and caveats in the paper)

- **Speed:** in Kalshi's 15-minute crypto markets, about 65% of a large spot-implied move appears within the same
  second and 96% within the minute.
- **Calibration:** Kalshi slightly compressed (slope 1.07), Polymarket slightly too extreme (0.93). Kalshi longshots
  under 5 cents lose 72% at the ask after fees, but recalibration rules do not profit out of sample.
- **Microstructure:** order flow predicts 30-second moves (R² 1.5%), but the moves are smaller than the cost of
  trading on them; replayed MM and spot-informed strategies lose 5-6 c/contract.
- **Arbitrage:** executable violations in under 1% of snapshots on complete Kalshi sets, gone in ~14 s. In simulation
  an LP arbitrageur cuts incidence from 99% to 70% and pricing error by 30%.
- **Liquidity:** in simulation, removing market makers raises pricing error by 55%; extra depth mostly pays informed
  traders. The thinnest Kalshi markets are the most compressed toward 50%.
- **Market making (32 paired seeds):** the automated MM earns $167 per simulated hour (27 of 32 seeds profitable).
  The trade-informed fair value drives the profit; inventory skew and spread width change risk, not returns; the
  cross-market portfolio skew hurts and is off by default.

## Extending to live trading

Not implemented on purpose. A live adapter would need the user's own exchange credentials, the venue's trading API
terms and per-endpoint rate limits, reduce-only and cancel-on-disconnect safeguards, and a separate review. The
strategy classes only talk to the simulator API (`submit` / `cancel` / books), which is the seam where an adapter would
sit.

## License

MIT; see `LICENSE`. The licence covers the code and paper in this repository, not market data you collect with it.
