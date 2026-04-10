# Research charter — pm-divergence-system

This repository is a **research validation** stack for causal time-series and backtesting. It is **not** production trading infrastructure.

## Do not

- Add ML model training/serving stacks or heavyweight model dependencies.
- Add reinforcement learning or policy-optimization loops.
- Introduce abstractions beyond what clarity and correctness require.
- Add production infrastructure (Kubernetes, queues, secrets managers, multi-tenant auth, etc.).

## Do

- **Prioritize correctness of time-series logic** — explicit event times, ordered prefixes, and conservative forward fills where horizons matter.
- **Ensure no leakage** — scores and snapshots must use only information available at or before the stated `asof` / decision clock; forward outcomes enter only where explicitly labeled (e.g. simulated fills at `t + latency`).
- **Keep everything reproducible** — document and thread seeds for synthetic data and simulation noise; pin behavior to `configs/system.yaml` where applicable.
- **Make the backtest deterministic** — same config + seeds ⇒ same trades, fills, and metrics (floating-point aside).

## How this maps in code (MVP)

- **No lookahead on the event clock:** `snapshot_at` / `event_replay` include only rows with `timestamp <= t`; trade simulator uses first quote at or after the exit/entry clock.
- **Deterministic slippage:** `trade_simulator.simulate_trades(..., rng_seed=...)`.
- **Synthetic path:** `backtest/synthetic.build_synthetic_report` / `build_dashboard_series` — deterministic given `SystemConfig`, dataset `seed`, and slippage `rng_seed`.

Reviewers should reject changes that violate the above without updating this charter.
