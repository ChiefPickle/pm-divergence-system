#!/usr/bin/env python3
"""Synthetic backtest CLI: replay → trades → metrics → JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo layout: pm-divergence-system/backtests/run_backtest.py → src/
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pm_divergence.backtest.engine import load_system_config
from pm_divergence.backtest.synthetic import build_synthetic_report, sanitize_for_json


def main() -> None:
    p = argparse.ArgumentParser(description="Run synthetic event replay backtest.")
    p.add_argument("-o", "--output", type=Path, default=None, help="Write JSON here (default: stdout)")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to system.yaml (default: configs/system.yaml via backtest engine)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=None, help="Override configs/system.yaml")
    p.add_argument("--latency-seconds", type=int, default=None, help="Override configs/system.yaml")
    p.add_argument("--holding-hours", type=float, default=None, help="Override configs/system.yaml")
    p.add_argument("--slippage-sigma", type=float, default=None, help="Override configs/system.yaml")
    p.add_argument("--rng-seed", type=int, default=7)
    args = p.parse_args()

    cfg = load_system_config(args.config)
    if str(cfg.slippage_model).lower() != "gaussian":
        raise SystemExit(f"Unsupported slippage_model {cfg.slippage_model!r} (only 'gaussian' is implemented)")

    from dataclasses import replace

    cfg = replace(
        cfg,
        threshold=float(cfg.threshold if args.threshold is None else args.threshold),
        latency_seconds=int(cfg.latency_seconds if args.latency_seconds is None else args.latency_seconds),
        holding_period_hours=float(
            cfg.holding_period_hours if args.holding_hours is None else args.holding_hours
        ),
        slippage_sigma=float(cfg.slippage_sigma if args.slippage_sigma is None else args.slippage_sigma),
    )

    report = build_synthetic_report(cfg, seed=args.seed, rng_seed=args.rng_seed)

    text = json.dumps(sanitize_for_json(report), indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
