"""Streamlit dashboard: synthetic backtest diagnostics (no auth)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pm_divergence.backtest.engine import load_system_config
from pm_divergence.backtest.synthetic import build_dashboard_series

_REGIME_ORDER = {"noise": 0, "liquidity": 1, "neutral": 2, "information": 3}


@st.cache_data(show_spinner=False)
def _series(seed: int, rng_seed: int, _refresh: int) -> dict:
    cfg = load_system_config()
    return build_dashboard_series(cfg, seed=seed, rng_seed=rng_seed)


def main() -> None:
    st.set_page_config(page_title="PM Divergence — Dashboard", layout="wide")
    st.title("PM Divergence — Research dashboard")

    if "refresh" not in st.session_state:
        st.session_state.refresh = 0

    with st.sidebar:
        st.header("Parameters")
        seed = st.number_input("Dataset seed", min_value=0, value=42, step=1)
        rng_seed = st.number_input("Slippage RNG seed", min_value=0, value=7, step=1)
        if st.button("Refresh data"):
            st.session_state.refresh += 1

    data = _series(int(seed), int(rng_seed), int(st.session_state.refresh))
    timeline = data["timeline"]
    pnl_steps = data["pnl_steps"]
    hit_rate = float(data["hit_rate"])
    n_trades = int(data["n_trades"])

    df_tl = pd.DataFrame(timeline)
    if not df_tl.empty:
        df_tl = df_tl.set_index("time").sort_index()
        df_tl["regime_code"] = df_tl["regime"].map(lambda r: _REGIME_ORDER.get(str(r), 2))

    df_pnl = pd.DataFrame(pnl_steps, columns=["time", "cumulative_pnl"]).set_index("time").sort_index()

    latest_regime = str(df_tl["regime"].iloc[-1]) if len(df_tl) else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hit rate", f"{hit_rate:.1%}")
    c2.metric("Simulated trades", str(n_trades))
    c3.metric("Latest regime", latest_regime)
    c4.metric("Final cumulative PnL", f"{df_pnl['cumulative_pnl'].iloc[-1]:.2%}" if len(df_pnl) else "—")

    st.subheader("PnL curve (compounded trade returns)")
    if len(df_pnl):
        st.line_chart(df_pnl)
    else:
        st.info("No PnL path (no trades).")

    st.subheader("Signal timeline")
    if len(df_tl):
        st.line_chart(df_tl[["signal"]])
    else:
        st.info("No timeline.")

    st.subheader("Regime indicator (ordinal)")
    st.caption("0=noise, 1=liquidity, 2=neutral, 3=information")
    if len(df_tl):
        st.line_chart(df_tl[["regime_code"]])
    else:
        st.info("No regime series.")

    st.subheader("Recent signals")
    if len(df_tl):
        recent = df_tl.reset_index().sort_values("time", ascending=False).head(25)
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No rows.")


if __name__ == "__main__":
    main()
