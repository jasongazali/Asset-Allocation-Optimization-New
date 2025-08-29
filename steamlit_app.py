import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from asset_allocation_optimizer import run_optimizer

st.set_page_config(page_title="Asset Allocation Optimizer", layout="wide")
st.title("Asset Allocation Optimizer")
st.caption("Equal-Weight · Min-Variance · Max Sharpe · Risk Parity (inv-vol)")

with st.sidebar:
    st.header("Inputs")
    st.write("Enter 2+ tickers (comma-separated). Examples:")
    st.code("XIU.TO,XSP.TO,XBB.TO,XEF.TO,XEC.TO,XGD.TO,\nSPY,AGG,GLD,EFA,EMB")
    tickers_text = st.text_input("Tickers", value="XIU.TO,XSP.TO,XBB.TO,XEF.TO,XEC.TO,XGD.TO")
    start = st.text_input("Start date (YYYY-MM-DD)", value="2015-01-01")
    end = st.text_input("End date (optional, YYYY-MM-DD)", value="")
    rf_annual = st.number_input("Risk-free rate (annual)", value=0.02, min_value=0.0, step=0.005, format="%.3f")
    cap = st.slider("Per-asset weight cap", min_value=0.10, max_value=1.00, value=0.40, step=0.05)
    n_portfolios = st.number_input("# Random portfolios", value=20000, min_value=2000, step=1000)
    st.caption("Note: cap must be ≥ 1 / (# tickers) to be function.")
    seed = st.number_input("Random seed", value=42, step=1)
    run_btn = st.button("Run Optimizer", type="primary")

def make_frontier_plot(res):
    rets = res["monthly_returns"]
    mu = res["mu"]
    cov = res["cov"]
    methods = res["weights"]

    # Build a small random cloud just for the plot (fast)
    rng = np.random.default_rng(0)
    A = len(rets.columns)
    W = rng.random((4000, A)) #Keep it at 4000 for illustrative purposes
    W = W / W.sum(axis=1, keepdims=True)
    port_mu = W @ mu
    port_vol = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))

    fig, ax = plt.subplots(figsize=(7,5))
    ann = 12.0
    ax.scatter(port_vol*np.sqrt(ann)*100, port_mu*ann*100, s=6, alpha=0.25)

    markers = {"Equal-Weight":"o","Min-Variance":"X","Max Sharpe":"*","Risk Parity (inv-vol)":"s"}
    for label, w in methods.items():
        x = (np.sqrt(w @ cov @ w) * np.sqrt(ann)) * 100
        y = (w @ mu) * ann * 100
        ax.scatter(x, y, s=120, marker=markers.get(label,"o"), label=label)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6,6))

    ax.set_xlabel("Volatility (annualized, %)")
    ax.set_ylabel("Expected Return (annualized, %)")
    ax.set_title("Efficient Frontier")
    ax.legend()
    st.pyplot(fig)

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

if run_btn:
    tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
    N = len(tickers)
    # Guard: ensure per-asset cap is feasible given number of tickers
    if cap * N < 1.0:
        st.warning(
            f"Cap {cap:.2f} is infeasible for {N} tickers. "
            f"Increase cap to at least {1.0/N:.2f} or add more tickers. Auto-relaxing cap for this run."
        )
        cap = max(cap, 1.0 / N + 1e-6)
    if len(tickers) < 2:
        st.error("Please enter at least 2 tickers.")
    else:
        res = run_optimizer(
            tickers=tickers,
            start=start,
            end=end if end.strip() else None,
            rf_annual=float(rf_annual),
            cap=float(cap),
            n_portfolios=int(n_portfolios),
            seed=int(seed),
        )

        # Summary table
        st.subheader("Allocations & Annualized Stats")
        st.dataframe(res["summary_table"], use_container_width=True)

        # Frontier plot
        st.subheader("Efficient Frontier")
        make_frontier_plot(res)
        st.caption(
            f"Graph uses 4,000 random portfolios for visualization speed; "
            f"highlighted portfolios are computed using your chosen n_portfolios = {int(n_portfolios)}."
        )

        # Data previews (expanders)
        with st.expander("Monthly Prices (Most Recent)"):
            st.dataframe(res["monthly_prices"].tail(), use_container_width=True)
        with st.expander("Monthly Returns % (Most Recent)"):
            st.dataframe((res["monthly_returns"].tail()*100).round(2), use_container_width=True)

        # Downloads
        st.subheader("Download")
        st.download_button(
            "Download summary CSV",
            data=to_csv_bytes(res["summary_table"]),
            file_name="optimizer_summary.csv",
            mime="text/csv"
        )
        st.caption("Tip: Use Canadian (.TO) or US tickers. Data are monthly (resampled). "
                   "Max Sharpe uses risk-free rate; per-asset cap controls concentration.")