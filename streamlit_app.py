import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from asset_allocation_optimizer import run_optimizer

st.set_page_config(page_title="Asset Allocation Optimizer", layout="wide")
st.title("Asset Allocation Optimizer")
st.caption("Equal-Weight · Min-Variance · Max Sharpe · Risk Parity (inv-vol)")

# Brief explanation of each method
st.markdown(
    "**Optimization Methods:**  \n"
    "- **Equal-Weight**: Splits your investment equally across all selected assets.  \n"
    "- **Min-Variance**: Finds the portfolio with the lowest possible overall risk (volatility).  \n"
    "- **Max Sharpe**: Seeks the portfolio with the highest risk-adjusted return (Sharpe ratio).  \n"
    "- **Risk Parity (inv-vol)**: Allocates so each asset contributes equally to portfolio risk, favoring less volatile assets."
)

# with st.expander("What do these methods mean?"):
#     st.markdown(
#         "- **Equal-Weight**: Splits money evenly across all assets.\n"
#         "- **Min-Variance**: Chooses weights that minimize overall risk.\n"
#         "- **Max Sharpe**: Maximizes risk-adjusted return (return per unit of risk).\n"
#         "- **Risk Parity (inv-vol)**: Balances risk so each asset contributes equally, giving less to volatile ones."
#     )

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
    st.caption("Note: cap must be ≥ 1 / (# tickers) to function.")
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
    W = rng.random((4000, A))  # Keep it at 4000 for illustrative purposes
    W = W / W.sum(axis=1, keepdims=True)
    port_mu = W @ mu
    port_vol = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))

    ann = 12.0
    fig = go.Figure()

    # Background cloud
    fig.add_trace(go.Scattergl(
        x=port_vol*np.sqrt(ann)*100,
        y=port_mu*ann*100,
        mode="markers",
        name="Random Portfolios",
        marker=dict(size=4, opacity=0.25, color="lightgray"),
        hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>"
    ))

    # Highlighted portfolios
    marker_map = {
        "Equal-Weight": "circle",
        "Min-Variance": "x",
        "Max Sharpe": "star",
        "Risk Parity (inv-vol)": "square"
    }
    color_map = {
        "Equal-Weight": "blue",
        "Min-Variance": "green",
        "Max Sharpe": "orange",
        "Risk Parity (inv-vol)": "red",
    }
    for label, w in methods.items():
        x = float(np.sqrt(w @ cov @ w) * np.sqrt(ann) * 100)
        y = float((w @ mu) * ann * 100)
        wt_pairs = [f"{t}: {w[i]*100:.1f}%" for i, t in enumerate(rets.columns)]
        wt_text = "<br>".join(wt_pairs[:6]) + ("<br>..." if len(wt_pairs) > 6 else "")

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            name=label,
            marker_symbol=marker_map.get(label, "circle"),
            marker=dict(size=12, line=dict(width=1), color=color_map.get(label, "black")),
            text=[label], textposition="top center",
            hovertemplate=(f"<b>{label}</b><br>"
                           f"Vol: {x:.2f}%<br>Ret: {y:.2f}%<br><br>"
                           f"<b>Weights</b><br>{wt_text}<extra></extra>")
        ))

    fig.update_layout(
        #title="Efficient Frontier (Interactive)",
        xaxis_title="Volatility (annualized, %)",
        yaxis_title="Expected Return (annualized, %)",
        legend_title="Portfolios",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig

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
            f"Cap {cap:.2f} is not feasible for {N} tickers. "
            f"Increase cap to at least {1.0/N:.2f} or add more tickers."
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
        fig = make_frontier_plot(res)
        st.plotly_chart(fig, use_container_width=True)
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
        # st.subheader("Download")
        # st.download_button(
        #     "Download summary CSV",
        #     data=to_csv_bytes(res["summary_table"]),
        #     file_name="optimizer_summary.csv",
        #     mime="text/csv"
        # )
        st.caption("Tip: Use Canadian (.TO) or US tickers. Data are monthly (resampled). "
                   "Max Sharpe uses risk-free rate; per-asset cap controls concentration.")