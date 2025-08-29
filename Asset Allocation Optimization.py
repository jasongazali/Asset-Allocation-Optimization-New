import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fetch_monthly_prices(tickers, start="2015-01-01", end=None):
    """
    Download Adjusted Close prices and resample to month-end.
    Falls back to synthetic data if yfinance is unavailable/offline.
    """
    try:
        import yfinance as yf
        px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        m_px = px.resample("M").last()
        return m_px
    except Exception as e:
        print("[warn] yfinance fetch failed, using synthetic mock data. Error:", e)
        rng = np.random.default_rng(7)
        n = 120  # 10 years monthly
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="M")
        A = len(tickers)
        # simple synthetic vols/corrs
        vol = np.linspace(0.015, 0.055, A)
        corr = 0.2*np.ones((A,A)) + 0.8*np.eye(A)
        cov = np.outer(vol, vol) * corr
        mu = np.linspace(0.003, 0.008, A)  # ~0.3%..0.8% monthly
        chol = np.linalg.cholesky(cov)
        z = rng.standard_normal(size=(n, A))
        rets = mu + z @ chol.T
        gross = 1.0 + rets
        start_prices = np.linspace(20.0, 40.0, A)
        prices = np.cumprod(np.vstack([np.ones(A), gross[:-1]]), axis=0) * start_prices
        m_px = pd.DataFrame(prices, index=dates, columns=tickers)
        return m_px
    
def annualize_stats(mu_monthly: float, vol_monthly: float, rf_annual: float):
    """
    mu_monthly = Mean Monthly Return
    vol_monthly = Monthly volatility
    rf_annual = Annual Risk Free rate
    """
    ann = 12.0 #Annualization Factor
    mu_a = mu_monthly * ann #Annualize Expected Return and we ignore compounding
    vol_a = vol_monthly * (ann ** 0.5) #Annualize volatility
    sharpe_a = (mu_a - rf_annual) / max(vol_a, 1e-12) #Annualized Sharpe Ratio (excess return/ volatility)
    return mu_a, vol_a, sharpe_a

def optimizer_random_frontier(rets: pd.DataFrame, cap=0.40, n_portfolios=20000, seed=42):
    """
    Random long-only portfolios (sum=1), optional per-asset cap. Returns stats for the cloud using Monte Carlo Simulation

    rets = monthly returns for each assets by tickets and months
    cap = max weight per asset
    n_portfolio = no of random portfolio that will be generated

    Returns:

    tickers
    mu = monthly mean return
    cov = monthly covariance matrix
    W = portfolio weights
    port_mu = monthly expected mean return for each portfolio
    port_vol = monthly volatilities for each portfolio
    """
    rng = np.random.default_rng(seed)
    tickers = list(rets.columns)
    A = len(tickers)
    mu = rets.mean().values               # monthly mean returns
    cov = rets.cov().values               # monthly covariance

    W = rng.random((n_portfolios, A))
    W = W / W.sum(axis=1, keepdims=True)
    if cap is not None:
        W = W[(W <= cap).all(axis=1)]
    if len(W) == 0:
        raise ValueError("No feasible portfolios under the chosen cap. Increase cap or reduce assets.")

    port_mu = W @ mu
    port_vol = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))
    return tickers, mu, cov, W, port_mu, port_vol

def pick_minvar_maxsharpe(W, port_mu, port_vol, rf_annual=0.02):
    """
    This scans through our Monte Carlo simulation and filters for lowest volatility and highest Sharpe
    """
    rf_m = rf_annual / 12.0
    sharpe = (port_mu - rf_m) / np.clip(port_vol, 1e-12, None)
    i_minvar = int(np.argmin(port_vol)) #Index of min var
    i_maxsharpe = int(np.argmax(sharpe)) #Index of max sharpe
    return i_minvar, i_maxsharpe, sharpe

def risk_parity_invvol(cov):
    """
    Simple Risk Parity via inverse-volatility weights:
    w_i ∝ 1 / sigma_i, where sigma_i = sqrt(cov_ii).
    This ignores correlations (true ERC needs an iterative solver),
    but it's a good beginner-friendly proxy.

    We allocate more weights to assets with lower volatility
    """
    sig = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    inv = 1.0 / sig
    w = inv / inv.sum()
    return w

def summarize_allocations(tickers, mu, cov, methods, rf_annual=0.02):
    rows = []
    for name, w in methods.items():
        mu_m = float(w @ mu) #Weighted average of expected returns
        vol_m = float(np.sqrt(w @ cov @ w)) #Portfolio expected volatility
        mu_a, vol_a, sharpe_a = annualize_stats(mu_m, vol_m, rf_annual)
        row = {"Method": name,
               "Ann. Return %": round(mu_a*100, 2),
               "Ann. Vol %": round(vol_a*100, 2),
               "Sharpe": round(sharpe_a, 2)}
        for i, t in enumerate(tickers):
            row[f"Weights_{t} %"] = round(w[i]*100, 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def plot_frontier(port_vol, port_mu, methods_dict, mu, cov, tickers):
    """
    Port_vol and Port_mu = Weighted average of returns and volatility for each portfolio
    methods_dict = weight vector for each method
    mu, cov, tickers = per period expected returns and covariance
    """
    ann = 12.0
    plt.figure(figsize=(7,5))
    if port_vol is not None and port_mu is not None:
        plt.scatter(port_vol*np.sqrt(ann)*100, port_mu*ann*100, s=5, alpha=0.3)
    for label, w in methods_dict.items():
        x = (np.sqrt(w @ cov @ w) * np.sqrt(ann)) * 100
        y = (w @ mu) * ann * 100
        marker = {"Equal-Weight":"o", "Min-Variance":"X", "Max Sharpe":"*", "Risk Parity (inv-vol)":"s"}.get(label, "o")
        plt.scatter(x, y, s=100, marker=marker, label=label)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(6,6))
    plt.xlabel("Volatility (annualized, %)")
    plt.ylabel("Expected Return (annualized, %)")
    plt.title("Efficient Frontier (Random Search)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_optimizer(tickers,
                  start="2015-01-01",
                  end=None,
                  rf_annual=0.02,
                  cap=0.40,
                  n_portfolios=20000,
                  seed=42):
    # 1) Data
    m_px = fetch_monthly_prices(tickers, start=start, end=end)
    rets = m_px.pct_change().dropna()

    # 2) Random frontier
    tickers, mu, cov, W, port_mu, port_vol = optimizer_random_frontier(
        rets, cap=cap, n_portfolios=n_portfolios, seed=seed
    )

    # 3) Pick portfolios
    i_minvar, i_maxsharpe, _ = pick_minvar_maxsharpe(W, port_mu, port_vol, rf_annual=rf_annual)
    w_equal = np.ones(len(tickers)) / len(tickers) #Equal weight
    w_minvar = W[i_minvar] #Minimum variance
    w_maxsharpe = W[i_maxsharpe] #Max Sharpe
    w_rp = risk_parity_invvol(cov) #Risk Parity

    methods = {"Equal-Weight": w_equal,
               "Min-Variance": w_minvar,
               "Max Sharpe": w_maxsharpe,
               "Risk Parity (inv-vol)": w_rp}

    # 4) Summary
    summary = summarize_allocations(tickers, mu, cov, methods, rf_annual=rf_annual)

    return {
        "monthly_prices": m_px,
        "monthly_returns": rets,
        "weights": methods,
        "summary_table": summary,
        "frontier_cloud": (port_vol, port_mu),
        "mu": mu,
        "cov": cov,
        "tickers": tickers
    }