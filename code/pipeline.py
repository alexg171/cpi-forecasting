"""
Computation toolbox for the project: data preparation, unit-root tests, ARIMA estimation and
diagnostics, structural-break tests and forecast evaluation. Each function returns tables
and figures; the modeling choices are made in the fit_<series>.ipynb notebooks that call it.

Inputs : data/raw/fred_prices_raw.csv   raw FRED index levels saved by data_memo.ipynb
Outputs: none directly (tables and figures are returned to the calling notebook)
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are returned to the notebook, not opened in windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parent.parent

# Sample, train / test split and known break dates (same as fit_headline_cpi.ipynb)
SAMPLE_START, SAMPLE_END = "1985-01-01", "2026-08-01"
TRAIN_END, TEST_START = "2023-12-01", "2024-01-01"
BREAK_DATES = {"COVID-19": "2020-03-01", "2025 tariffs": "2025-04-01", "2026 Iran conflict": "2026-02-01"}
ALPHA = 0.05
BURN = 13               # residuals dropped before diagnostics (covers lag-12 + lag-1 terms)
HAC = {"cov_type": "HAC", "cov_kwds": {"maxlags": 12}}
SPEC_LABELS = {"ct": "Constant + trend", "c": "Constant only"}

# Andrews (1993) sup-Wald critical values, 15% trimming, q restrictions
ANDREWS_CV = {1: {"10%": 7.12, "5%": 8.68, "1%": 12.16},
              2: {"10%": 10.00, "5%": 11.72, "1%": 15.56}}

# Bai-Perron (2003) sequential sup-F(l+1 | l) 5% critical values, 15% trimming, for
# l = 0..4, by q = number of coefficients allowed to break
BP_SEQ_CV = {1: [8.58, 10.13, 11.14, 11.83, 12.25],
             2: [11.47, 12.95, 14.03, 14.85, 15.29]}


# ---------------------------------------------------------------- Step 0-1: data

def load_prices():
    """Raw FRED index levels, monthly, full history, exactly as saved by data_memo.ipynb."""
    return pd.read_csv(ROOT / "data" / "raw" / "fred_prices_raw.csv",
                       parse_dates=["date"], index_col="date")


def step1_prepare(prices_raw, sid):
    """Sample window + October 2025 fill for one series.

    Returns the price level for SAMPLE_START-1 month to SAMPLE_END (the extra month lets
    January 1985 keep an inflation value) and the list of filled dates. Missing interior
    months are filled log-linearly: ln P_Oct = (ln P_Sep + ln P_Nov) / 2.
    """
    start = pd.Timestamp(SAMPLE_START) - pd.DateOffset(months=1)
    p = prices_raw[sid].loc[start:SAMPLE_END]
    missing = p.isna()
    p = np.exp(np.log(p).interpolate(method="linear", limit_area="inside"))
    filled = [f"{d:%Y-%m}" for d in p.index[missing & p.notna()]]
    return p.asfreq("MS"), filled


def step3_inflation(level):
    """Annualized monthly inflation, 1200 * Δln P, on the sample window."""
    return (1200 * np.log(level).diff()).loc[SAMPLE_START:].asfreq("MS")


def split(y):
    """Training set and hold-out."""
    return y.loc[:TRAIN_END], y.loc[TEST_START:]


# ---------------------------------------------------------------- Step 2/4: unit roots

def unit_root_tests(y, regression):
    """ADF and KPSS on `y` with the same deterministic spec ("c" or "ct").

    ADF (H0: unit root): lags chosen by AIC from p_max = 12(T/100)^(1/4) (Schwert rule).
    KPSS (H0: stationary): automatic bandwidth; p-values bounded to [0.01, 0.10].
    """
    p_max = int(12 * (len(y) / 100) ** 0.25)
    with warnings.catch_warnings():
        # KPSS p-values outside its lookup table, and newer statsmodels' notices about
        # future return types (results are unchanged)
        warnings.simplefilter("ignore", InterpolationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        adf_stat, adf_p, adf_lags, *_ = adfuller(y, regression=regression, maxlag=p_max, autolag="AIC")
        kpss_stat, kpss_p, kpss_lags, _ = kpss(y, regression=regression, nlags="auto")
    return {"ADF stat": adf_stat, "ADF p": adf_p, "ADF lags": adf_lags,
            "KPSS stat": kpss_stat, "KPSS p": kpss_p, "KPSS bandwidth": kpss_lags}


def unit_root_table(y, specs):
    """ADF/KPSS for each spec in `specs` (main spec first)."""
    t = pd.DataFrame({SPEC_LABELS[s]: unit_root_tests(y, s) for s in specs}).T
    t[["ADF lags", "KPSS bandwidth"]] = t[["ADF lags", "KPSS bandwidth"]].astype(int)
    return t.rename_axis("Deterministic spec")


# ---------------------------------------------------------------- figures

def fig_raw_level(prices_raw, sid, label):
    """Raw index level, full history (Step 2)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prices_raw.index, prices_raw[sid], lw=1.2)
    s = prices_raw[sid].dropna()
    ax.set_xlim(s.index.min(), s.index.max())
    ax.set_title(f"{label} ({sid}), seasonally adjusted index level")
    ax.set_ylabel("Index")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_acf_pacf(y, title, zero=True):
    """Sample ACF / PACF to lag 36, seasonal lags 12 / 24 / 36 marked."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(y, lags=36, zero=zero, ax=axes[0], title="ACF")
    plot_pacf(y, lags=36, zero=zero, ax=axes[1], method="ywm", title="PACF")
    for ax in axes:
        for lag in (12, 24, 36):
            ax.axvline(lag, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Lag (months)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def fig_inflation(pi_train, label):
    """Inflation time plot, training set, with its mean (Step 3)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pi_train.index, pi_train, lw=0.9)
    ax.axhline(pi_train.mean(), color="gray", ls="--", lw=0.8, label=f"Mean = {pi_train.mean():.2f}%")
    ax.set_title(f"{label} inflation, annualized monthly log difference (training set)")
    ax.set_ylabel("% per year")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Step 7-9: models

def n_arma(order, seasonal):
    """Number of ARMA terms (p + q + P + Q), for the Ljung-Box df adjustment.
    p may be a list of specific lags (subset AR), e.g. [2, 5]."""
    p = len(order[0]) if isinstance(order[0], (list, tuple)) else order[0]
    return p + order[2] + seasonal[0] + seasonal[2]


def trend_for(order):
    """Constant (mean) when inflation is modelled in levels (d = 0); none when it is
    differenced (d = 1), since differenced inflation has no drift."""
    return "c" if order[1] == 0 else "n"


def ols_ar_start(y, order, seasonal):
    """Starting values from a least-squares AR regression, for pure AR models (no MA or
    seasonal terms). Subset AR models can wander into non-stationary regions from the
    statsmodels default start, which crashes or returns a degenerate fit. Returns None
    when the model is not pure AR."""
    if order[2] or seasonal[0] or seasonal[2]:
        return None
    lags = list(order[0]) if isinstance(order[0], (list, tuple)) else list(range(1, order[0] + 1))
    if not lags:
        return None
    x = np.asarray(y, dtype=float)
    for _ in range(order[1]):           # difference d times, as ARIMA does internally
        x = np.diff(x)
    const = trend_for(order) == "c"
    L, n = max(lags), len(x)
    cols = [np.ones(n - L)] if const else []
    cols += [x[L - l:n - l] for l in lags]
    X, t = np.column_stack(cols), x[L:]
    b = np.linalg.lstsq(X, t, rcond=None)[0]
    sigma2 = np.var(t - X @ b)
    return np.r_[b[:1], b[1:], sigma2] if const else np.r_[b, sigma2]


def fit(y, order, seasonal, exog=None):
    """MLE fit of one ARIMA specification.

    The default start values can fail for subset AR models (LinAlgError, or a fit with
    log-likelihood 0). In that case the fit is retried from least-squares AR starting values.
    If the default optimizer (L-BFGS) does not converge, the fit is retried with more
    iterations and with Powell's method; the converged fit with the highest likelihood is
    kept (or, if none converge, the highest likelihood overall). `converged` is attached
    to the result so a fit that never converged can be spotted.
    """
    model = ARIMA(y, exog=exog, order=order, seasonal_order=seasonal, trend=trend_for(order))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            first = model.fit()
        except np.linalg.LinAlgError:
            first = None
        if first is None or first.llf == 0:
            start = ols_ar_start(y, order, seasonal)
            if start is None:
                raise RuntimeError(f"fit failed for order={order}, seasonal={seasonal}")
            first = model.fit(start_params=start)
        fits = [first]
        if not fits[0].mle_retvals.get("converged", True):
            for kw in ({"maxiter": 2000}, {"method": "powell", "maxiter": 5000}):
                fits.append(model.fit(method_kwargs=kw))
    ok = [r for r in fits if r.mle_retvals.get("converged", True)]
    res = max(ok or fits, key=lambda r: r.llf)
    res.converged = bool(ok)
    return res


def step7_estimate(pi_train, candidates):
    """MLE fit of each candidate on the training set."""
    return {name: fit(pi_train, o, s) for name, (o, s) in candidates.items()}


def diagnostics(res, k):
    """Residual-diagnostic p-values (small = problem); k = number of ARMA terms."""
    e = res.resid.iloc[BURN:]
    lb = acorr_ljungbox(e, lags=[6, 12, 24], model_df=k)
    return {"Ljung-Box(6) p": lb.loc[6, "lb_pvalue"], "Ljung-Box(12) p": lb.loc[12, "lb_pvalue"],
            "Ljung-Box(24) p": lb.loc[24, "lb_pvalue"], "Jarque-Bera p": jarque_bera(e)[1]}


def step7_table(fits, candidates):
    """Comparison table: coefficients 'estimate (p)', fit statistics, residual diagnostics."""
    coef = pd.DataFrame({n: {k: f"{b:.3f} ({r.pvalues[k]:.3f})" for k, b in r.params.items()}
                         for n, r in fits.items()})
    first = [k for k in ("const", "sigma2") if k in coef.index]
    coef = coef.loc[first + [k for k in coef.index if k not in first]]
    fit = pd.DataFrame({n: {"Log-lik": r.llf, "AIC": r.aic, "BIC": r.bic, "HQIC": r.hqic}
                        for n, r in fits.items()}).map("{:.1f}".format)
    diag = pd.DataFrame({n: diagnostics(r, n_arma(*candidates[n])) for n, r in fits.items()}
                        ).map("{:.3f}".format)
    return pd.concat({"Coefficients": coef, "Fit": fit, "Residual diagnostics (p)": diag},
                     names=["Block", "Statistic"]).fillna("")


def step8_diagnostics(res, k):
    """Full residual diagnostics for the selected model, plus AR / MA root moduli."""
    e = res.resid.iloc[BURN:]
    lb = acorr_ljungbox(e, lags=[6, 12, 24], model_df=k)
    jb_stat, jb_p, skew, kurt = jarque_bera(e)
    rows = [*[(f"Ljung-Box({lag})", "Serial correlation", lb.loc[lag, "lb_stat"], lb.loc[lag, "lb_pvalue"])
              for lag in (6, 12, 24)],
            ("Jarque-Bera", f"Normality (skew {skew:.2f}, kurtosis {kurt:.1f})", jb_stat, jb_p)]
    t = pd.DataFrame(rows, columns=["Test", "Checks", "Statistic", "p-value"]).set_index("Test")
    roots = pd.DataFrame({"Modulus": [*abs(res.arroots), *abs(res.maroots)]},
                         index=[f"AR root {i + 1}" for i in range(len(res.arroots))]
                         + [f"MA root {i + 1}" for i in range(len(res.maroots))])
    roots["Outside unit circle"] = roots["Modulus"] > 1
    return t, roots


def step9_selection(fits, candidates):
    """Information criteria, distance from the best (Δ), short / long Ljung-Box p-values."""
    ic = pd.DataFrame({n: {"Params": len(r.params), "AIC": r.aic, "BIC": r.bic, "HQIC": r.hqic}
                       for n, r in fits.items()}).T
    for c in ("AIC", "BIC", "HQIC"):
        ic[f"Δ{c}"] = ic[c] - ic[c].min()
    for lag in (6, 24):
        ic[f"Ljung-Box({lag}) p"] = [
            acorr_ljungbox(r.resid.iloc[BURN:], lags=[lag], model_df=n_arma(*candidates[n]))["lb_pvalue"].iloc[0]
            for n, r in fits.items()]
    return ic


# ---------------------------------------------------------------- Step 10: breaks

def _ar1_data(pi):
    return pd.DataFrame({"pi": pi, "pi_lag": pi.shift(1)}).dropna()


def _break_design(reg, date):
    D = (reg.index >= date).astype(float)
    return D, pd.DataFrame({"const": 1.0, "pi_lag": reg["pi_lag"], "D": D, "D_x_lag": D * reg["pi_lag"]},
                           index=reg.index)


def step10_known(pi_full):
    """Known-date tests (full sample, HAC): mean shift, Chow (AR(1) dummy-interaction form),
    variance ratio F, and the robust Brown-Forsythe variance test."""
    reg = _ar1_data(pi_full)
    out = {}
    for name, date in BREAK_DATES.items():
        D, X = _break_design(reg, date)
        mean_fit = sm.OLS(reg["pi"], X[["const", "D"]]).fit(**HAC)
        chow_fit = sm.OLS(reg["pi"], X).fit(**HAC)
        chow = chow_fit.wald_test("D = 0, D_x_lag = 0", scalar=True)
        e = sm.OLS(reg["pi"], X[["const", "pi_lag"]]).fit().resid
        pre, post = e[D == 0], e[D == 1]
        f = post.var() / pre.var()
        f_p = 2 * min(stats.f.cdf(f, len(post) - 1, len(pre) - 1), stats.f.sf(f, len(post) - 1, len(pre) - 1))
        out[name] = {
            "Months after": int(D.sum()),
            "Mean before": reg["pi"][D == 0].mean(), "Mean after": reg["pi"][D == 1].mean(),
            "Mean shift p": mean_fit.pvalues["D"],
            "AR(1) before": chow_fit.params["pi_lag"],
            "AR(1) after": chow_fit.params["pi_lag"] + chow_fit.params["D_x_lag"],
            "Persistence change p": chow_fit.pvalues["D_x_lag"],
            "Chow p": chow.pvalue,
            "Variance ratio": f, "Variance ratio p": f_p,
            "Brown-Forsythe p": stats.levene(pre, post, center="median").pvalue,
        }
    return pd.DataFrame(out).T.astype({"Months after": int})


def step10_qlr(pi_full, label, trim=0.15):
    """QLR / sup-Wald (HAC) over candidate dates with 15% trimming, mean shift and Chow."""
    reg = _ar1_data(pi_full)
    n = len(reg)
    dates = reg.index[int(trim * n):int((1 - trim) * n)]
    rows = []
    for d in dates:
        _, X = _break_design(reg, d)
        mean_w = sm.OLS(reg["pi"], X[["const", "D"]]).fit(**HAC).tvalues["D"] ** 2
        chow_w = sm.OLS(reg["pi"], X).fit(**HAC).wald_test("D = 0, D_x_lag = 0", scalar=True).statistic
        rows.append((mean_w, chow_w))
    path = pd.DataFrame(rows, index=dates, columns=["Mean shift", "Chow (mean + persistence)"])
    table = pd.DataFrame({
        name: {"sup-Wald": path[name].max(), "Break date": f"{path[name].idxmax():%Y-%m}",
               **{f"CV {k}": v for k, v in ANDREWS_CV[q].items()},
               "Significant at 5%": path[name].max() > ANDREWS_CV[q]["5%"]}
        for name, q in (("Mean shift", 1), ("Chow (mean + persistence)", 2))}).T

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, (name, q) in zip(axes, (("Mean shift", 1), ("Chow (mean + persistence)", 2))):
        ax.plot(path.index, path[name], lw=1)
        ax.axhline(ANDREWS_CV[q]["5%"], color="red", ls="--", lw=0.8, label="Andrews 5% critical value")
        ax.axvline(pd.Timestamp(BREAK_DATES["COVID-19"]), color="gray", ls=":", lw=0.8, label="COVID-19")
        ax.set_title(f"{label}: QLR, HAC Wald statistic by candidate break date ({name})")
        ax.set_ylabel("Wald statistic")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return table, fig


def _segment_ssr(y, x, h):
    """SSR of OLS y on (1, x) for every segment [i, j) at least h long; inf elsewhere.

    Built from running sums, so all ~n^2 / 2 segment regressions cost one pass per start.
    """
    n = len(y)
    ssr = np.full((n + 1, n + 1), np.inf)
    for i in range(n - h + 1):
        yy, xx = y[i:], x[i:]
        m = np.arange(1, n - i + 1)
        sx, sy = np.cumsum(xx), np.cumsum(yy)
        sxx, sxy, syy = np.cumsum(xx * xx), np.cumsum(xx * yy), np.cumsum(yy * yy)
        det = m * sxx - sx ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            # y'y - b'X'y with b = (X'X)^-1 X'y, written out for the 2-regressor case
            fitted = (sxx * sy ** 2 - 2 * sx * sy * sxy + m * sxy ** 2) / det
        seg = syy - fitted
        ok = m >= h
        ssr[i, i + m[ok]] = seg[ok]
    return ssr


def _global_breaks(ssr, n, h, m):
    """Bai-Perron (2003) dynamic programme: the m break points minimizing total SSR.
    Returns (total SSR, break positions = first observation of each new regime)."""
    cost, arg = [ssr[0].copy()], []
    for k in range(1, m + 1):
        c, a = np.full(n + 1, np.inf), np.zeros(n + 1, dtype=int)
        for j in range((k + 1) * h, n + 1):
            cand = cost[k - 1][k * h:j - h + 1] + ssr[k * h:j - h + 1, j]
            a[j] = k * h + int(np.argmin(cand))
            c[j] = cand.min()
        cost.append(c)
        arg.append(a)
    breaks, j = [], n
    for k in range(m, 0, -1):
        j = arg[k - 1][j]
        breaks.insert(0, j)
    return cost[m][n], breaks


def step10_bai_perron(pi_full, label, max_breaks=5, trim=0.15):
    """Bai-Perron multiple breaks in the AR(1) regression (mean and persistence both shift).

    Break dates are the global SSR minimizers for each number of breaks m (dynamic
    programme, segments at least `trim` of the sample). m is chosen three ways: BIC, LWZ
    (BIC with a heavier penalty) and the sequential sup-F(l+1 | l) test at 5%. The F tests
    assume homoskedastic, serially uncorrelated errors, so they are less robust than the
    HAC-based QLR above. Segment estimates are shown for the sequential choice.
    """
    reg = _ar1_data(pi_full)
    y, x, idx = reg["pi"].to_numpy(), reg["pi_lag"].to_numpy(), reg.index
    n, q = len(y), 2
    h = int(np.ceil(trim * n))
    max_breaks = min(max_breaks, n // h - 1)
    ssr = _segment_ssr(y, x, h)
    sols = {m: _global_breaks(ssr, n, h, m) for m in range(max_breaks + 1)}

    rows = {}
    for m, (s, b) in sols.items():
        p = (m + 1) * q + m                       # coefficients plus break dates
        rows[m] = {"SSR": s, "BIC": n * np.log(s / n) + p * np.log(n),
                   "LWZ": n * np.log(s / (n - p)) + p * 0.299 * np.log(n) ** 2.1,
                   "Break dates": ", ".join(f"{idx[k]:%Y-%m}" for k in b) or "none"}
    by_m = pd.DataFrame(rows).T.rename_axis("Breaks (m)")
    m_bic, m_lwz = int(by_m["BIC"].astype(float).idxmin()), int(by_m["LWZ"].astype(float).idxmin())

    # Sequential test: add one break inside whichever current regime it helps most
    seq, m_seq = [], 0
    for l in range(max_breaks):
        s_l, b_l = sols[l]
        edges = [0, *b_l, n]
        best = min(ssr[a, k] + ssr[k, e] - ssr[a, e]
                   for a, e in zip(edges[:-1], edges[1:]) for k in range(a + h, e - h + 1))
        s_new = s_l + best
        f = (s_l - s_new) / (s_new / (n - (l + 2) * q - (l + 1)))
        cv = BP_SEQ_CV[q][l]
        seq.append({"Test": f"sup-F({l + 1} | {l})", "F": f, "CV 5%": cv, "Reject": f > cv})
        if f <= cv:
            break
        m_seq = l + 1
    seq = pd.DataFrame(seq).set_index("Test")

    edges = [0, *sols[m_seq][1], n]
    seg = {}
    for a, e in zip(edges[:-1], edges[1:]):
        r = sm.OLS(y[a:e], sm.add_constant(x[a:e])).fit()
        seg[f"{idx[a]:%Y-%m} to {idx[e - 1]:%Y-%m}"] = {
            "Months": e - a, "Mean": y[a:e].mean(), "AR(1)": r.params[1], "Residual SD": np.sqrt(r.scale)}
    segments = pd.DataFrame(seg).T.astype({"Months": int}).rename_axis("Regime")

    choice = pd.DataFrame({"Breaks": [m_bic, m_lwz, m_seq],
                           "Break dates": [by_m.loc[m, "Break dates"] for m in (m_bic, m_lwz, m_seq)]},
                          index=pd.Index(["BIC", "LWZ", "Sequential sup-F (5%)"], name="Criterion"))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(reg.index, y, lw=0.7, color="C0", alpha=0.7, label="Inflation")
    for i, (a, e) in enumerate(zip(edges[:-1], edges[1:])):
        ax.hlines(y[a:e].mean(), idx[a], idx[e - 1], color="C3", lw=2,
                  label="Regime mean" if i == 0 else None)
    for k in sols[m_seq][1]:
        ax.axvline(idx[k], color="C3", ls="--", lw=0.8)
    ax.set_title(f"{label}: Bai-Perron regimes ({m_seq} break{'s' * (m_seq != 1)}, sequential test)")
    ax.set_ylabel("% per year")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return by_m, seq, choice, segments, fig


def step10_regimes(pi_full, order, seasonal):
    """Selected model estimated pre / post COVID; LR test of one parameter set for both."""
    covid = BREAK_DATES["COVID-19"]
    pre_end = pd.Timestamp(covid) - pd.DateOffset(months=1)
    samples = {"Full sample": pi_full, "Pre-COVID": pi_full.loc[:pre_end], "Post-COVID": pi_full.loc[covid:]}
    fits = {n: fit(y, order, seasonal) for n, y in samples.items()}
    table = pd.DataFrame({
        n: {"Start": f"{samples[n].index[0]:%Y-%m}", "Months": int(r.nobs), "Converged": r.converged,
            "Log-lik": round(r.llf, 1), **{k: round(v, 3) for k, v in r.params.items()}}
        for n, r in fits.items()}).T
    k = len(fits["Full sample"].params)
    lr = 2 * (fits["Pre-COVID"].llf + fits["Post-COVID"].llf - fits["Full sample"].llf)
    return table, {"LR": lr, "df": k, "p": stats.chi2.sf(lr, k)}


# ---------------------------------------------------------------- Step 11-12: forecasts

def step11_forecast(pi_full, pi_train, order, seasonal, label):
    """Selected model, with and without a COVID mean-shift dummy, estimated on the training
    set; multi-step (from TRAIN_END) and one-step-ahead (fixed parameters) forecasts."""
    covid = BREAK_DATES["COVID-19"]
    D_full = pd.Series((pi_full.index >= covid).astype(float), index=pi_full.index, name="COVID")
    D_train = D_full.loc[:TRAIN_END]
    models = {
        "Selected": (fit(pi_train, order, seasonal), None),
        "Selected + COVID": (fit(pi_train, order, seasonal, exog=D_train), D_full),
    }
    h = len(pi_full.loc[TEST_START:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multi, one_step = {}, {}
        for name, (res, exog) in models.items():
            exog_test = None if exog is None else exog.loc[TEST_START:]
            multi[name] = res.get_forecast(steps=h, exog=exog_test).summary_frame(alpha=ALPHA)
            one_step[name] = res.apply(pi_full, exog=exog, refit=False).get_prediction(
                start=TEST_START).summary_frame(alpha=ALPHA)
    covid_coef = models["Selected + COVID"][0]
    covid_row = {"COVID dummy": covid_coef.params["COVID"], "p-value": covid_coef.pvalues["COVID"]}

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (title, fcs) in zip(axes, (("Multi-step from 2023-12", multi), ("One-step-ahead", one_step))):
        ax.plot(pi_full.loc["2022-01":].index, pi_full.loc["2022-01":], color="black", lw=1.2, label="Actual")
        for (name, f), color in zip(fcs.items(), ("C0", "C1")):
            ax.plot(f.index, f["mean"], color=color, lw=1.2, label=name)
            ax.fill_between(f.index, f["mean_ci_lower"], f["mean_ci_upper"], color=color, alpha=0.12)
        ax.axvline(pd.Timestamp(TEST_START), color="gray", ls="--", lw=0.8)
        for name in ("2025 tariffs", "2026 Iran conflict"):
            ax.axvline(pd.Timestamp(BREAK_DATES[name]), color="gray", ls=":", lw=0.8)
        ax.set_title(f"{label}: {title} forecasts, 95% intervals")
        ax.set_ylabel("% per year (annualized)")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return multi, one_step, covid_row, fig


def diebold_mariano(e_other, e_base, h=1):
    """DM test, squared-error loss, HLN small-sample correction; mean(d) > 0 = base better."""
    d = e_other ** 2 - e_base ** 2
    T = len(d)
    dm = d.mean() / np.sqrt(d.var(ddof=0) / T)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return {"Mean loss diff": d.mean(), "DM (HLN)": hln, "p-value": 2 * stats.t.sf(abs(hln), T - 1)}


def step12_evaluate(pi_full, pi_train, multi, one_step, label):
    """Accuracy vs. random-walk / historical-mean benchmarks, RMSE before / after the 2025
    date, Diebold-Mariano vs. the selected one-step forecast, interval coverage."""
    actual = pi_full.loc[TEST_START:]
    fc = pd.DataFrame({
        "Selected, one-step": one_step["Selected"]["mean"],
        "Selected + COVID, one-step": one_step["Selected + COVID"]["mean"],
        "Selected, multi-step": multi["Selected"]["mean"],
        "Selected + COVID, multi-step": multi["Selected + COVID"]["mean"],
        "Random walk": pi_full.shift(1).loc[TEST_START:],
        "Historical mean": pd.Series(pi_train.mean(), index=actual.index),
    })
    err = fc.rsub(actual, axis=0)
    naive_mae = pi_train.diff().abs().mean()
    rw_rmse = np.sqrt((err["Random walk"] ** 2).mean())
    acc = pd.DataFrame({n: {"ME": e.mean(), "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                            "RMSE / RW": np.sqrt((e ** 2).mean()) / rw_rmse,
                            "MASE": e.abs().mean() / naive_mae} for n, e in err.items()}).T

    tariff = pd.Timestamp(BREAK_DATES["2025 tariffs"])
    before, after = err.loc[:tariff - pd.DateOffset(months=1)], err.loc[tariff:]
    split_rmse = pd.DataFrame({f"RMSE to {before.index[-1]:%Y-%m}": np.sqrt((before ** 2).mean()),
                               f"RMSE from {tariff:%Y-%m}": np.sqrt((after ** 2).mean())})
    split_rmse["Ratio (after / before)"] = split_rmse.iloc[:, 1] / split_rmse.iloc[:, 0]

    base = "Selected, one-step"
    dm = pd.DataFrame({n: diebold_mariano(err[n], err[base]) for n in err if n != base}).T

    coverage = pd.DataFrame({
        f"{n}, {kind}": {"95% coverage": ((actual >= f["mean_ci_lower"]) & (actual <= f["mean_ci_upper"])).mean(),
                         "Average width": (f["mean_ci_upper"] - f["mean_ci_lower"]).mean()}
        for kind, fcs in (("one-step", one_step), ("multi-step", multi)) for n, f in fcs.items()}).T

    cum = pd.DataFrame({n: (err[n] ** 2 - err[base] ** 2).cumsum()
                        for n in ("Selected + COVID, one-step", "Random walk", "Historical mean")})
    fig, ax = plt.subplots(figsize=(10, 4))
    for n in cum:
        ax.plot(cum.index, cum[n], lw=1.2, label=f"{n} minus selected")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(tariff, color="gray", ls=":", lw=0.8)
    ax.set_title(f"{label}: cumulative squared-error loss differential vs. selected model, one-step")
    ax.set_ylabel("Cumulative loss difference")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return acc, split_rmse, dm, coverage, fig
