"""
GeoAI ML Lab: MLR, Decision Tree, Random Forest and XGBoost for air-quality data,
with figures and a Word (.docx) report.

Every number comes from the data and the fitted models. Text is generated from those numbers.
"""

import datetime as dt
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

WX_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER = {
    "temperature_2m": "Temperature (°C)",
    "relative_humidity_2m": "Relative humidity (%)",
    "precipitation": "Precipitation (mm)",
    "wind_speed_10m": "Wind speed (km/h)",
    "surface_pressure": "Surface pressure (hPa)",
    "cloud_cover": "Cloud cover (%)",
    "boundary_layer_height": "Boundary layer height (m)",
}
DERIVED = {
    "wind_u": "Wind east–west component",
    "wind_v": "Wind north–south component",
    "hour_sin": "Hour of day (sine)",
    "hour_cos": "Hour of day (cosine)",
    "weekday": "Day of week",
}
MODELS = ["Multiple linear regression", "Decision tree", "Random forest", "XGBoost"]
GREEN = "#5fae0e"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_weather(lat, lon, start, end):
    hourly = list(WEATHER) + ["wind_direction_10m"]
    p = {"latitude": lat, "longitude": lon, "hourly": ",".join(hourly), "start_date": str(start),
         "end_date": str(end), "timezone": "auto"}
    r = requests.get(WX_URL, params=p, timeout=90)
    if r.status_code != 200:  # some variables are not always available: try without boundary layer height
        p["hourly"] = ",".join(h for h in hourly if h != "boundary_layer_height")
        r = requests.get(WX_URL, params=p, timeout=90)
    r.raise_for_status()
    w = pd.DataFrame(r.json()["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    return w


def build_table(aq, wx, pol, use_time=True, lags=False):
    d = aq[["time", pol]].merge(wx, on="time", how="inner").sort_values("time").reset_index(drop=True)
    feats = [c for c in WEATHER if c in d.columns and d[c].notna().mean() > 0.8]
    if "wind_direction_10m" in d and "wind_speed_10m" in d:
        rad = np.radians(d["wind_direction_10m"])
        d["wind_u"] = -d["wind_speed_10m"] * np.sin(rad)
        d["wind_v"] = -d["wind_speed_10m"] * np.cos(rad)
        feats += ["wind_u", "wind_v"]
    if use_time:
        d["hour_sin"] = np.sin(2 * np.pi * d["time"].dt.hour / 24)
        d["hour_cos"] = np.cos(2 * np.pi * d["time"].dt.hour / 24)
        d["weekday"] = d["time"].dt.dayofweek
        feats += ["hour_sin", "hour_cos", "weekday"]
    if lags:
        d["lag_1h"] = d[pol].shift(1)
        d["lag_24h"] = d[pol].shift(24)
        feats += ["lag_1h", "lag_24h"]
    d = d.dropna(subset=feats + [pol]).reset_index(drop=True)
    return d, feats


def nice(f, label=""):
    if f in WEATHER:
        return WEATHER[f]
    if f in DERIVED:
        return DERIVED[f]
    if f == "lag_1h":
        return f"{label} 1 hour before"
    if f == "lag_24h":
        return f"{label} 24 hours before"
    return f


def split(d, how):
    if how.startswith("Time"):
        cut = int(len(d) * 0.75)
        return d.iloc[:cut], d.iloc[cut:]
    idx = np.random.default_rng(42).permutation(len(d))
    cut = int(len(d) * 0.75)
    return d.iloc[np.sort(idx[:cut])], d.iloc[np.sort(idx[cut:])]


def metrics(y, p):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"R²": float(r2_score(y, p)), "RMSE": float(mean_squared_error(y, p) ** 0.5),
            "MAE": float(mean_absolute_error(y, p))}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def run_model(name, tr, te, feats, pol, params=None):
    from sklearn.inspection import permutation_importance
    params = params or {}
    Xtr, ytr, Xte, yte = tr[feats], tr[pol], te[feats], te[pol]
    extra = {}
    if name == "Multiple linear regression":
        from scipy import stats
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(Xtr, ytr)
        # ordinary least squares statistics (standard errors, t, p)
        X = np.column_stack([np.ones(len(Xtr)), Xtr.values])
        resid = ytr.values - m.predict(Xtr)
        dof = max(len(X) - X.shape[1], 1)
        sigma2 = resid @ resid / dof
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(X.shape[1], np.nan)
        coefs = np.concatenate([[m.intercept_], m.coef_])
        t = coefs / se
        p = 2 * stats.t.sf(np.abs(t), dof)
        # standardised coefficients (effect of +1 standard deviation)
        std_beta = np.concatenate([[np.nan], m.coef_ * Xtr.std().values / ytr.std()])
        extra["coefficients"] = pd.DataFrame({
            "Variable": ["(Intercept)"] + feats, "Coefficient": coefs, "Std. error": se, "t": t, "p-value": p,
            "Standardised β": std_beta})
        # variance inflation factors
        vif = []
        for i, f in enumerate(feats):
            others = [g for g in feats if g != f]
            if not others:
                vif.append(1.0)
                continue
            r2 = LinearRegression().fit(Xtr[others], Xtr[f]).score(Xtr[others], Xtr[f])
            vif.append(1 / max(1 - r2, 1e-9))
        extra["coefficients"]["VIF"] = [np.nan] + vif
        extra["adj_r2_train"] = 1 - (1 - m.score(Xtr, ytr)) * (len(Xtr) - 1) / dof
    elif name == "Decision tree":
        from sklearn.tree import DecisionTreeRegressor, export_text
        depth = params.get("max_depth", 4)
        m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(10, len(Xtr) // 100), random_state=42)
        m.fit(Xtr, ytr)
        extra["rules"] = export_text(m, feature_names=feats, decimals=1)
        extra["depth"] = depth
    elif name == "Random forest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=params.get("n_estimators", 300), min_samples_leaf=3, oob_score=True,
                                  n_jobs=-1, random_state=42).fit(Xtr, ytr)
        extra["oob_r2"] = float(m.oob_score_)
    elif name == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise RuntimeError("XGBoost is not installed. In the terminal run: python -m pip install xgboost")
        m = XGBRegressor(n_estimators=params.get("n_estimators", 400), max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42).fit(Xtr, ytr)
    else:
        raise ValueError(name)
    pred_tr, pred_te = m.predict(Xtr), m.predict(Xte)
    pi = permutation_importance(m, Xte, yte, n_repeats=5, random_state=42)
    imp = pd.DataFrame({"Variable": feats, "Importance": pi.importances_mean,
                        "Std": pi.importances_std}).sort_values("Importance", ascending=False)
    return {"name": name, "model": m, "train": metrics(ytr, pred_tr), "test": metrics(yte, pred_te),
            "pred": pd.DataFrame({"time": te["time"].values, "Observed": yte.values, "Predicted": pred_te}),
            "importance": imp, "extra": extra, "n_train": len(tr), "n_test": len(te),
            "train_period": (tr["time"].min(), tr["time"].max()), "test_period": (te["time"].min(), te["time"].max())}


# ---------------------------------------------------------------------------
# Figures (PNG bytes)
# ---------------------------------------------------------------------------
def _png(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return b.getvalue()


def fig_scatter(res, label):
    p = res["pred"]
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.scatter(p["Observed"], p["Predicted"], s=8, alpha=0.45, color=GREEN)
    lo, hi = float(min(p.min(numeric_only=True))), float(max(p.max(numeric_only=True)))
    ax.plot([lo, hi], [lo, hi], color="#d7191c", lw=1, ls="--", label="1:1 line")
    ax.set_xlabel(f"Observed {label} (µg/m³)")
    ax.set_ylabel(f"Predicted {label} (µg/m³)")
    ax.set_title(f"{res['name']} · test R² = {res['test']['R²']:.2f}", fontsize=10)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return _png(fig)


def fig_timeseries(res, label):
    p = res["pred"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(p["time"], p["Observed"], color="#1d2733", lw=1, label="Observed")
    ax.plot(p["time"], p["Predicted"], color=GREEN, lw=1.3, label="Predicted")
    ax.set_ylabel(f"{label} (µg/m³)")
    ax.set_title(f"{res['name']} · test period", fontsize=10)
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    return _png(fig)


def fig_importance(res, label):
    imp = res["importance"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh([nice(v, label) for v in imp["Variable"]], imp["Importance"], xerr=imp["Std"], color=GREEN)
    ax.set_xlabel("Drop in R² when the variable is shuffled")
    ax.set_title(f"{res['name']} · permutation importance", fontsize=10)
    ax.grid(axis="x", alpha=0.2)
    return _png(fig)


def fig_tree(res, feats, label):
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(14, 6.5))
    plot_tree(res["model"], feature_names=[nice(f, label) for f in feats], filled=True, rounded=True,
              precision=1, max_depth=3, fontsize=7, ax=ax, impurity=False)
    ax.set_title("Decision tree (first 3 levels) · each box: rule, number of hours, mean value", fontsize=10)
    return _png(fig)


def fig_compare(results):
    names = list(results)
    r2 = [results[n]["test"]["R²"] for n in names]
    rmse = [results[n]["test"]["RMSE"] for n in names]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    axs[0].bar(names, r2, color=GREEN)
    axs[0].set_title("Test R² (higher is better)", fontsize=10)
    axs[0].axhline(0, color="black", lw=0.8)
    axs[1].bar(names, rmse, color="#2c7bb6")
    axs[1].set_title("Test RMSE, µg/m³ (lower is better)", fontsize=10)
    for a in axs:
        a.tick_params(axis="x", labelrotation=20, labelsize=8)
        a.grid(axis="y", alpha=0.2)
    return _png(fig)


def fig_corr(d, feats, pol, label):
    cols = [pol] + [f for f in feats if f not in ("hour_sin", "hour_cos", "weekday", "lag_1h", "lag_24h")]
    c = d[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    names = [label] + [nice(x, label).split(" (")[0] for x in cols[1:]]
    ax.set_xticks(range(len(cols)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), names, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(c.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title("Correlation matrix", fontsize=10)
    return _png(fig)


# ---------------------------------------------------------------------------
# Text written from the numbers
# ---------------------------------------------------------------------------
def describe(res, label):
    t, tr = res["test"], res["train"]
    out = [f"{res['name']} was trained on {res['n_train']} hours and tested on {res['n_test']} hours it had not seen. "
           f"On the test hours it explained {t['R²'] * 100:.0f}% of the variation in {label} (R² = {t['R²']:.2f}), "
           f"with a typical error of {t['MAE']:.1f} µg/m³ (MAE) and RMSE = {t['RMSE']:.1f} µg/m³."]
    if tr["R²"] - t["R²"] > 0.25:
        out.append(f"The training R² ({tr['R²']:.2f}) is much higher than the test R², which suggests overfitting: "
                   "the model learned patterns of the training period that did not carry over.")
    if t["R²"] < 0:
        out.append("A negative R² means the model did worse than simply predicting the average of the test period. "
                   "The test period probably differed from the training period (for example a seasonal change).")
    top = [nice(v, label) for v in res["importance"]["Variable"].head(3)]
    out.append("The most important variables were " + ", ".join(top[:-1]) + f" and {top[-1]}." if len(top) > 1 else "")
    if res["name"] == "Multiple linear regression":
        c = res["extra"]["coefficients"].iloc[1:]
        sig = c[c["p-value"] < 0.05].sort_values("Standardised β", key=abs, ascending=False)
        if len(sig):
            parts = [f"{nice(r['Variable'], label)} ({'+' if r['Coefficient'] > 0 else '−'}{abs(r['Coefficient']):.3g} "
                     f"µg/m³ per unit, standardised β = {r['Standardised β']:.2f})" for _, r in sig.head(4).iterrows()]
            out.append("Statistically significant predictors (p < 0.05): " + "; ".join(parts) + ".")
        hv = c[c["VIF"] > 10]
        if len(hv):
            out.append("Some predictors are strongly correlated with each other (VIF > 10: " +
                       ", ".join(nice(v, label) for v in hv["Variable"]) + "), so their individual coefficients "
                       "should be interpreted with care.")
    if res["name"] == "Random forest" and "oob_r2" in res["extra"]:
        out.append(f"The out-of-bag R² on the training data was {res['extra']['oob_r2']:.2f}.")
    if res["name"] == "Decision tree":
        out.append(f"The tree has a maximum depth of {res['extra']['depth']}, which keeps its rules readable.")
    return " ".join(x for x in out if x)


# ---------------------------------------------------------------------------
# Word report
# ---------------------------------------------------------------------------
def word_report(*, results, d, feats, pol, label, who, place, start, end, split_how, lags, sources):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2)
        s.top_margin = s.bottom_margin = Cm(2)

    def table(df, decimals=3):
        tb = doc.add_table(rows=1, cols=len(df.columns))
        tb.style = "Light Grid Accent 1"
        for i, c in enumerate(df.columns):
            tb.rows[0].cells[i].text = str(c)
        for row in df.itertuples(index=False):
            cells = tb.add_row().cells
            for i, v in enumerate(row):
                if df.columns[i] == "p-value" and isinstance(v, (float, np.floating)) and np.isfinite(v) and v < 0.001:
                    cells[i].text = "< 0.001"
                elif isinstance(v, (float, np.floating)):
                    cells[i].text = "–" if not np.isfinite(v) else (f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.{decimals}f}")
                else:
                    cells[i].text = str(v)
        for r in tb.rows:
            for c in r.cells:
                for p in c.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(9)
        doc.add_paragraph()

    def picture(png, width=16, caption=None):
        doc.add_picture(io.BytesIO(png), width=Cm(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            p = doc.add_paragraph(caption)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.runs[0].font.size = Pt(9)
            p.runs[0].italic = True

    h = doc.add_heading(f"{label} at {place}: machine-learning analysis", 0)
    h.runs[0].font.color.rgb = RGBColor(0x1D, 0x27, 0x33)
    meta = doc.add_paragraph()
    meta.add_run(f"Period: {start:%d %B %Y} – {end:%d %B %Y}   ·   Generated: {dt.datetime.now():%d %B %Y %H:%M}"
                 "   ·   GeoAI").font.size = Pt(9)

    # 1 Summary
    doc.add_heading("1. Summary", 1)
    best = max(results.values(), key=lambda r: r["test"]["R²"])
    doc.add_paragraph(
        f"This report models hourly {label} at {place} from weather" + (" and time-of-day" if "hour_sin" in feats else "")
        + (" and earlier pollution values" if lags else "") + f" using {len(results)} method(s): "
        + ", ".join(results) + f". The best result on unseen test hours came from {best['name']} "
        f"(R² = {best['test']['R²']:.2f}, RMSE = {best['test']['RMSE']:.1f} µg/m³).")
    mean = d[pol].mean()
    doc.add_paragraph(f"Over the period, the mean {label} was {mean:.1f} µg/m³ (median {d[pol].median():.1f}, "
                      f"maximum {d[pol].max():.1f}). The WHO 2021 short-term guideline for {label} is {who:g} µg/m³.")

    # 2 Data
    doc.add_heading("2. Data", 1)
    doc.add_paragraph(f"{len(d)} hourly records with complete data were used. Target: {label} (µg/m³). Predictors: "
                      + ", ".join(nice(f, label) for f in feats) + ".")
    desc = d[[pol] + feats].describe().T[["mean", "std", "min", "50%", "max"]].reset_index()
    desc.columns = ["Variable", "Mean", "Std", "Min", "Median", "Max"]
    desc["Variable"] = [label] + [nice(f, label) for f in feats]
    table(desc, 2)
    picture(fig_corr(d, feats, pol, label), 13, "Figure 1. Spearman correlations between the pollutant and weather.")

    # 3 Methods
    doc.add_heading("3. Methods", 1)
    doc.add_paragraph(
        ("The data were split in time: the first 75% of hours for training and the last 25% for testing, so "
         "every model is judged on a later period it has not seen. " if split_how.startswith("Time") else
         "The hours were split at random: 75% for training and 25% for testing (random seed 42). Random splits "
         "usually give higher scores than time-based splits because neighbouring hours are similar. ")
        + "Performance is reported as R² (share of variation explained), RMSE and MAE in µg/m³. Variable "
        "importance is measured by permutation on the test hours (how much R² drops when a variable is shuffled).")
    methods_txt = {
        "Multiple linear regression": "Multiple linear regression (ordinary least squares) with standard errors, "
                                      "t-statistics, p-values, standardised coefficients and variance inflation factors (VIF).",
        "Decision tree": "A regression tree (CART) with limited depth and a minimum number of hours per leaf, "
                         "giving readable if–then rules.",
        "Random forest": "An ensemble of 300 regression trees on bootstrap samples (minimum 3 hours per leaf), with "
                         "out-of-bag validation.",
        "XGBoost": "Gradient-boosted trees (400 trees, depth 5, learning rate 0.05, 80% row and column sampling).",
    }
    for n in results:
        doc.add_paragraph(methods_txt[n], style="List Bullet")

    # 4 Results
    doc.add_heading("4. Results", 1)
    comp = pd.DataFrame([{"Model": n, "Train R²": r["train"]["R²"], "Test R²": r["test"]["R²"],
                          "Test RMSE": r["test"]["RMSE"], "Test MAE": r["test"]["MAE"]} for n, r in results.items()])
    table(comp, 3)
    if len(results) > 1:
        picture(fig_compare(results), 15, "Figure 2. Model comparison on the test hours.")
    fig_no = 3
    for n, r in results.items():
        doc.add_heading(f"4.{list(results).index(n) + 1} {n}", 2)
        doc.add_paragraph(describe(r, label))
        if n == "Multiple linear regression":
            c = r["extra"]["coefficients"].copy()
            c["Variable"] = ["(Intercept)"] + [nice(v, label) for v in c["Variable"].iloc[1:]]
            table(c, 3)
        picture(fig_scatter(r, label), 9, f"Figure {fig_no}. Observed vs predicted, test hours ({n}).")
        fig_no += 1
        picture(fig_timeseries(r, label), 16, f"Figure {fig_no}. Test period over time ({n}).")
        fig_no += 1
        picture(fig_importance(r, label), 12, f"Figure {fig_no}. Permutation importance ({n}).")
        fig_no += 1
        if n == "Decision tree":
            picture(fig_tree(r, feats, label), 16.5, f"Figure {fig_no}. The fitted decision tree (top levels).")
            fig_no += 1

    # 5 Limitations
    doc.add_heading("5. Limitations", 1)
    for txt in [
        "The target values are CAMS model estimates (about 11 km in Europe, coarser elsewhere), not station "
        "measurements, so the models learn how the CAMS model responds to weather.",
        f"The period is short ({(end - start).days + 1} days). Results may not hold in other seasons.",
        "Associations between weather and pollution do not prove cause and effect; emissions are not included.",
    ] + (["Lagged pollution values make prediction much easier but mostly reflect persistence, not drivers."]
         if lags else []):
        doc.add_paragraph(txt, style="List Bullet")

    doc.add_heading("6. Data sources", 1)
    doc.add_paragraph(sources)
    b = io.BytesIO()
    doc.save(b)
    return b.getvalue()
