"""
GeoAI analysis engine.

The request reader turns typed words into an analysis plan (rule-based, free).
Every number in the results is calculated here from downloaded data. Nothing is invented.
"""

import io
import re
import unicodedata

import altair as alt
import numpy as np
import pandas as pd
import requests

from places import all_places

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"

POLLUTANTS = {
    "pm2_5": ("PM2.5", 15.0, "24-hour mean"),
    "pm10": ("PM10", 45.0, "24-hour mean"),
    "nitrogen_dioxide": ("NO₂", 25.0, "24-hour mean"),
    "ozone": ("O₃", 100.0, "8-hour mean"),
}
WEATHER = {
    "temperature_2m": "Temperature (°C)",
    "relative_humidity_2m": "Humidity (%)",
    "precipitation": "Precipitation (mm)",
    "wind_speed_10m": "Wind speed (km/h)",
    "wind_direction_10m": "Wind direction (°)",
    "surface_pressure": "Pressure (hPa)",
    "cloud_cover": "Cloud cover (%)",
}
PLACES = all_places()
MAX_DAYS = 92

HELP = """**What you can type** (any place on Earth):

- *Average PM2.5 in Delhi last 30 days by hour*
- *Compare NO2 in Budapest, Berlin and Paris last 2 weeks*
- *PM2.5 trend in Dhaka last 3 months*
- *How does weather affect PM2.5 in Lahore?*
- *Train a random forest to predict PM2.5 in Beijing from weather*
- *Train xgboost for PM10 in Cairo with lags*
- *What is the ozone now in Madrid?*
- *Analyse 23.8, 90.4* (latitude, longitude)

Pollutants: PM2.5, PM10, NO₂, ozone. Periods: up to the last 92 days.
Group with *by hour*, *by weekday*, *by day*. """


def plain(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


# ---------------------------------------------------------------------------
# Request reader
# ---------------------------------------------------------------------------
STOP = r"(?:last|past|over|during|from|with|using|by|vs|versus|and|in the|this|for the|\d)"


def _geocode(name):
    try:
        r = requests.get(GEO_URL, params={"name": name, "count": 1, "language": "en"}, timeout=20)
        r.raise_for_status()
        res = r.json().get("results") or []
        if res:
            g = res[0]
            label = g["name"] + (f", {g['country']}" if g.get("country") else "")
            return label, float(g["latitude"]), float(g["longitude"])
    except Exception:
        pass
    return None


def find_places(text):
    """Return list of (label, lat, lon) and a list of names we could not find."""
    t = plain(text)
    found, missing = [], []
    # coordinates like "23.8, 90.4"
    for la, lo in re.findall(r"(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)", text):
        la, lo = float(la), float(lo)
        if -90 <= la <= 90 and -180 <= lo <= 180:
            found.append((f"{la:.2f}, {lo:.2f}", la, lo))
    # built-in places
    hits = []
    for name, (la, lo, _) in PLACES.items():
        m = re.search(r"\b" + re.escape(plain(name)) + r"\b", t)
        if m:
            hits.append((m.start(), name, la, lo))
    for _, name, la, lo in sorted(hits):
        if name not in [f[0] for f in found]:
            found.append((name, la, lo))
    if found:
        return found, missing
    # anything after in/for/at/of/near -> look it up
    m = re.search(r"\b(?:in|for|at|of|near|around)\s+([a-z][a-z' .\-]+?)(?=\s+" + STOP + r"|[?.!]|$)", t)
    if m:
        for part in re.split(r"\s*(?:,|\band\b|\bvs\b|\bversus\b)\s*", m.group(1)):
            part = part.strip()
            if len(part) < 2 or part in {"the", "hungary air"}:
                continue
            g = _geocode(part)
            (found.append(g) if g else missing.append(part))
    return found, missing


def parse(text):
    t = plain(text)
    if re.search(r"pm\s*10\b", t):
        pol = "pm10"
    elif re.search(r"no\s*2|nitrogen", t):
        pol = "nitrogen_dioxide"
    elif re.search(r"ozone|\bo3\b", t):
        pol = "ozone"
    else:
        pol = "pm2_5"
    m = re.search(r"(\d+)\s*(day|week|month|year)s?\b", t)
    asked = None
    if m:
        asked = int(m.group(1)) * {"day": 1, "week": 7, "month": 30, "year": 365}[m.group(2)]
    elif re.search(r"\byears?\b", t):
        asked = 365
    elif "quarter" in t:
        asked = 92
    elif re.search(r"\bmonths?\b", t):
        asked = 30
    elif re.search(r"\bweeks?\b", t):
        asked = 7
    days = min(asked or 30, MAX_DAYS)

    tasks = []
    if re.search(r"model|predict|machine learning|\bml\b|random forest|xgboost|regression|train", t):
        tasks.append("model")
    if re.search(r"weather|correlat|relationship|relation|affect|influence|wind|temperature|humidity|rain|precip|pressure|cloud", t) and "model" not in tasks:
        tasks.append("weather")
    if re.search(r"trend|getting|increas|decreas|improv|worse|change", t):
        tasks.append("trend")
    if re.search(r"compare|\bvs\b|versus|difference|rank", t):
        tasks.append("compare")
    if re.search(r"average|mean|median|statistic|summary|max|min|by hour|by day|weekday|daily|hourly|distribution|report|analy", t):
        tasks.append("stats")
    now = bool(re.search(r"\bnow\b|right now|current|today|forecast|tomorrow|live", t))
    if not tasks and not now:
        tasks.append("stats")

    group = "hour"
    if re.search(r"weekday|day of (the )?week|by week", t):
        group = "weekday"
    elif re.search(r"by day|daily|per day", t):
        group = "day"

    model = "xgboost" if "xgboost" in t or "xgb" in t else "random_forest"
    lags = bool(re.search(r"\blag|persist|autoregress|previous hour", t))
    return {"pollutant": pol, "days": days, "asked_days": asked, "tasks": tasks, "now": now,
            "group": group, "model": model, "lags": lags}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _multi(url, places, hourly, past_days, forecast_days, extra=None):
    params = {"latitude": ",".join(str(p[1]) for p in places), "longitude": ",".join(str(p[2]) for p in places),
              "hourly": ",".join(hourly), "past_days": past_days, "forecast_days": forecast_days, "timezone": "auto"}
    params.update(extra or {})
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    p = r.json()
    p = p if isinstance(p, list) else [p]
    frames = []
    for (label, _, _), item in zip(places, p):
        df = pd.DataFrame(item["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df["place"] = label
        df["tz"] = item.get("timezone", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def get_air(places, days, forecast=1):
    return _multi(AQ_URL, places, list(POLLUTANTS), days, forecast)


def get_weather(places, days):
    return _multi(WX_URL, places, list(WEATHER), days, 1)


def trim_to_now(df):
    """Keep hours up to 'now' in each place's local time (the APIs return local times)."""
    out = []
    for place, d in df.groupby("place", sort=False):
        tz = d["tz"].iloc[0] or "UTC"
        try:
            now = pd.Timestamp.now(tz=tz).tz_localize(None)
        except Exception:
            now = pd.Timestamp.now("UTC").tz_localize(None)
        out.append(d[d["time"] <= now])
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def daily(df, pol):
    d = df[["place", "time", pol]].dropna().copy()
    d["date"] = d["time"].dt.date
    if pol == "ozone":
        d = d.sort_values("time")
        d["v"] = d.groupby("place")[pol].transform(lambda s: s.rolling(8, min_periods=6).mean())
        return d.groupby(["place", "date"], as_index=False)["v"].max().dropna()
    cnt = d.groupby(["place", "date"])[pol].transform("count")
    d = d[cnt >= 18]
    return d.groupby(["place", "date"], as_index=False)[pol].mean().rename(columns={pol: "v"}).dropna()


def strength(r):
    a = abs(r)
    return "very weak" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong" if a < 0.7 else "very strong"


WHO_RULE = lambda who, axis="y": alt.Chart(pd.DataFrame({"w": [who]})).mark_rule(  # noqa: E731
    strokeDash=[4, 4], color="#ffffff").encode(**{axis: "w:Q"})


# ---------------------------------------------------------------------------
# Analyses — each returns a list of blocks: (kind, content)
# ---------------------------------------------------------------------------
def a_stats(air, pol, group):
    label, who, kind = POLLUTANTS[pol]
    d = daily(air, pol)
    rows = []
    for place, g in air.groupby("place", sort=False):
        s = g[pol].dropna()
        dd = d[d["place"] == place]["v"]
        if s.empty:
            continue
        rows.append({"Place": place, "Mean": s.mean(), "Median": s.median(), "95th pct": s.quantile(.95),
                     "Max hour": s.max(), "Min hour": s.min(), "Std": s.std(),
                     "Days above WHO": int((dd > who).sum()), "Days": int(dd.count())})
    tab = pd.DataFrame(rows).round(1)
    blocks = [("h", f"📊 {label} statistics")]
    for r in rows:
        blocks.append(("text", f"**{r['Place']}**: mean {r['Mean']:.1f} µg/m³ (median {r['Median']:.1f}, "
                               f"highest hour {r['Max hour']:.1f}). Daily value above the WHO {kind} guideline "
                               f"({who:g}) on {r['Days above WHO']} of {r['Days']} days."))
    blocks.append(("table", tab))
    g = air[["place", "time", pol]].dropna().copy()
    if group == "weekday":
        g["key"] = g["time"].dt.day_name().str[:3]
        order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        xenc = alt.X("key:N", sort=order, title="Day of week")
        gt = "day of week"
    elif group == "day":
        g["key"] = g["time"].dt.normalize()
        xenc = alt.X("key:T", title=None)
        gt = "day"
    else:
        g["key"] = g["time"].dt.hour
        xenc = alt.X("key:O", title="Hour of day (local)")
        gt = "hour of day"
    prof = g.groupby(["place", "key"], as_index=False)[pol].mean()
    ch = alt.Chart(prof).mark_line(point=True).encode(x=xenc, y=alt.Y(f"{pol}:Q", title=f"Mean {label} (µg/m³)"),
                                                     color=alt.Color("place:N", title=None),
                                                     tooltip=["place", "key", alt.Tooltip(f"{pol}:Q", format=".1f")])
    blocks.append(("chart", (ch, f"Average {label} by {gt}")))
    if len(prof):
        top = prof.loc[prof[pol].idxmax()]
        low = prof.loc[prof[pol].idxmin()]
        blocks.append(("text", f"Highest average by {gt}: **{top['key']}** ({top[pol]:.1f} µg/m³, {top['place']}); "
                               f"lowest: **{low['key']}** ({low[pol]:.1f}, {low['place']})."
                       if gt != "day" else
                       f"Highest daily average: {pd.Timestamp(top['key']):%d %b} in {top['place']} ({top[pol]:.1f})."))
    return blocks, tab


def a_trend(air, pol):
    label, who, kind = POLLUTANTS[pol]
    d = daily(air, pol)
    blocks = [("h", f"📈 {label} trend")]
    fits = []
    for place, g in d.groupby("place", sort=False):
        if len(g) < 7:
            blocks.append(("text", f"{place}: not enough days for a trend ({len(g)})."))
            continue
        x = np.arange(len(g))
        slope, icpt = np.polyfit(x, g["v"].values, 1)
        pred = slope * x + icpt
        ss_res, ss_tot = ((g["v"] - pred) ** 2).sum(), ((g["v"] - g["v"].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        change = slope * (len(g) - 1)
        pct = 100 * change / g["v"].mean() if g["v"].mean() else 0
        word = "rising" if slope > 0 else "falling"
        blocks.append(("text", f"**{place}**: {word} by about {abs(slope) * 7:.2f} µg/m³ per week "
                               f"({change:+.1f} µg/m³, {pct:+.0f}% over {len(g)} days). The straight line explains "
                               f"{r2 * 100:.0f}% of the day-to-day variation, so "
                               f"{'the trend is clear' if r2 >= .3 else 'weather swings matter more than the trend'}."))
        fits.append(pd.DataFrame({"place": place, "date": pd.to_datetime(g["date"]), "fit": pred}))
    dd = d.copy()
    dd["date"] = pd.to_datetime(dd["date"])
    dd["roll7"] = dd.groupby("place")["v"].transform(lambda s: s.rolling(7, min_periods=3).mean())
    base = alt.Chart(dd).encode(x=alt.X("date:T", title=None), color=alt.Color("place:N", title=None))
    ch = base.mark_line(opacity=.35).encode(y=alt.Y("v:Q", title=f"Daily {label} (µg/m³)")) + \
        base.mark_line(strokeWidth=3).encode(y="roll7:Q")
    if fits:
        ch = ch + alt.Chart(pd.concat(fits)).mark_line(strokeDash=[6, 3]).encode(
            x="date:T", y="fit:Q", color=alt.Color("place:N", title=None))
    blocks.append(("chart", (ch + WHO_RULE(who), f"Daily {label} (thin), 7-day average (thick), trend line (dashed). "
                                                 f"White dashed = WHO guideline {who:g}.")))
    blocks.append(("text", "_A few weeks of data show the recent direction, not a long-term trend._"))
    return blocks


def a_compare(air, pol):
    label, who, kind = POLLUTANTS[pol]
    d = daily(air, pol)
    s = d.groupby("place")["v"].agg(mean="mean", days="count", over=lambda x: int((x > who).sum())).reset_index()
    s = s.sort_values("mean", ascending=False)
    blocks = [("h", f"⚖️ Comparing {label}")]
    if len(s) >= 2:
        hi, lo = s.iloc[0], s.iloc[-1]
        blocks.append(("text", f"**{hi['place']}** had the highest average ({hi['mean']:.1f} µg/m³), "
                               f"**{hi['mean'] / lo['mean']:.1f}×** that of **{lo['place']}** ({lo['mean']:.1f})."
                       if lo["mean"] > 0 else f"{hi['place']} had the highest average."))
    ch = alt.Chart(s).mark_bar(cornerRadiusEnd=4, color="#b5e655").encode(
        x=alt.X("mean:Q", title=f"Mean daily {label} (µg/m³)"), y=alt.Y("place:N", sort="-x", title=None),
        tooltip=["place", alt.Tooltip("mean:Q", format=".1f"), "over", "days"])
    blocks.append(("chart", (ch + WHO_RULE(who, "x"), f"Average daily {label}; white dashed = WHO guideline")))
    return blocks


def _merged(air, wx, pol):
    m = air[["place", "time", pol]].merge(wx[["place", "time"] + list(WEATHER)], on=["place", "time"], how="inner")
    return m.dropna(subset=[pol])


def a_weather(air, wx, pol):
    label, who, kind = POLLUTANTS[pol]
    place = air["place"].iloc[0]
    m = _merged(air[air["place"] == place], wx[wx["place"] == place], pol)
    blocks = [("h", f"🌦️ Weather and {label} in {place}")]
    rows = []
    for v, name in WEATHER.items():
        if v == "wind_direction_10m":
            continue
        x = m[[pol, v]].dropna()
        if len(x) < 30 or x[v].std() == 0:
            continue
        rows.append({"Weather": name, "Pearson r": x[pol].corr(x[v]),
                     "Spearman ρ": x[pol].corr(x[v], method="spearman"), "Hours": len(x)})
    if not rows:
        return blocks + [("text", "Not enough matching weather data.")]
    tab = pd.DataFrame(rows).sort_values("Spearman ρ", key=abs, ascending=False).round(2)
    top = tab.iloc[0]
    blocks.append(("text", f"The strongest link is with **{top['Weather'].split(' (')[0].lower()}**: "
                           f"a {strength(top['Spearman ρ'])} {'positive' if top['Spearman ρ'] > 0 else 'negative'} "
                           f"association (ρ = {top['Spearman ρ']:.2f}) over {top['Hours']} hours. "
                           f"{'Higher' if top['Spearman ρ'] > 0 else 'Lower'} values go with more {label}."))
    for _, r in tab.iloc[1:3].iterrows():
        blocks.append(("text", f"{r['Weather'].split(' (')[0]}: {strength(r['Spearman ρ'])} "
                               f"({'+' if r['Spearman ρ'] > 0 else '−'}), ρ = {r['Spearman ρ']:.2f}."))
    blocks.append(("table", tab))
    bar = alt.Chart(tab).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Spearman ρ:Q", scale=alt.Scale(domain=[-1, 1])), y=alt.Y("Weather:N", sort="-x", title=None),
        color=alt.condition("datum['Spearman ρ'] > 0", alt.value("#ff9f43"), alt.value("#5ab0ff")))
    blocks.append(("chart", (bar, "Correlation of each weather variable with the pollutant (orange = goes up together)")))
    var = [k for k, v in WEATHER.items() if v == top["Weather"]][0]
    sc = m[[pol, var]].dropna().sample(min(3000, len(m)), random_state=1)
    blocks.append(("chart", (alt.Chart(sc).mark_circle(size=14, opacity=.35, color="#b5e655").encode(
        x=alt.X(f"{var}:Q", title=top["Weather"]), y=alt.Y(f"{pol}:Q", title=f"{label} (µg/m³)")),
        f"Each dot is one hour: {top['Weather']} vs {label}")))
    # pollution rose
    w = m[[pol, "wind_direction_10m"]].dropna().copy()
    if len(w) > 50:
        sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        w["from"] = [sectors[int(((a + 22.5) % 360) // 45)] for a in w["wind_direction_10m"]]
        rose = w.groupby("from", as_index=False)[pol].mean()
        hi = rose.loc[rose[pol].idxmax()]
        blocks.append(("chart", (alt.Chart(rose).mark_bar(color="#b36bff", cornerRadiusEnd=4).encode(
            x=alt.X("from:N", sort=sectors, title="Wind from"), y=alt.Y(f"{pol}:Q", title=f"Mean {label}")),
            "Pollution by wind direction")))
        blocks.append(("text", f"{label} is highest on average when the wind comes from the **{hi['from']}** "
                               f"({hi[pol]:.1f} µg/m³), which hints at where sources or polluted air lie."))
    blocks.append(("text", "_These are associations, not proof of cause. Air-quality values are CAMS model data._"))
    return blocks


def a_model(air, wx, pol, which, lags):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    label, who, kind = POLLUTANTS[pol]
    place = air["place"].iloc[0]
    m = _merged(air[air["place"] == place], wx[wx["place"] == place], pol).sort_values("time")
    m["hour_sin"] = np.sin(2 * np.pi * m["time"].dt.hour / 24)
    m["hour_cos"] = np.cos(2 * np.pi * m["time"].dt.hour / 24)
    m["weekday"] = m["time"].dt.dayofweek
    m["wind_u"] = -m["wind_speed_10m"] * np.sin(np.radians(m["wind_direction_10m"]))
    m["wind_v"] = -m["wind_speed_10m"] * np.cos(np.radians(m["wind_direction_10m"]))
    feats = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_u", "wind_v",
             "surface_pressure", "cloud_cover", "hour_sin", "hour_cos", "weekday"]
    if lags:
        m["lag_1h"] = m[pol].shift(1)
        m["lag_24h"] = m[pol].shift(24)
        feats += ["lag_1h", "lag_24h"]
    m = m.dropna(subset=feats + [pol])
    blocks = [("h", f"🤖 Machine learning: predicting {label} in {place}")]
    if len(m) < 200:
        return blocks + [("text", f"Only {len(m)} usable hours. Ask for a longer period (e.g. last 60 days).")], None
    cut = int(len(m) * .75)
    tr, te = m.iloc[:cut], m.iloc[cut:]
    X_tr, y_tr, X_te, y_te = tr[feats], tr[pol], te[feats], te[pol]

    models, note = {}, ""
    models["Linear regression (baseline)"] = LinearRegression().fit(X_tr, y_tr)
    if which == "xgboost":
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=.05, subsample=.8,
                                             colsample_bytree=.8, random_state=42).fit(X_tr, y_tr)
        except ImportError:
            note = "XGBoost is not installed (run `python -m pip install xgboost`), so Random Forest was used."
            which = "random_forest"
    if which == "random_forest":
        models["Random Forest"] = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                                        random_state=42).fit(X_tr, y_tr)
    res = []
    for name, mdl in models.items():
        p = mdl.predict(X_te)
        res.append({"Model": name, "R²": r2_score(y_te, p), "RMSE": mean_squared_error(y_te, p) ** .5,
                    "MAE": mean_absolute_error(y_te, p)})
    res = pd.DataFrame(res)
    best_name = [n for n in models if n != "Linear regression (baseline)"][0]
    best = models[best_name]
    br, base = res[res["Model"] == best_name].iloc[0], res[res["Model"] != best_name].iloc[0]
    if note:
        blocks.append(("text", note))
    blocks.append(("metrics", [("Model", best_name), ("R² (test)", f"{br['R²']:.2f}"),
                               ("RMSE", f"{br['RMSE']:.1f} µg/m³"), ("MAE", f"{br['MAE']:.1f} µg/m³")]))
    blocks.append(("text", f"Trained on {len(tr)} hours ({tr['time'].min():%d %b} → {tr['time'].max():%d %b}) and "
                           f"tested on the **following {len(te)} hours** it had never seen. {best_name} explains "
                           f"**{br['R²'] * 100:.0f}%** of the variation (typical error {br['MAE']:.1f} µg/m³), "
                           f"compared with {base['R²'] * 100:.0f}% for a simple linear regression."))
    if br["R²"] < 0:
        blocks.append(("text", f"⚠️ An R² below 0 means the model did **worse than simply guessing the average** for "
                               f"the test hours. The later weeks were different from the training weeks (mean "
                               f"{y_tr.mean():.1f} → {y_te.mean():.1f} µg/m³), for example a seasonal build-up, and "
                               f"weather alone could not explain the change. Try a longer period (e.g. *last 92 days*) "
                               f"or add *with lags* so the model can use recent pollution levels."))
    pi = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=42)
    imp = pd.DataFrame({"Feature": feats, "Importance": pi.importances_mean}).sort_values("Importance", ascending=False)
    names = {"temperature_2m": "Temperature", "relative_humidity_2m": "Humidity", "precipitation": "Precipitation",
             "wind_speed_10m": "Wind speed", "wind_u": "Wind east-west", "wind_v": "Wind north-south",
             "surface_pressure": "Pressure", "cloud_cover": "Cloud cover", "hour_sin": "Hour of day (sin)",
             "hour_cos": "Hour of day (cos)", "weekday": "Day of week", "lag_1h": f"{label} 1 h before",
             "lag_24h": f"{label} 24 h before"}
    imp["Feature"] = imp["Feature"].map(names)
    blocks.append(("text", "Most important inputs: " + ", ".join(f"**{f}**" for f in imp["Feature"].head(3)) +
                   " (permutation importance on the test hours)."))
    blocks.append(("table", res.round(3)))
    blocks.append(("chart", (alt.Chart(imp.head(10)).mark_bar(color="#b5e655", cornerRadiusEnd=4).encode(
        x=alt.X("Importance:Q", title="Drop in R² when shuffled"), y=alt.Y("Feature:N", sort="-x", title=None)),
        "Feature importance")))
    pr = te[["time"]].copy()
    pr["Observed"] = y_te.values
    pr["Predicted"] = best.predict(X_te)
    long = pr.melt("time", var_name="series", value_name="v")
    blocks.append(("chart", (alt.Chart(long).mark_line().encode(
        x=alt.X("time:T", title=None), y=alt.Y("v:Q", title=f"{label} (µg/m³)"),
        color=alt.Color("series:N", scale=alt.Scale(range=["#ffffff", "#b5e655"]), title=None)),
        "Test period: observed (white) vs predicted (green)")))
    caveat = ("_Inputs: weather + time of day/week" + (" + earlier pollution values (lags make prediction much "
              "easier)" if lags else "") + ". The target is CAMS model output, not station measurements, and "
              "the period is short, so treat this as exploration, not a validated model._")
    blocks.append(("text", caveat))
    buf = io.BytesIO()
    try:
        import joblib
        joblib.dump({"model": best, "features": feats, "target": pol, "place": place}, buf)
        model_bytes = buf.getvalue()
    except Exception:
        model_bytes = None
    files = [("⬇️ Predictions (CSV)", pr.to_csv(index=False), f"GeoAI_predictions_{pol}.csv", "text/csv")]
    if model_bytes:
        files.append(("⬇️ Trained model (.joblib)", model_bytes, f"GeoAI_model_{pol}.joblib", "application/octet-stream"))
    blocks.append(("files", files))
    return blocks, res


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run(text, live_cities=None):
    """Returns dict(title, reply, blocks)."""
    plan = parse(text)
    pol = plan["pollutant"]
    label, who, kind = POLLUTANTS[pol]
    if re.search(r"\bhelp\b|what can (i|you)", plain(text)):
        return {"title": "Help", "reply": HELP, "blocks": [("text", HELP)], "plan": plan}
    places, missing = find_places(text)
    if not places:
        msg = ("I couldn't tell which place you mean" + (f" (I couldn't find: {', '.join(missing)})" if missing else "")
               + ". Name a city or country, or give coordinates like *23.8, 90.4*.")
        return {"title": "Which place?", "reply": msg, "blocks": [], "plan": plan}
    places = places[:8]
    names = ", ".join(p[0] for p in places)
    notes = []
    if missing:
        notes.append(f"I couldn't find: {', '.join(missing)}.")
    if plan["asked_days"] and plan["asked_days"] > MAX_DAYS:
        notes.append(f"This version can look back {MAX_DAYS} days, so I used the last {MAX_DAYS}.")

    blocks, reply = [], []
    if plan["now"]:
        fc = trim_future = get_air(places, 1, 4)
        reply.append(f"**{label} now and forecast:**")
        blocks.append(("h", f"⏱️ {label} now and forecast"))
        rows = []
        for place, g in fc.groupby("place", sort=False):
            tz = g["tz"].iloc[0] or "UTC"
            now = pd.Timestamp.now(tz=tz).tz_localize(None).floor("h")
            past = g[g["time"] <= now].dropna(subset=[pol])
            fut = g[g["time"] > now].dropna(subset=[pol])
            if past.empty:
                continue
            cur = past.iloc[-1]
            pk = fut.loc[fut[pol].idxmax()] if not fut.empty else None
            rows.append({"Place": place, "Now": cur[pol], "Forecast peak": None if pk is None else pk[pol],
                         "Peak time (local)": None if pk is None else f"{pk['time']:%a %d %b %H:%M}"})
            reply.append(f"{place}: **{cur[pol]:.1f}** µg/m³ now ({cur[pol] / who:.1f}× WHO)" +
                         (f", forecast peak {pk[pol]:.0f} on {pk['time']:%a %H:%M}." if pk is not None else "."))
        if rows:
            blocks.append(("table", pd.DataFrame(rows).round(1)))
        fcd = trim_future.dropna(subset=[pol])
        blocks.append(("chart", (alt.Chart(fcd).mark_line().encode(
            x=alt.X("time:T", title=None), y=alt.Y(f"{pol}:Q", title=f"{label} (µg/m³)"),
            color=alt.Color("place:N", title=None)) + WHO_RULE(who), "Yesterday → 4-day forecast (local time)")))
        if not plan["tasks"] or plan["tasks"] == ["stats"] and not re.search(
                r"average|mean|statistic|summary|report|analy", plain(text)):
            return {"title": f"{label} now · {names}", "reply": "\n\n".join(notes + reply), "blocks": blocks,
                    "plan": plan}

    air = trim_to_now(get_air(places, plan["days"]))
    if air[pol].dropna().empty:
        return {"title": "No data", "reply": "No data came back for that place and period.", "blocks": [], "plan": plan}
    start, end = air["time"].min(), air["time"].max()
    head = f"**{label}** · {names} · {start:%d %b %Y} → {end:%d %b %Y}"
    blocks.insert(0, ("text", head))
    tasks = plan["tasks"]
    if len(places) > 1 and "compare" not in tasks and not ({"model", "weather"} & set(tasks)):
        tasks = tasks + ["compare"]
    wx = None
    if {"model", "weather"} & set(tasks):
        wx = trim_to_now(get_weather(places[:1], plan["days"]))
    main_table = None
    for tsk in tasks:
        if tsk == "stats":
            b, main_table = a_stats(air, pol, plan["group"])
            blocks += b
            reply += [x[1] for x in b if x[0] == "text"][:2]
        elif tsk == "trend":
            b = a_trend(air, pol)
            blocks += b
            reply += [x[1] for x in b if x[0] == "text"][:2]
        elif tsk == "compare":
            b = a_compare(air, pol)
            blocks += b
            reply += [x[1] for x in b if x[0] == "text"][:1]
        elif tsk == "weather":
            b = a_weather(air, wx, pol)
            blocks += b
            reply += [x[1] for x in b if x[0] == "text"][:1]
        elif tsk == "model":
            b, _ = a_model(air, wx, pol, plan["model"], plan["lags"])
            blocks += b
            reply += [x[1] for x in b if x[0] == "text"][:2]
    if len(places) > 1 and {"model", "weather"} & set(tasks):
        notes.append(f"Weather and ML analyses use the first place ({places[0][0]}).")
    reply = notes + [head] + reply + ["Full results are in the 🔬 Analysis section below the map."]
    blocks.append(("text", "_Data: CAMS air-quality model (Copernicus) and weather via Open-Meteo. "
                           f"WHO {kind} guideline for {label}: {who:g} µg/m³._"))
    return {"title": f"{label} · {names}", "reply": "\n\n".join(reply), "blocks": blocks, "plan": plan,
            "table": main_table}
