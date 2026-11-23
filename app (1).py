"""
GeoAI - World air quality: live globe, satellite data, reports, QGIS export,
and analysis / machine learning by typing.

Run:  python -m streamlit run app.py
"""

import datetime as dt
import hashlib
import html
import json
import pickle
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

import boundaries as B
import engine
import mapmaker
import mllab
import satellite as sat
from engine import POLLUTANTS
from places import all_places

st.set_page_config(page_title="GeoAI · World air quality", page_icon="🌍", layout="wide")
HERE = Path(__file__).parent
PLACES = all_places()
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
TODAY = dt.date.today()

SOURCE = ("Air quality: CAMS global and European air-quality models (Copernicus Atmosphere Monitoring Service) via the "
          "Open-Meteo Air Quality API. Satellite: Copernicus Sentinel-5P and NASA MODIS via Google Earth Engine. "
          "Weather and place search: Open-Meteo. Basemap: Esri World Imagery. Guidelines: WHO Global Air Quality "
          "Guidelines (2021). CAMS values are modelled, not station measurements.")


def _clean(seq):
    return [None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 2) for v in seq]


# ---------------------------------------------------------------------------
# Data downloads
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_world_cities():
    items = list(PLACES.items())
    out = []
    for i in range(0, len(items), 100):
        part = items[i:i + 100]
        params = {"latitude": ",".join(str(v[0]) for _, v in part), "longitude": ",".join(str(v[1]) for _, v in part),
                  "hourly": ",".join(POLLUTANTS), "past_days": 1, "forecast_days": 3, "timezone": "GMT"}
        r = requests.get(engine.AQ_URL, params=params, timeout=90)
        r.raise_for_status()
        p = r.json()
        out += p if isinstance(p, list) else [p]
    times = out[0]["hourly"]["time"]
    cities = [{"name": n, "country": v[2], "lat": v[0], "lon": v[1],
               "v": {p: _clean(o["hourly"].get(p, [None] * len(times))) for p in POLLUTANTS}}
              for (n, v), o in zip(items, out)]
    now = pd.Timestamp.now("UTC").tz_localize(None).floor("h")
    ni = int(np.argmin(np.abs((pd.to_datetime(times) - now).total_seconds())))
    return {"times": times, "now_index": ni, "cities": cities, "fetched": f"{pd.Timestamp.now('UTC'):%H:%M} UTC"}


@st.cache_data(ttl=3600, show_spinner=False)
def cams_selection(lat, lon, pol, start, end, whole_country=False, wide=False):
    """Hourly values at the place (all pollutants) + period-mean grid for the chosen pollutant,
    either around the city (±2.5°, 0.25°) or over the whole country (clipped to its borders)."""
    p = {"latitude": lat, "longitude": lon, "hourly": ",".join(POLLUTANTS), "start_date": str(start),
         "end_date": str(end), "timezone": "auto"}
    r = requests.get(engine.AQ_URL, params=p, timeout=90)
    r.raise_for_status()
    j = r.json()
    series = pd.DataFrame(j["hourly"])
    series["time"] = pd.to_datetime(series["time"])

    hy = 9.0 if wide else 2.0
    hx = float(min(hy / max(np.cos(np.radians(lat)), 0.3), 4 * hy))  # ~square on the map
    area = {"name": None, "bbox": (lon - hx, lat - hy, lon + hx, lat + hy), "outer": None, "regions": [],
            "iso3": None}
    res = 0.75 if wide else 0.25
    country = None
    try:
        country = B.country_at(lon, lat)
    except Exception:
        country = None
    if whole_country and country:
        allr = np.vstack(country["outer"])
        x0, y0 = allr.min(axis=0)
        x1, y1 = allr.max(axis=0)
        if x1 - x0 > 60:  # e.g. countries crossing the date line: fall back to the city surroundings
            whole_country = False
        else:
            area = {"name": country["name"], "bbox": (x0, y0, x1, y1), "outer": country["outer"],
                    "regions": B.regions(country["iso3"]), "iso3": country["iso3"]}
            res = float(np.clip(np.ceil(np.sqrt((x1 - x0) * (y1 - y0) / 600) / 0.05) * 0.05, 0.1, 1.0))
    if not whole_country:
        area["name"] = f"{country['name']} (around the city)" if country else "around the city"
        try:
            area["borders"] = B.borders_in(area["bbox"])
        except Exception:
            area["borders"] = []
        if country:
            area["regions"] = [rg for rg in B.regions(country["iso3"])]
            area["iso3"] = country["iso3"]
    x0, y0, x1, y1 = area["bbox"]
    lats = np.round(np.arange(y0 - res, y1 + res + 1e-9, res), 3)
    lons = np.round(np.arange(x0 - res, x1 + res + 1e-9, res), 3)
    lats = lats[(lats > -85) & (lats < 85)]
    gx, gy = np.meshgrid(lons, lats)
    want = np.ones(gx.shape, dtype=bool)
    if area["outer"]:
        want = B.contains(area["outer"], gx, gy, radius=res * 1.5)
    pts = [(la, lo) for la, lo in zip(gy[want], gx[want])]
    means = []
    for i in range(0, len(pts), 100):
        part = pts[i:i + 100]
        q = {"latitude": ",".join(str(x[0]) for x in part), "longitude": ",".join(str(x[1]) for x in part),
             "hourly": pol, "start_date": str(start), "end_date": str(end), "timezone": "GMT"}
        rr = requests.get(engine.AQ_URL, params=q, timeout=120)
        rr.raise_for_status()
        resj = rr.json()
        resj = resj if isinstance(resj, list) else [resj]
        for item in resj:
            vals = np.array([np.nan if v is None else v for v in item["hourly"][pol]], dtype=float)
            means.append(np.nanmean(vals) if np.isfinite(vals).any() else np.nan)
    grid = np.full(gx.shape, np.nan)
    grid[want] = means
    if not area["outer"]:  # keep regions that touch the box only
        bx0, by0, bx1, by1 = area["bbox"]
        area["regions"] = [rg for rg in area["regions"]
                           if any((r[:, 0].max() > bx0) and (r[:, 0].min() < bx1) and (r[:, 1].max() > by0)
                                  and (r[:, 1].min() < by1) for r in rg["outer"])]
    return {"series": series, "tz": j.get("timezone", ""), "lats": lats.tolist(), "lons": lons.tolist(),
            "grid": grid.tolist(), "res": res, "area": area, "points": len(pts)}


SURF_LATS = np.arange(-54, 79, 6.0)
SURF_LONS = np.arange(-180, 180, 6.0)


def surface_cached():
    f = CACHE_DIR / "world_surface.pkl"
    if f.exists() and time.time() - f.stat().st_mtime < 6 * 3600:
        return pickle.loads(f.read_bytes())
    return None


def fetch_world_surface(progress):
    cached = surface_cached()
    if cached:
        return cached
    pts = [(la, lo) for la in SURF_LATS for lo in SURF_LONS]
    vals = []
    for i in range(0, len(pts), 100):
        part = pts[i:i + 100]
        params = {"latitude": ",".join(str(p[0]) for p in part), "longitude": ",".join(str(p[1]) for p in part),
                  "hourly": "pm2_5", "past_days": 1, "forecast_days": 3, "timezone": "GMT"}
        r = requests.get(engine.AQ_URL, params=params, timeout=90)
        r.raise_for_status()
        p = r.json()
        p = p if isinstance(p, list) else [p]
        vals += [np.array([np.nan if v is None else v for v in x["hourly"]["pm2_5"]], dtype=float) for x in p]
        progress(min(1.0, (i + 100) / len(pts)))
    arr = np.stack(vals)[:, ::3]
    nr, nc, nt = len(SURF_LATS), len(SURF_LONS), arr.shape[1]
    cube = arr.reshape(nr, nc, nt)
    fl = np.arange(SURF_LATS[0], SURF_LATS[-1] + 0.01, 3.0)
    fo = np.arange(-180, 180, 3.0)
    out = np.full((len(fl), len(fo), nt), np.nan)
    lon_ext = np.concatenate([SURF_LONS, [SURF_LONS[0] + 360]])
    for t in range(nt):
        z = np.concatenate([cube[:, :, t], cube[:, :1, t]], axis=1)
        tmp = np.array([np.interp(fo, lon_ext, row) for row in z])
        out[:, :, t] = np.array([np.interp(fl, SURF_LATS, col) for col in tmp.T]).T
    res = {"lats": fl.tolist(), "lons": fo.tolist(), "dlat": 3.0, "dlon": 3.0, "step": 3,
           "values": [_clean(out[:, :, t].ravel()) for t in range(nt)]}
    (CACHE_DIR / "world_surface.pkl").write_bytes(pickle.dumps(res))
    return res


def static_base():
    """Path prefix of the app. On Streamlit Community Cloud apps live under '/~/+', locally under '/'."""
    try:
        from urllib.parse import urlparse
        u = urlparse(st.context.url or "")
        path = u.path.rstrip("/")
        if "/~/+" in path:
            return path[:path.index("/~/+") + 4]
        if (u.hostname or "").endswith(".streamlit.app"):
            return "/~/+"
        return path if path not in ("", "/") else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Styling (light)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1750px;}
button[kind="primary"], button[kind="primary"] p, [data-testid="stBaseButton-primary"] p {color:#0b0f17 !important; font-weight:700;}
[data-testid="stButtonGroup"] button[aria-checked="true"] {background:#9bd13d !important; border-color:#9bd13d !important;}
[data-testid="stButtonGroup"] button[aria-checked="true"] p {color:#0b0f17 !important; font-weight:700;}
.brand {display:flex; align-items:center; gap:12px;}
.brand .logo {background:#9bd13d; color:#0b0f17; font-weight:900; border-radius:10px; padding:4px 11px; font-size:1.3rem;}
.brand h2 {margin:0;}
.live {display:inline-flex; align-items:center; gap:6px; color:#3f7d0b; font-weight:700; font-size:.9rem;}
.dot {width:9px; height:9px; border-radius:50%; background:#5fae0e; animation:pulse 1.4s infinite;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(95,174,14,.6)}70%{box-shadow:0 0 0 9px rgba(95,174,14,0)}100%{box-shadow:0 0 0 0 rgba(95,174,14,0)}}
.kpi {background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:12px 16px; height:100%;
      box-shadow:0 1px 3px rgba(16,24,40,.06);}
.kpi .lab {color:#6b7686; font-size:.75rem; text-transform:uppercase; letter-spacing:.06em;}
.kpi .val {font-size:1.5rem; font-weight:800; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#1d2733;}
.kpi .sub {color:#6b7686; font-size:.85rem;}
.src {color:#6b7686; font-size:.8rem;}
.pill {border-radius:999px; padding:1px 9px; font-size:.75rem; font-weight:700; color:#0b0f17;}
</style>
""", unsafe_allow_html=True)


def kpi(col, lab, val, sub=""):
    col.markdown(f'<div class="kpi"><div class="lab">{lab}</div><div class="val">{val}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def level(ratio):
    for lim, name, col in [(1, "Within WHO guideline", "#7bd88f"), (2, "Up to 2× guideline", "#f5d547"),
                           (3, "2–3× guideline", "#ff9f43"), (5, "3–5× guideline", "#ff5c5c"),
                           (1e9, "Over 5× guideline", "#b36bff")]:
        if ratio <= lim:
            return name, col


# ---------------------------------------------------------------------------
# State + assistant callbacks
# ---------------------------------------------------------------------------
st.session_state.setdefault("chat", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("result", None)
st.session_state.setdefault("sel", None)
settings = sat.load_settings()
st.session_state.setdefault("ee_project", settings.get("ee_project", ""))
st.session_state.setdefault("ee_ok", False)

SUGGESTIONS = [
    "What is PM2.5 now in Debrecen and Delhi?",
    "Average PM2.5 in Dhaka last 30 days by hour",
    "Compare NO2 in Budapest, Berlin and Paris last 2 weeks",
    "How does weather affect PM2.5 in Beijing?",
    "Train a random forest to predict PM2.5 in Delhi last 92 days with lags",
    "help",
]


def submit(text):
    st.session_state["chat"].append(("user", text))
    st.session_state["pending"] = text


def on_chat():
    t = (st.session_state.get("chat_in") or "").strip()
    if t:
        submit(t)


def on_suggest():
    t = st.session_state.get("suggest")
    if t:
        submit(t)
    st.session_state["suggest"] = None


# ---------------------------------------------------------------------------
# Live data + header
# ---------------------------------------------------------------------------
try:
    with st.spinner("🛰️ Loading live air quality for world cities…"):
        live = fetch_world_cities()
except Exception as e:
    st.error("Could not download live data from Open-Meteo. Check your internet connection, then refresh (F5).")
    st.caption(f"Technical detail: {e}")
    st.stop()
ni = live["now_index"]
now_df = pd.DataFrame([{"City": c["name"], "Country": c["country"],
                        **{POLLUTANTS[p][0]: c["v"][p][ni] for p in POLLUTANTS}} for c in live["cities"]])

h1, h2 = st.columns([3, 2], vertical_alignment="center")
h1.markdown('<div class="brand"><span class="logo">G</span><h2>GeoAI</h2>'
            '<span class="src">Air-quality intelligence for the whole world</span></div>', unsafe_allow_html=True)
h2.markdown(f'<div style="text-align:right"><span class="live"><span class="dot"></span>LIVE</span>'
            f'<span class="src"> &nbsp;latest model hour {live["times"][ni][11:16]} UTC · loaded {live["fetched"]} · '
            f'{len(live["cities"])} cities</span></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Selection bar: data source · city · pollutant/product · time frame
# ---------------------------------------------------------------------------
st.write("")
bar = st.container(border=True)
with bar:
    b1, b2, b2b, b3, b4, b5 = st.columns([1.5, 1.15, 1.15, 1.15, 1.35, 0.8], vertical_alignment="bottom")
    source = b1.selectbox("🛰️ Data source", [sat.CAMS] + list(sat.SATELLITES), key="src")
    names = sorted(PLACES, key=lambda n: (PLACES[n][2] != "Hungary", n))
    place = b2.selectbox("📍 City", names + ["🔎 Search another place…"], index=names.index("Debrecen"), key="place")
    area_mode = b2b.selectbox("🗺️ Area", ["Whole country", "City surroundings", "Wide region (~2,000 km)"],
                              key="area_mode")
    if source == sat.CAMS:
        product = b3.selectbox("🧪 Pollutant", list(POLLUTANTS), format_func=lambda k: POLLUTANTS[k][0], key="pol")
        min_d, max_d = TODAY - dt.timedelta(days=92), TODAY + dt.timedelta(days=3)
    else:
        product = b3.selectbox("🧪 Product", list(sat.SATELLITES[source]), key=f"prod_{source}")
        min_d, max_d = dt.date(2018, 7, 10), TODAY
    dates = b4.date_input("📅 Time frame", value=(TODAY - dt.timedelta(days=7), min(TODAY, max_d)),
                          min_value=min_d, max_value=max_d, key=f"dates_{source == sat.CAMS}", format="DD.MM.YYYY")
    go = b5.button("▶ Generate", type="primary", width="stretch")

    custom = None
    if place.startswith("🔎"):
        q = st.text_input("Type a city, town or country", placeholder="e.g. Nairobi, Kenya")
        if q:
            g = engine._geocode(q)
            if g:
                custom = g
                st.caption(f"Found: {g[0]} ({g[1]:.2f}, {g[2]:.2f})")
            else:
                st.warning("Place not found. Try adding the country, e.g. 'Kano, Nigeria'.")
    if source != sat.CAMS:
        with st.expander("🔑 Google Earth Engine connection" + (" ✅" if st.session_state["ee_ok"] else ""),
                         expanded=not st.session_state["ee_ok"]):
            st.markdown("Satellite data comes from **Google Earth Engine** (free for research and education). One-time "
                        "setup in the VS Code terminal:\n\n`python -m pip install earthengine-api`\n\n"
                        "`earthengine authenticate`\n\nThen paste your Google Cloud **project ID** (the project you "
                        "use in the Earth Engine Code Editor).")
            pc1, pc2 = st.columns([3, 1], vertical_alignment="bottom")
            proj = pc1.text_input("Earth Engine project ID", value=st.session_state["ee_project"])
            if pc2.button("Connect", width="stretch") or (proj and not st.session_state["ee_ok"]
                                                          and st.session_state.get("_ee_tried") != proj):
                st.session_state["_ee_tried"] = proj
                ok, msg = sat.ee_connect(proj)
                st.session_state["ee_ok"] = ok
                if ok:
                    st.session_state["ee_project"] = proj
                    sat.save_settings({**settings, "ee_project": proj})
                    st.success(msg)
                else:
                    st.error(msg)

if isinstance(dates, (list, tuple)) and len(dates) == 2:
    start, end = dates
else:
    start = end = dates if not isinstance(dates, (list, tuple)) else dates[0]
if source == sat.CAMS and (end - start).days > 92:
    st.warning("CAMS selections are limited to 92 days. Using the last 92 days of your range.")
    start = end - dt.timedelta(days=92)

if place.startswith("🔎"):
    loc = custom
else:
    loc = (place, PLACES[place][0], PLACES[place][1])

if go:
    if not loc:
        st.warning("Choose a city first.")
    elif source != sat.CAMS and not st.session_state["ee_ok"]:
        st.warning("Connect Google Earth Engine first (box above), or choose the CAMS data source.")
    else:
        try:
            if source == sat.CAMS:
                with st.spinner(f"Downloading {POLLUTANTS[product][0]} for {loc[0]} "
                                f"({'whole country' if area_mode == 'Whole country' else 'surroundings'})… "
                                "this can take up to a minute"):
                    d = cams_selection(loc[1], loc[2], product, start, end, area_mode == "Whole country",
                                       area_mode.startswith("Wide"))
                st.session_state["sel"] = {"kind": "cams", "loc": loc, "pol": product, "start": start, "end": end,
                                           **d}
            else:
                with st.spinner(f"Asking Google Earth Engine for {product} around {loc[0]}…"):
                    ctry = None
                    if area_mode == "Whole country":
                        try:
                            ctry = B.country_at(loc[2], loc[1])
                        except Exception:
                            ctry = None
                    if ctry:
                        allr = np.vstack(ctry["outer"])
                        bb = (*allr.min(axis=0), *allr.max(axis=0))
                        sarea = {"name": ctry["name"], "bbox": bb, "outer": ctry["outer"], "regions": []}
                    else:
                        bb = None
                        sarea = {"name": None, "bbox": (loc[2] - 2.5, loc[1] - 2.5, loc[2] + 2.5, loc[1] + 2.5),
                                 "outer": None, "regions": []}
                    layer = sat.ee_layer(source, product, start, end, loc[1], loc[2], bbox=bb)
                    series = sat.ee_series(source, product, start, end, loc[1], loc[2])
                st.session_state["sel"] = {"kind": "ee", "loc": loc, "source": source, "product": product,
                                           "start": start, "end": end, "layer": layer, "series": series,
                                           "area": sarea}
        except Exception as e:
            st.error(f"Could not get the data ({e}).")

sel = st.session_state["sel"]

# ---------------------------------------------------------------------------
# World KPIs
# ---------------------------------------------------------------------------
pm = now_df.dropna(subset=["PM2.5"])
if not pm.empty:
    k = st.columns(5)
    w, b = pm.loc[pm["PM2.5"].idxmax()], pm.loc[pm["PM2.5"].idxmin()]
    hu = pm[pm["Country"] == "Hungary"]["PM2.5"].mean()
    kpi(k[0], "Most polluted city now", w["City"], f"{w['PM2.5']:.0f} µg/m³ PM2.5 · {w['Country']}")
    kpi(k[1], "Cleanest city now", b["City"], f"{b['PM2.5']:.1f} µg/m³ PM2.5 · {b['Country']}")
    kpi(k[2], "Cities above WHO now", f"{int((pm['PM2.5'] > 15).sum())} / {len(pm)}", "PM2.5 above 15 µg/m³")
    kpi(k[3], "Median of all cities", f"{pm['PM2.5'].median():.1f} µg/m³", "PM2.5, latest hour")
    kpi(k[4], "Hungary average now", f"{hu:.1f} µg/m³" if not np.isnan(hu) else "–", "PM2.5, 20 Hungarian cities")
st.write("")

# ---------------------------------------------------------------------------
# Globe | assistant
# ---------------------------------------------------------------------------
g_col, ai_col = st.columns([2.6, 1], gap="medium")
with ai_col:
    box = st.container(border=True, height=680)
    with box:
        st.markdown("**✨ GeoAI assistant**")
        chat_area = st.container(height=380, border=False)
        st.pills("Suggestions", SUGGESTIONS, key="suggest", on_change=on_suggest, label_visibility="collapsed")
        st.chat_input("Ask, analyse or train a model…", key="chat_in", on_submit=on_chat)

if st.session_state["pending"]:
    text = st.session_state["pending"]
    st.session_state["pending"] = None
    try:
        with st.spinner("🔬 Downloading data and running the analysis…"):
            res = engine.run(text)
        st.session_state["result"] = res
        st.session_state["chat"].append(("ai", res["reply"]))
        if res.get("plan"):
            st.session_state["map_pol"] = res["plan"]["pollutant"]
    except Exception as e:
        st.session_state["chat"].append(("ai", f"Sorry, something went wrong while getting the data ({e}). "
                                               "Please try again, or try another place or period."))

with g_col:
    data = dict(live)
    data["pollutants"] = {k_: {"label": v[0], "who": v[1]} for k_, v in POLLUTANTS.items()}
    data["start_pollutant"] = "pm2_5"
    data["surface"] = surface_cached()
    data["focus"] = None
    data["region"] = None
    data["ee"] = None
    if sel:
        data["focus"] = {"lon": sel["loc"][2], "lat": sel["loc"][1], "name": sel["loc"][0]}
        if sel["kind"] == "cams":
            data["start_pollutant"] = sel["pol"]
            data["region"] = {"lats": sel["lats"], "lons": sel["lons"], "d": sel["res"], "pol": sel["pol"],
                              "values": _clean(np.array(sel["grid"], dtype=float).ravel()),
                              "label": f"Mean {sel['start']:%d %b} – {sel['end']:%d %b %Y}"}
        else:
            L = sel["layer"]
            data["ee"] = {"tiles": L["tiles"], "min": L["min"], "max": L["max"], "unit": L["unit"],
                          "label": f"{sel['product']} · {sel['start']:%d %b} – {sel['end']:%d %b %Y}",
                          "palette": sat.PALETTE}
    if st.session_state.get("map_pol") and not (sel and sel["kind"] == "cams"):
        data["start_pollutant"] = st.session_state["map_pol"]
    blob = json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    version = hashlib.md5(blob.encode()).hexdigest()[:10]
    target = HERE / "static" / "globe_data.json"
    if not target.exists() or st.session_state.get("_globe_v") != version:
        target.write_text(blob, encoding="utf-8")
        st.session_state["_globe_v"] = version
    st.iframe(f"{static_base()}/app/static/globe.html?v={version}", height=680)
    c1, c2 = st.columns([3, 1.2])
    c1.markdown('<div class="src">Drag to spin · scroll to zoom · hover a city · <b>click anywhere</b> for that '
                'spot\'s air quality · ▶ plays yesterday → 3-day forecast.</div>', unsafe_allow_html=True)
    if data["surface"] is None:
        if c2.button("🌍 Load world PM2.5 surface", width="stretch",
                     help="Downloads PM2.5 for ~1,400 points around the world (about a minute). Kept for 6 hours."):
            pbar = st.progress(0.0, text="Downloading world surface…")
            try:
                fetch_world_surface(lambda x: pbar.progress(x, text="Downloading world surface…"))
                st.rerun()
            except Exception as e:
                st.error(f"Could not load the surface ({e}).")
    else:
        c2.markdown('<div class="src" style="text-align:right">World surface loaded ✅</div>', unsafe_allow_html=True)

with chat_area:
    if not st.session_state["chat"]:
        st.chat_message("ai", avatar="✨").markdown(
            "Hi! Ask me about air quality **anywhere in the world**: **now & forecast**, **averages**, **trends**, "
            "**comparisons**, the **link with weather**, or **train a machine-learning model**. "
            "Tap a suggestion or type *help*.")
    for role, msg in st.session_state["chat"][-8:]:
        st.chat_message(role, avatar="✨" if role == "ai" else "🧑").markdown(msg)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.write("")
t_sel, t_ml, t_qgis, t_res, t_now, t_help = st.tabs(["📍 Selection report", "🤖 ML Lab", "🖼️ Download map",
                                                     "🔬 Assistant analysis",
                                               "🏙️ World cities now", "📘 Help"])

with t_sel:
    if not sel:
        st.info("Choose a **data source, city, pollutant and time frame** above, then press **▶ Generate**.")
    elif sel["kind"] == "cams":
        pol = sel["pol"]
        label, who, kind = POLLUTANTS[pol]
        s = sel["series"].dropna(subset=[pol])
        st.subheader(f"{label} · {sel['loc'][0]} · {sel['start']:%d %b %Y} → {sel['end']:%d %b %Y}")
        if s.empty:
            st.warning("No data for this selection.")
        else:
            d = s.assign(date=s["time"].dt.date).groupby("date")[pol].agg(["mean", "count"]).reset_index()
            d = d[d["count"] >= 18]
            k = st.columns(4)
            nm, colr = level(s[pol].mean() / who)
            kpi(k[0], f"Mean {label}", f"{s[pol].mean():.1f} µg/m³", f'<span class="pill" style="background:{colr}">{nm}</span>')
            kpi(k[1], "Highest hour", f"{s[pol].max():.1f} µg/m³", f"{s.loc[s[pol].idxmax(), 'time']:%d %b %H:%M} local")
            kpi(k[2], "Days above WHO", f"{int((d['mean'] > who).sum())} / {len(d)}", f"guideline {who:g} µg/m³ ({kind})")
            g = np.array(sel["grid"], dtype=float)
            kpi(k[3], f"Area mean · {sel['area']['name'] or 'surroundings'}"[:40], f"{np.nanmean(g):.1f} µg/m³",
                f"max {np.nanmax(g):.1f} µg/m³ · grid {sel['res']:g}°")
            st.write("")
            ch = alt.Chart(s).mark_line(color="#5fae0e").encode(x=alt.X("time:T", title=None),
                                                                 y=alt.Y(f"{pol}:Q", title=f"{label} (µg/m³)"))
            rule = alt.Chart(pd.DataFrame({"w": [who]})).mark_rule(strokeDash=[4, 4], color="#d7191c").encode(y="w:Q")
            st.altair_chart(ch + rule, height=300, width="stretch")
            st.caption(f"Hourly {label} (local time). Red dashed line = WHO {kind} guideline ({who:g} µg/m³). "
                       "The coloured area around the city is shown on the globe.")
            other = [p for p in POLLUTANTS if p != pol and p in s.columns]
            st.markdown("**Other pollutants in the same period**")
            st.dataframe(pd.DataFrame([{"Pollutant": POLLUTANTS[p][0], "Mean (µg/m³)": sel["series"][p].mean(),
                                        "Max hour": sel["series"][p].max(), "WHO guideline": POLLUTANTS[p][1]}
                                       for p in [pol] + other]).round(1), hide_index=True, width="stretch")
    else:
        se = sel["series"]
        L = sel["layer"]
        st.subheader(f"{sel['product']} · {sel['loc'][0]} · {sel['start']:%d %b %Y} → {sel['end']:%d %b %Y}")
        st.caption(f"{sel['source']} · {L['images']} satellite images in the period")
        if se.empty:
            st.warning("No valid satellite pixels over the city in this period (clouds or no overpass). The map "
                       "on the globe may still show the area.")
        else:
            k = st.columns(4)
            kpi(k[0], "Mean over the city", f"{se['v'].mean():.2f}", L["unit"] + " · 10 km around the centre")
            kpi(k[1], "Highest day", f"{se['v'].max():.2f}", f"{pd.Timestamp(se.loc[se['v'].idxmax(), 'date']):%d %b %Y}")
            kpi(k[2], "Days with data", f"{len(se)}", "cloud-free overpasses")
            kpi(k[3], "Map range (2–98%)", f"{L['min']:.2f} – {L['max']:.2f}", L["unit"])
            st.write("")
            se2 = se.assign(date=pd.to_datetime(se["date"]))
            st.altair_chart(alt.Chart(se2).mark_line(point=True, color="#2c7bb6").encode(
                x=alt.X("date:T", title=None), y=alt.Y("v:Q", title=f"{sel['product']} ({L['unit']})")),
                height=300, width="stretch")
            st.caption("Daily mean over a 10 km circle around the city centre. Satellite columns measure the whole "
                       "air column, not ground-level concentration.")

@st.cache_data(ttl=3600, show_spinner=False)
def weather_for(lat, lon, start, end):
    return mllab.get_weather(lat, lon, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def air_for(lat, lon, start, end):
    p = {"latitude": lat, "longitude": lon, "hourly": ",".join(POLLUTANTS), "start_date": str(start),
         "end_date": str(end), "timezone": "auto"}
    r = requests.get(engine.AQ_URL, params=p, timeout=90)
    r.raise_for_status()
    a = pd.DataFrame(r.json()["hourly"])
    a["time"] = pd.to_datetime(a["time"])
    return a


with t_ml:
    st.markdown("### 🤖 ML Lab")
    st.markdown("Model the selected pollutant at the selected city from **weather** (and time of day). "
                "Press a button to run a method, or run all of them and compare. Everything can be downloaded "
                "as a **Word report**. The ML Lab uses its **own, longer training period** (below), because "
                "machine learning needs many weeks of data.")
    if not sel or sel["kind"] != "cams":
        st.info("Choose the **CAMS** data source, a city, a pollutant and a time frame above, then press **▶ Generate**. "
                "Tip: use at least 30 days for machine learning.")
    else:
        pol = sel["pol"]
        label, who, _ = POLLUTANTS[pol]
        o1, o2, o3 = st.columns([1.4, 1, 1])
        ml_days = o1.selectbox("Training data", [92, 60, 30], format_func=lambda d: f"Last {d} days "
                               f"({d * 24:,} hours)" + (" · recommended" if d == 92 else ""))
        split_how = o1.radio("Test data", ["Time-based (last 25% of hours)", "Random 25% of hours"],
                             help="Time-based is the honest test: the model must predict a later period.")
        use_time = o2.checkbox("Use hour of day and day of week", value=True)
        lags = o3.checkbox("Use earlier pollution values (lags)", value=False,
                           help="Adds the value 1 h and 24 h before. Prediction gets much easier, but it mostly "
                                "shows persistence, not causes.")
        depth = o2.slider("Decision tree depth", 2, 8, 4)
        bcols = st.columns(5)
        clicked = []
        for col, (lab_, name) in zip(bcols, [("📈 MLR", "Multiple linear regression"), ("🌳 Decision tree", "Decision tree"),
                                             ("🌲 Random forest", "Random forest"), ("⚡ XGBoost", "XGBoost")]):
            if col.button(lab_, width="stretch"):
                clicked = [name]
        if bcols[4].button("▶ Run all & compare", type="primary", width="stretch"):
            clicked = list(mllab.MODELS)
        cfg = (id(sel), split_how, use_time, lags, depth, ml_days)
        if st.session_state.get("_ml_cfg") != cfg:
            st.session_state["_ml_cfg"] = cfg
            st.session_state["ml"] = {}
        if clicked:
            try:
                ml_end = TODAY - dt.timedelta(days=1)
                ml_start = ml_end - dt.timedelta(days=ml_days - 1)
                with st.spinner(f"Downloading {ml_days} days of air-quality and weather data…"):
                    aq = air_for(sel["loc"][1], sel["loc"][2], ml_start, ml_end)
                    wx = weather_for(sel["loc"][1], sel["loc"][2], ml_start, ml_end)
                d, feats = mllab.build_table(aq, wx, pol, use_time, lags)
                st.session_state["_ml_period"] = (ml_start, ml_end)
                if len(d) < 300:
                    st.warning(f"Only {len(d)} complete hours were available. Choose a longer training period.")
                else:
                    tr, te = mllab.split(d, split_how)
                    st.session_state["_ml_data"] = (d, feats)
                    for name in clicked:
                        with st.spinner(f"Training {name}…"):
                            try:
                                st.session_state["ml"][name] = mllab.run_model(name, tr, te, feats, pol,
                                                                               {"max_depth": depth})
                            except RuntimeError as e:
                                st.warning(str(e))
            except Exception as e:
                st.error(f"Could not run the analysis ({e}).")
        results = {n: st.session_state["ml"][n] for n in mllab.MODELS if n in st.session_state.get("ml", {})}
        if results:
            d, feats = st.session_state["_ml_data"]
            comp = pd.DataFrame([{"Model": n, "Train R²": r["train"]["R²"], "Test R²": r["test"]["R²"],
                                  "Test RMSE (µg/m³)": r["test"]["RMSE"], "Test MAE (µg/m³)": r["test"]["MAE"]}
                                 for n, r in results.items()]).round(3)
            st.markdown("#### Comparison")
            st.dataframe(comp, hide_index=True, width="stretch")
            if len(results) > 1:
                st.image(mllab.fig_compare(results), width="stretch")
            for n, r in results.items():
                with st.expander(f"{n} · test R² = {r['test']['R²']:.2f}", expanded=len(results) == 1):
                    k = st.columns(4)
                    kpi(k[0], "Test R²", f"{r['test']['R²']:.2f}", f"train {r['train']['R²']:.2f}")
                    kpi(k[1], "RMSE", f"{r['test']['RMSE']:.1f} µg/m³")
                    kpi(k[2], "MAE", f"{r['test']['MAE']:.1f} µg/m³")
                    kpi(k[3], "Hours", f"{r['n_train']} / {r['n_test']}", "train / test")
                    st.write("")
                    st.markdown(mllab.describe(r, label))
                    if n == "Multiple linear regression":
                        c = r["extra"]["coefficients"].copy()
                        c["Variable"] = ["(Intercept)"] + [mllab.nice(v, label) for v in c["Variable"].iloc[1:]]
                        st.dataframe(c.round(4), hide_index=True, width="stretch")
                    g1, g2 = st.columns(2)
                    g1.image(mllab.fig_scatter(r, label), width="stretch")
                    g2.image(mllab.fig_importance(r, label), width="stretch")
                    st.image(mllab.fig_timeseries(r, label), width="stretch")
                    if n == "Decision tree":
                        st.image(mllab.fig_tree(r, feats, label), width="stretch")
                        st.code(r["extra"]["rules"], language=None)
            st.markdown("#### Downloads")
            dl = st.columns(3)
            place = sel["loc"][0]
            try:
                docx_bytes = mllab.word_report(results=results, d=d, feats=feats, pol=pol, label=label, who=who,
                                               place=place, start=st.session_state["_ml_period"][0],
                                               end=st.session_state["_ml_period"][1], split_how=split_how,
                                               lags=lags, sources=SOURCE)
                dl[0].download_button("⬇️ Word report (.docx)", docx_bytes,
                                      f"GeoAI_ML_{label.replace('.', '')}_{place.split(',')[0]}_{sel['end']}.docx",
                                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                      type="primary", width="stretch")
            except Exception as e:
                dl[0].error(f"Word report failed ({e}). Run: python -m pip install python-docx")
            preds = None
            for n, r in results.items():
                p = r["pred"].rename(columns={"Predicted": n})
                preds = p if preds is None else preds.merge(p[["time", n]], on="time", how="outer")
            dl[1].download_button("⬇️ Predictions (CSV)", preds.to_csv(index=False), "GeoAI_ML_predictions.csv",
                                  "text/csv", width="stretch")
            dl[2].download_button("⬇️ Model data (CSV)", d.to_csv(index=False), "GeoAI_ML_data.csv", "text/csv",
                                  width="stretch")


with t_qgis:
    st.markdown("### 🖼️ Ready-made map")
    st.markdown("A finished, print-ready map of your selection (A4 landscape) with title, legend, scale bar, "
                "north arrow, key numbers and data credits. Download it and use it straight away in a report, "
                "presentation or poster.")
    if not sel:
        st.info("Choose a data source, city, pollutant and time frame above, then press **▶ Generate**.")
    else:
        has_regions = (sel["kind"] == "cams" and bool(sel.get("area", {}).get("regions"))
                       and bool(sel["area"].get("outer")))
        m0, m0b, m1, m2, m3 = st.columns([1.3, 1.2, 1.3, 0.9, 1], vertical_alignment="bottom")
        map_type = m0.selectbox("Map type", ["Surface + city circles", "Smooth surface"]
                                + (["By region (counties / states)"] if has_regions else []))
        classes = m0b.selectbox("Colours", ["Local contrast", "WHO classes"],
                                help="Local contrast shows differences inside the area. WHO classes use fixed "
                                     "steps (1×, 2×, 3×, 5×, 8× the WHO guideline) so maps can be compared.")
        basemap = m1.selectbox("Background", list(mapmaker.BASEMAPS))
        fmt = m2.selectbox("Format", ["PNG image", "PDF document"])
        make = m3.button("🖼️ Create map", type="primary", width="stretch")
        key = (id(sel), map_type, classes, basemap, fmt)
        if make:
            try:
                with st.spinner("Drawing your map…"):
                    f = "pdf" if fmt.startswith("PDF") else "png"
                    lname = sel["loc"][0]
                    period = f"{sel['start']:%d %b %Y} – {sel['end']:%d %b %Y}"
                    if sel["kind"] == "cams":
                        lab, who, kind_ = POLLUTANTS[sel["pol"]]
                        s_ = sel["series"].dropna(subset=[sel["pol"]])
                        dd = s_.assign(date=s_["time"].dt.date).groupby("date")[sel["pol"]].agg(["mean", "count"])
                        dd = dd[dd["count"] >= 18]
                        g_ = np.array(sel["grid"], dtype=float)
                        ar = sel["area"]
                        where = ar["name"] if ar.get("outer") else lname.split(",")[0] + (
                            " and the wider region" if sel["res"] >= 0.75 else " and surroundings")
                        country_pl = [(n, PLACES[n][1], PLACES[n][0], False) for n in PLACES]
                        out, zst = mapmaker.make_area_map(
                            area=ar, grid={"lats": sel["lats"], "lons": sel["lons"], "values": g_},
                            title=f"{lab} · {where}",
                            subtitle=f"Mean {period} · CAMS air-quality model (Copernicus)"
                                     + (" · by region" if map_type.startswith("By region") else ""),
                            legend_label=f"{lab} (µg/m³)",
                            credits=f"Data: CAMS (Copernicus) via Open-Meteo · WHO 2021 {kind_} guideline {who:g} µg/m³ "
                                    "· Boundaries: Natural Earth, geoBoundaries",
                            who=who, basemap=basemap, fmt=f,
                            mode="regions" if map_type.startswith("By region") else
                                 ("bubbles" if map_type.startswith("Surface +") else "surface"),
                            classes="who" if classes == "WHO classes" else "quantile",
                            places=country_pl + [(lname.split(",")[0], sel["loc"][2], sel["loc"][1], True)],
                            contains=B.contains, label_point=B.label_point,
                            stats=[(f"Mean at {lname.split(',')[0]}", f"{s_[sel['pol']].mean():.1f} µg/m³"),
                                   ("Highest hour there", f"{s_[sel['pol']].max():.1f} µg/m³"),
                                   ("Days above WHO there", f"{int((dd['mean'] > who).sum())} / {len(dd)}"),
                                   ("Area mean / max", f"{np.nanmean(g_):.1f} / {np.nanmax(g_):.1f}")])
                        fname = f"GeoAI_{lab.replace('.', '')}_{(ar['name'] or lname).split(',')[0].replace(' ', '_')}_{sel['end']}"
                        st.session_state["_zonal"] = (id(sel), zst)
                    else:
                        L = sel["layer"]
                        se = sel["series"]
                        ar = sel["area"]
                        bx0, by0, bx1, by1 = L["bbox"]
                        ar = {**ar, "bbox": (bx0, by0, bx1, by1)}
                        out, _ = mapmaker.make_area_map(
                            area=ar, grid=None, image=sat.ee_thumbnail(L), image_range=(L["min"], L["max"]),
                            palette=sat.PALETTE, who=None,
                            title=f"{sel['product']} · {ar['name'] or lname.split(',')[0] + ' and surroundings'}",
                            subtitle=f"Mean {period} · {sel['source'].split(' (')[0]}",
                            legend_label=f"{sel['product']} ({L['unit']})",
                            credits="Data: Copernicus Sentinel-5P / NASA MODIS via Google Earth Engine · Boundaries: Natural Earth",
                            basemap=basemap, fmt=f, contains=B.contains, label_point=B.label_point,
                            places=[(n, PLACES[n][1], PLACES[n][0], False) for n in PLACES]
                                   + [(lname.split(",")[0], sel["loc"][2], sel["loc"][1], True)],
                            stats=([("Mean over the city", f"{se['v'].mean():.3g} {L['unit']}"),
                                    ("Highest day", f"{se['v'].max():.3g} {L['unit']}"),
                                    ("Days with data", f"{len(se)}")] if not se.empty else []) +
                                  [("Satellite images", f"{L['images']}")])
                        fname = f"GeoAI_{sel['product'].split(' ')[0]}_{lname.split(',')[0]}_{sel['end']}"
                st.session_state["_map"] = (key, out, f, fname)
            except Exception as e:
                st.error(f"Could not draw the map ({e}).")
        mp = st.session_state.get("_map")
        if mp and mp[0] == key:
            _, out, f, fname = mp
            if f == "png":
                st.image(out, width="stretch")
            else:
                st.success("Your PDF map is ready.")
            dcols = st.columns(3)
            dcols[0].download_button(f"⬇️ Download map ({f.upper()})", out, f"{fname}.{f}",
                                     "application/pdf" if f == "pdf" else "image/png", type="primary", width="stretch")
            zs = st.session_state.get("_zonal")
            if sel["kind"] == "cams" and zs and zs[0] == id(sel) and zs[1]:
                ztab = pd.DataFrame([{"Region": k_, "Mean": v["mean"], "Min": v["min"], "Max": v["max"],
                                      "Class": v.get("class", "")} for k_, v in zs[1].items()]
                                    ).sort_values("Mean", ascending=False).round(2)
                dcols[1].download_button("⬇️ Regions as shapefile (.zip)",
                                         B.shapefile_zip(sel["area"]["regions"], zs[1], f"{fname}_regions"),
                                         f"{fname}_regions_shapefile.zip", "application/zip", width="stretch")
                dcols[2].download_button("⬇️ Regions table (CSV)", ztab.to_csv(index=False), f"{fname}_regions.csv",
                                         "text/csv", width="stretch")
                with st.expander(f"Values by region ({len(ztab)})"):
                    st.dataframe(ztab, hide_index=True, width="stretch")

    st.write("")
    with st.expander("🧭 For GIS users: QGIS package (editable layers)"):
        st.markdown("Unzip, then drag the **`.asc`** (map grid), **`.tif`** (satellite) or **`.geojson`** (cities) "
                    "file into QGIS. Don't drag the `.qml` file: it is the colour style, and QGIS applies it "
                    "automatically when it sits next to the map file with the same name.")
        if not sel:
            st.caption("Generate a selection first to add its map layer. The world-cities layer is always included.")
        files = {}
        readme = ["GeoAI QGIS package", "=" * 18, "",
                  "How to open: start QGIS (free, qgis.org) and drag the .asc / .tif / .geojson files into the map window.", "Do not drag the .qml files: they are colour styles and load automatically.",
                  "Background map: Browser panel -> XYZ Tiles -> OpenStreetMap (double-click).", ""]
        cities_rows = [{"lat": c["lat"], "lon": c["lon"], "city": c["name"], "country": c["country"],
                        **{POLLUTANTS[p][0]: c["v"][p][ni] for p in POLLUTANTS}} for c in live["cities"]]
        files["world_cities_latest.geojson"] = sat.geojson_points(cities_rows)
        readme.append(f"world_cities_latest.geojson - {len(cities_rows)} cities, latest model hour "
                      f"{live['times'][ni].replace('T', ' ')} UTC (µg/m³).")
        if sel and sel["kind"] == "cams":
            pol = sel["pol"]
            label, who, _ = POLLUTANTS[pol]
            base = f"{sel['loc'][0].split(',')[0].replace(' ', '_')}_{label.replace('.', '').replace('₂', '2').replace('₃', '3')}_mean"
            files[f"{base}.asc"] = sat.asc_grid(sel["lats"], sel["lons"], sel["grid"])
            files[f"{base}.prj"] = sat.WGS84_PRJ
            stops = [(round(who * r, 2), c, f"{who * r:g} µg/m³" + (" (WHO)" if r == 1 else ""))
                     for r, c in [(0, "#2f9e6f"), (1, "#7bd88f"), (2, "#f5d547"), (3, "#ff9f43"), (5, "#ff5c5c"), (8, "#b36bff")]]
            files[f"{base}.qml"] = sat.qml_raster(stops)
            files[f"{base}_hourly.csv"] = sel["series"].to_csv(index=False)
            readme += [f"{base}.asc - mean {label} {sel['start']} to {sel['end']}, 0.25° grid around {sel['loc'][0]} (µg/m³),",
                       f"   CAMS model via Open-Meteo. Style {base}.qml is applied automatically.",
                       f"{base}_hourly.csv - hourly values of all pollutants at the city (local time)."]
        elif sel and sel["kind"] == "ee":
            L = sel["layer"]
            base = f"{sel['loc'][0].split(',')[0].replace(' ', '_')}_{sel['product'].split(' ')[0].replace('₂', '2').replace('₃', '3')}"
            if st.button("🛰️ Prepare satellite GeoTIFF (from Earth Engine)"):
                try:
                    with st.spinner("Downloading GeoTIFF from Earth Engine…"):
                        st.session_state["_tif"] = (base, sat.ee_geotiff(L))
                except Exception as e:
                    st.error(f"Could not download the GeoTIFF ({e}). Try a shorter period or smaller area.")
            if st.session_state.get("_tif") and st.session_state["_tif"][0] == base:
                files[f"{base}.tif"] = st.session_state["_tif"][1]
                n = len(sat.PALETTE)
                stops = [(round(L["min"] + (L["max"] - L["min"]) * i / (n - 1), 4), c,
                          f"{L['min'] + (L['max'] - L['min']) * i / (n - 1):.3g}") for i, c in enumerate(sat.PALETTE)]
                files[f"{base}.qml"] = sat.qml_raster(stops)
                readme.append(f"{base}.tif - mean {sel['product']} ({L['unit']}), {sel['start']} to {sel['end']}, "
                              f"{sel['source']}, 2 km pixels.")
            if not sel["series"].empty:
                files[f"{base}_daily.csv"] = sel["series"].to_csv(index=False)
        readme += ["", SOURCE]
        pack = sat.qgis_pack(files, "\n".join(readme))
        st.download_button(f"⬇️ Download QGIS package ({len(files)} files)", pack, "GeoAI_QGIS_package.zip",
                           "application/zip", type="primary")
        st.caption("Files: " + ", ".join(files))


with t_res:
    res = st.session_state["result"]
    if not res or not res["blocks"]:
        st.info("Results of what you type in the assistant appear here, e.g. "
                "*Train a random forest to predict PM2.5 in Delhi last 92 days with lags*.")
    else:
        st.subheader(res["title"])
        parts = []
        for kind_, content in res["blocks"]:
            if kind_ == "h":
                st.markdown(f"#### {content}")
                parts.append(f"<h2>{html.escape(content)}</h2>")
            elif kind_ == "text":
                st.markdown(content)
                parts.append(f"<p>{html.escape(content.replace('**', '').replace('_', ''))}</p>")
            elif kind_ == "table":
                st.dataframe(content, hide_index=True, width="stretch")
                parts.append(content.to_html(index=False, border=0))
            elif kind_ == "chart":
                ch, cap = content
                st.altair_chart(ch, height=300, width="stretch")
                st.caption(cap)
            elif kind_ == "metrics":
                cols = st.columns(len(content))
                for col, (lab, val) in zip(cols, content):
                    kpi(col, lab, val)
                st.write("")
                parts.append("<p>" + " · ".join(f"<b>{html.escape(l)}</b>: {html.escape(v)}" for l, v in content) + "</p>")
            elif kind_ == "files":
                cols = st.columns(len(content) + 1)
                for col, (lab, dat, fn, mime) in zip(cols, content):
                    col.download_button(lab, dat, fn, mime, width="stretch")
        report = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>GeoAI analysis</title><style>"
                  "body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;color:#222}h1{color:#3f7d0b}"
                  "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px}"
                  "th{background:#f1f7e8}.s{color:#666;font-size:12px}</style></head><body>"
                  f"<h1>GeoAI · {html.escape(res['title'])}</h1><p class='s'>Generated "
                  f"{pd.Timestamp.now():%Y-%m-%d %H:%M}</p>{''.join(parts)}<p class='s'>{SOURCE}</p></body></html>")
        st.download_button("⬇️ Download this analysis as a report (HTML)", report, "GeoAI_analysis.html",
                           "text/html", type="primary")

with t_now:
    st.caption(f"Latest model hour: {live['times'][ni].replace('T', ' ')} UTC. Click a column to sort.")
    st.dataframe(now_df.sort_values("PM2.5", ascending=False).round(1), hide_index=True, width="stretch", height=420)

with t_help:
    st.markdown(engine.HELP)
    st.markdown("""**Data sources**
- **CAMS model (default, no setup):** hourly PM2.5, PM10, NO₂ and O₃ near the ground, last 92 days plus 3-day forecast.
- **Sentinel-5P TROPOMI:** satellite columns of NO₂, CO, SO₂, HCHO, O₃ and the UV aerosol index (from July 2018).
- **MODIS MAIAC:** aerosol optical depth at 550 nm.
Satellite sources need a free Google Earth Engine account (see the 🔑 box that appears when you choose one).""")

st.markdown(f'<div class="src" style="margin-top:16px">{SOURCE}</div>', unsafe_allow_html=True)
