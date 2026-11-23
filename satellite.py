"""
Satellite data through Google Earth Engine (optional), plus QGIS export helpers.

Earth Engine needs a one-time setup on your computer:
    python -m pip install earthengine-api
    earthengine authenticate
and a Google Cloud project ID that is registered for Earth Engine.
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SETTINGS = Path(__file__).parent / ".streamlit" / "geoai_settings.json"

CAMS = "CAMS model (Copernicus) · no setup"
# source -> product -> (collection, band, scale to display units, unit, default palette range)
SATELLITES = {
    "Sentinel-5P TROPOMI (Earth Engine)": {
        "NO₂ tropospheric column": ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density", 1e6, "µmol/m²"),
        "CO total column": ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density", 1e3, "mmol/m²"),
        "SO₂ total column": ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density", 1e6, "µmol/m²"),
        "HCHO tropospheric column": ("COPERNICUS/S5P/OFFL/L3_HCHO", "tropospheric_HCHO_column_number_density", 1e6, "µmol/m²"),
        "O₃ total column": ("COPERNICUS/S5P/OFFL/L3_O3", "O3_column_number_density", 1e3, "mmol/m²"),
        "UV aerosol index": ("COPERNICUS/S5P/OFFL/L3_AER_AI", "absorbing_aerosol_index", 1, "index"),
    },
    "MODIS MAIAC aerosol (Earth Engine)": {
        "Aerosol optical depth 550 nm": ("MODIS/061/MCD19A2_GRANULES", "Optical_Depth_055", 0.001, "AOD"),
    },
}
PALETTE = ["#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#f29e2e", "#e76818", "#d7191c"]


# ---------------------------------------------------------------------------
# Settings (Earth Engine project ID)
# ---------------------------------------------------------------------------
def load_settings():
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        return {}


def save_settings(d):
    SETTINGS.parent.mkdir(exist_ok=True)
    SETTINGS.write_text(json.dumps(d, indent=2))


# ---------------------------------------------------------------------------
# Earth Engine
# ---------------------------------------------------------------------------
def ee_connect(project):
    """Returns (ok, message)."""
    try:
        import ee
    except ImportError:
        return False, "The Earth Engine library is not installed. In the VS Code terminal run: " \
                      "`python -m pip install earthengine-api`, then `earthengine authenticate`."
    try:
        ee.Initialize(project=project)
        ee.Number(1).getInfo()
        return True, "Connected to Google Earth Engine."
    except Exception as e:
        return False, (f"Could not connect to Earth Engine ({e}). In the terminal run `earthengine authenticate`, "
                       "check the project ID, then try again.")


def _collection(source, product, start, end, region):
    import ee
    coll_id, band, scale, unit = SATELLITES[source][product]
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    c = ee.ImageCollection(coll_id).filterDate(str(start), end_excl).filterBounds(region).select(band)
    return c, band, scale, unit


def ee_layer(source, product, start, end, lat, lon, half_deg=2.5, bbox=None):
    """Mean image for the period around a place (or a bbox): map tiles, value range, image (for download)."""
    import ee
    box = bbox or (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)
    region = ee.Geometry.Rectangle(list(box))
    c, band, scale, unit = _collection(source, product, start, end, region)
    n = c.size().getInfo()
    if n == 0:
        raise ValueError("No satellite images for this place and period. Try a longer period, or dates at least "
                         "a few days ago (processed data arrives with a delay).")
    img = c.mean().multiply(scale).rename("v").clip(region)
    stats = img.reduceRegion(ee.Reducer.percentile([2, 98]), region, scale=5000, bestEffort=True).getInfo()
    lo, hi = stats.get("v_p2"), stats.get("v_p98")
    if lo is None or hi is None or hi <= lo:
        lo, hi = (lo or 0), (lo or 0) + 1
    tiles = img.getMapId({"min": lo, "max": hi, "palette": PALETTE})["tile_fetcher"].url_format
    return {"img": img, "region": region, "bbox": box, "tiles": tiles, "min": lo, "max": hi, "unit": unit, "images": n,
            "label": f"{product} ({unit})"}


def ee_series(source, product, start, end, lat, lon, radius_m=10000):
    import ee
    pt = ee.Geometry.Point([lon, lat]).buffer(radius_m)
    c, band, scale, unit = _collection(source, product, start, end, pt)

    def one(im):
        v = im.reduceRegion(ee.Reducer.mean(), pt, scale=1113.2, bestEffort=True).get(band)
        return ee.Feature(None, {"t": im.date().format("YYYY-MM-dd"), "v": v})

    feats = c.map(one).filter(ee.Filter.notNull(["v"])).limit(5000).getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in feats])
    if df.empty:
        return df
    df["v"] = df["v"].astype(float) * scale
    return df.groupby("t", as_index=False)["v"].mean().rename(columns={"t": "date"})


def ee_thumbnail(layer, size=1200):
    """Coloured PNG of the layer in web-mercator (for finished maps)."""
    url = layer["img"].getThumbURL({"region": layer["region"], "dimensions": size, "crs": "EPSG:3857",
                                    "format": "png", "min": layer["min"], "max": layer["max"], "palette": PALETTE})
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return r.content


def ee_geotiff(layer, scale_m=2000):
    url = layer["img"].getDownloadURL({"region": layer["region"], "scale": scale_m, "format": "GEO_TIFF",
                                       "crs": "EPSG:4326"})
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# QGIS export
# ---------------------------------------------------------------------------
WGS84_PRJ = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
             'UNIT["degree",0.0174532925199433]]')


def asc_grid(lats, lons, values):
    """ESRI ASCII grid (opens directly in QGIS). lats/lons = cell centres (ascending), values[lat][lon]."""
    lats, lons = np.asarray(lats), np.asarray(lons)
    cell = float(lons[1] - lons[0])
    v = np.asarray(values, dtype=float)[::-1]  # north row first
    v = np.where(np.isnan(v), -9999, v)
    head = (f"ncols {len(lons)}\nnrows {len(lats)}\nxllcorner {lons.min() - cell / 2:.6f}\n"
            f"yllcorner {lats.min() - cell / 2:.6f}\ncellsize {cell:.6f}\nNODATA_value -9999\n")
    return head + "\n".join(" ".join(f"{x:.2f}" for x in row) for row in v) + "\n"


def qml_raster(stops, opacity=0.8):
    """QGIS style: stops = [(value, hex colour, label)]. Applied automatically when it has the layer's name."""
    items = "\n".join(f'          <item alpha="255" value="{v}" label="{lab}" color="{c}"/>' for v, c, lab in stops)
    lo, hi = stops[0][0], stops[-1][0]
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="{opacity}" alphaBand="-1"
                    classificationMin="{lo}" classificationMax="{hi}" nodataColor="">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0" minimumValue="{lo}" maximumValue="{hi}">
{items}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""


def geojson_points(rows):
    """rows: list of dicts with lat, lon and properties."""
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
              "properties": {k: v for k, v in r.items() if k not in ("lat", "lon")}} for r in rows]
    return json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))


def qgis_pack(files: dict, readme: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("README.txt", readme)
    return buf.getvalue()
