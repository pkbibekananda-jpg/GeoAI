"""
Country and region (county / state / province) boundaries.

- Countries: Natural Earth 1:50m (public domain)
- Regions (admin level 1): geoBoundaries (gbOpen, open licences)
Downloaded once and kept in the ./cache folder.
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import requests
from matplotlib.path import Path as MPath

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
GB_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso}/ADM1/"


def _rings(geom):
    """All polygon rings (outer + holes) of a GeoJSON geometry as numpy arrays (lon, lat)."""
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        polys = geom["coordinates"]
    else:
        return []
    return [np.asarray(r, dtype=float)[:, :2] for p in polys for r in p]


def _outer_rings(geom):
    if geom["type"] == "Polygon":
        return [np.asarray(geom["coordinates"][0], dtype=float)[:, :2]]
    if geom["type"] == "MultiPolygon":
        return [np.asarray(p[0], dtype=float)[:, :2] for p in geom["coordinates"]]
    return []


def contains(outer_rings, lon, lat, radius=0.0):
    """Boolean mask: which points (arrays) are inside any of the rings."""
    pts = np.column_stack([np.ravel(lon), np.ravel(lat)])
    inside = np.zeros(len(pts), dtype=bool)
    for r in outer_rings:
        x0, y0 = r.min(axis=0) - radius
        x1, y1 = r.max(axis=0) + radius
        cand = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1) & ~inside
        if cand.any():
            inside[cand] |= MPath(r).contains_points(pts[cand], radius=radius)
            if radius:
                inside[cand] |= MPath(r).contains_points(pts[cand], radius=-radius)
    return inside.reshape(np.shape(lon))


def _load_countries():
    f = CACHE / "ne_50m_countries.geojson"
    if not f.exists():
        r = requests.get(NE_URL, timeout=120)
        r.raise_for_status()
        f.write_bytes(r.content)
    return json.loads(f.read_text(encoding="utf-8"))


def country_at(lon, lat):
    """Country containing a point: dict(name, iso3, rings, outer) or None."""
    for feat in _load_countries()["features"]:
        outer = _outer_rings(feat["geometry"])
        if contains(outer, np.array([lon]), np.array([lat]))[0]:
            p = feat["properties"]
            iso3 = p.get("ISO_A3") if p.get("ISO_A3") not in (None, "-99") else p.get("ADM0_A3")
            return {"name": p.get("NAME_LONG") or p.get("NAME") or p.get("ADMIN"), "iso3": iso3,
                    "rings": _rings(feat["geometry"]), "outer": outer, "geometry": feat["geometry"]}
    return None


def borders_in(bbox):
    """Outer rings of all countries touching a lon/lat box (for drawing borders)."""
    x0, y0, x1, y1 = bbox
    out = []
    for feat in _load_countries()["features"]:
        for r in _outer_rings(feat["geometry"]):
            if r[:, 0].max() > x0 and r[:, 0].min() < x1 and r[:, 1].max() > y0 and r[:, 1].min() < y1:
                out.append(r)
    return out


def regions(iso3):
    """Admin-1 regions of a country: list of dict(name, rings, outer, geometry). Empty list if unavailable."""
    f = CACHE / f"adm1_{iso3}.geojson"
    if not f.exists():
        try:
            meta = requests.get(GB_API.format(iso=iso3), timeout=30).json()
            url = meta.get("simplifiedGeometryGeoJSON") or meta.get("gjDownloadURL")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            json.loads(r.content)  # make sure it is real GeoJSON
            f.write_bytes(r.content)
        except Exception:
            return []
    d = json.loads(f.read_text(encoding="utf-8"))
    out = []
    for feat in d.get("features", []):
        g = feat.get("geometry")
        if not g:
            continue
        name = feat["properties"].get("shapeName") or feat["properties"].get("name") or "?"
        out.append({"name": name, "rings": _rings(g), "outer": _outer_rings(g), "geometry": g})
    return out


def label_point(outer_rings):
    """A point to put a label on: centre of the largest ring (moved inside if needed)."""
    big = max(outer_rings, key=lambda r: np.ptp(r[:, 0]) * np.ptp(r[:, 1]))
    c = big.mean(axis=0)
    if MPath(big).contains_point(c):
        return c
    xs = np.linspace(big[:, 0].min(), big[:, 0].max(), 25)
    ys = np.linspace(big[:, 1].min(), big[:, 1].max(), 25)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    ok = MPath(big).contains_points(pts)
    if ok.any():
        cand = pts[ok]
        return cand[np.argmin(((cand - c) ** 2).sum(axis=1))]
    return c


def shapefile_zip(regs, stats, base):
    """Zipped ESRI shapefile of regions with their statistics (opens in QGIS / ArcGIS)."""
    import shapefile  # pyshp
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    w = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYGON)
    w.field("NAME", "C", size=80)
    w.field("MEAN", "N", decimal=2)
    w.field("MIN", "N", decimal=2)
    w.field("MAX", "N", decimal=2)
    w.field("CLASS", "C", size=40)
    for reg in regs:
        s = stats.get(reg["name"])
        if not s:
            continue
        parts = []
        for r in reg["rings"]:
            ring = [list(map(float, p)) for p in r]
            parts.append(ring)
        w.poly(parts)
        w.record(reg["name"][:80], s["mean"], s["min"], s["max"], s.get("class", "")[:40])
    w.close()
    from satellite import WGS84_PRJ
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{base}.shp", shp.getvalue())
        z.writestr(f"{base}.shx", shx.getvalue())
        z.writestr(f"{base}.dbf", dbf.getvalue())
        z.writestr(f"{base}.prj", WGS84_PRJ)
        z.writestr(f"{base}.cpg", "UTF-8")
    return buf.getvalue()
