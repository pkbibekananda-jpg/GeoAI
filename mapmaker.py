"""
Finished, print-ready maps (PNG / PDF) — no GIS skills needed.

Each map has: background map, coloured data layer, city marker, title, legend,
scale bar, north arrow, key numbers and data credits.
"""

import io
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

R = 6378137.0
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/"
BASEMAPS = {
    "Light (clean, for printing)": (ESRI + "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                                    ESRI + "Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                                    "Basemap: Esri, HERE, Garmin, © OpenStreetMap contributors"),
    "Satellite imagery": (ESRI + "World_Imagery/MapServer/tile/{z}/{y}/{x}",
                          ESRI + "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                          "Imagery: Esri, Maxar, Earthstar Geographics, GIS User Community"),
}


def merc(lon, lat):
    lat = np.clip(lat, -85, 85)
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def _stitch(url, lon0, lat0, lon1, lat1, z):
    x0, y0 = _tile_xy(lon0, lat1, z)
    x1, y1 = _tile_xy(lon1, lat0, z)
    tx0, tx1, ty0, ty1 = int(x0), int(x1), int(y0), int(y1)
    img = Image.new("RGBA", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (240, 243, 246, 255))
    from concurrent.futures import ThreadPoolExecutor

    def get(t):
        try:
            r = requests.get(url.format(z=z, x=t[0], y=t[1]), timeout=10)
            r.raise_for_status()
            return t, Image.open(io.BytesIO(r.content)).convert("RGBA")
        except Exception:
            return t, None

    jobs = [(tx, ty) for tx in range(tx0, tx1 + 1) for ty in range(ty0, ty1 + 1)]
    ok = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for (tx, ty), tile in pool.map(get, jobs):
            if tile is not None:
                img.paste(tile, ((tx - tx0) * 256, (ty - ty0) * 256))
                ok += 1
    n = 2 ** z
    # extent of the stitched image in web-mercator metres
    def tile_lon(x): return x / n * 360 - 180
    def tile_lat(y): return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    ex0, ey1 = merc(tile_lon(tx0), tile_lat(ty0))
    ex1, ey0 = merc(tile_lon(tx1 + 1), tile_lat(ty1 + 1))
    return img, (ex0, ex1, ey0, ey1), ok


def _zoom_for(width_deg):
    return int(max(3, min(11, round(math.log2(360 / width_deg * 1100 / 256)))))


def _nice_km(target):
    for v in [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]:
        if v >= target:
            return v
    return 1000


def make_map(*, lon, lat, place, title, subtitle, legend_label, credits, basemap="Light (clean, for printing)",
             grid=None, stops=None, image=None, image_range=None, palette=None, stats=None, fmt="png",
             half_deg=2.5):
    """grid = dict(lats, lons, values, d) in µg/m³ with stops [(value, colour)];  or
       image = PNG bytes in EPSG:3857 covering the box, with image_range=(min,max), palette=[colours]."""
    lon0, lon1, lat0, lat1 = lon - half_deg, lon + half_deg, lat - half_deg, lat + half_deg
    fig = plt.figure(figsize=(11.69, 8.27), dpi=200)  # A4 landscape
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.04, 0.10, 0.66, 0.74])
    mx0, my0 = merc(lon0, lat0)
    mx1, my1 = merc(lon1, lat1)

    base_url, label_url, base_credit = BASEMAPS[basemap]
    z = _zoom_for(2 * half_deg)
    try:
        bimg, bext, ok = _stitch(base_url, lon0, lat0, lon1, lat1, z)
        if ok:
            ax.imshow(bimg, extent=bext, origin="upper", zorder=0, interpolation="bilinear")
        else:
            base_credit = ""
    except Exception:
        base_credit = ""

    if grid is not None:
        vals, cols = zip(*[(v, c) for v, c in stops])
        vmax = vals[-1]
        cmap = LinearSegmentedColormap.from_list("geoai", [(v / vmax, c) for v, c in stops])
        g = np.array(grid["values"], dtype=float)
        la, lo, d = np.array(grid["lats"]), np.array(grid["lons"]), grid["d"]
        ex, _ = merc(np.append(lo - d / 2, lo[-1] + d / 2), 0)
        _, ey = merc(0, np.append(la - d / 2, la[-1] + d / 2))
        mesh = ax.pcolormesh(ex, ey, np.ma.masked_invalid(g), cmap=cmap, vmin=0, vmax=vmax, alpha=0.62,
                             shading="flat", zorder=1, edgecolors="none")
        cax = fig.add_axes([0.73, 0.62, 0.23, 0.025])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.set_ticks([v for v, _ in stops])
        cb.ax.tick_params(labelsize=8)
        cb.set_label(legend_label, fontsize=9)
        who = stops[1][0]
        cax.axvline(who, color="black", lw=2)
        cax.text(who, 1.6, "WHO", transform=cax.get_xaxis_transform(), ha="center", fontsize=7, fontweight="bold")
    elif image is not None:
        im = Image.open(io.BytesIO(image)).convert("RGBA")
        ax.imshow(im, extent=(mx0, mx1, my0, my1), origin="upper", alpha=0.75, zorder=1, interpolation="bilinear")
        cmap = LinearSegmentedColormap.from_list("sat", palette)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(*image_range))
        cax = fig.add_axes([0.73, 0.62, 0.23, 0.025])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=8)
        cb.set_label(legend_label, fontsize=9)

    try:
        limg, lext, ok = _stitch(label_url, lon0, lat0, lon1, lat1, z)
        if ok:
            ax.imshow(limg, extent=lext, origin="upper", zorder=2, interpolation="bilinear")
    except Exception:
        pass

    cx, cy = merc(lon, lat)
    ax.plot(cx, cy, marker="o", ms=11, mfc="#d7191c", mec="white", mew=2, zorder=5)
    ax.annotate(place, (cx, cy), xytext=(8, 8), textcoords="offset points", fontsize=11, fontweight="bold",
                color="#1d2733", zorder=6, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(mx0, mx1)
    ax.set_ylim(my0, my1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#9aa4b5")

    # scale bar (true distance at the map centre)
    m_per_unit = math.cos(math.radians(lat))
    width_km = (mx1 - mx0) * m_per_unit / 1000
    km = _nice_km(width_km / 5)
    bar = km * 1000 / m_per_unit
    bx, by = mx0 + (mx1 - mx0) * 0.04, my0 + (my1 - my0) * 0.05
    ax.add_patch(plt.Rectangle((bx - bar * 0.05, by - (my1 - my0) * 0.02), bar * 1.1, (my1 - my0) * 0.06,
                               fc="white", ec="none", alpha=0.85, zorder=6))
    ax.plot([bx, bx + bar], [by, by], color="black", lw=3, zorder=7, solid_capstyle="butt")
    ax.plot([bx, bx + bar / 2], [by, by], color="white", lw=1.2, zorder=8, solid_capstyle="butt")
    ax.text(bx + bar / 2, by + (my1 - my0) * 0.015, f"{km} km", ha="center", fontsize=8, zorder=8)

    # north arrow
    ax.annotate("N", xy=(0.95, 0.94), xytext=(0.95, 0.84), xycoords="axes fraction", ha="center", va="center",
                fontsize=13, fontweight="bold", zorder=8,
                arrowprops=dict(facecolor="black", width=4, headwidth=12, headlength=12))

    # coordinates on the frame
    for frac, val, ha in [(0, lon0, "left"), (0.5, lon, "center"), (1, lon1, "right")]:
        ax.text(frac, -0.025, f"{abs(val):.1f}°{'E' if val >= 0 else 'W'}", transform=ax.transAxes, ha=ha,
                fontsize=7, color="#6b7686")
    for frac, val, va in [(0, lat0, "bottom"), (0.5, lat, "center"), (1, lat1, "top")]:
        ax.text(-0.012, frac, f"{abs(val):.1f}°{'N' if val >= 0 else 'S'}", transform=ax.transAxes, ha="right",
                va=va, fontsize=7, color="#6b7686", rotation=90)

    # titles
    fig.text(0.04, 0.935, title, fontsize=20, fontweight="bold", color="#1d2733")
    fig.text(0.04, 0.895, subtitle, fontsize=11, color="#4a5565")
    fig.text(0.73, 0.70, "Legend", fontsize=12, fontweight="bold", color="#1d2733")

    # key numbers
    if stats:
        y = 0.52
        fig.text(0.73, y, "Key numbers", fontsize=12, fontweight="bold", color="#1d2733")
        for k, v in stats:
            y -= 0.045
            fig.text(0.73, y, k, fontsize=9, color="#6b7686")
            fig.text(0.96, y, v, fontsize=10, fontweight="bold", color="#1d2733", ha="right")

    # inset: where in the world
    ins = fig.add_axes([0.80, 0.115, 0.16, 0.13])
    wx0, wy0 = merc(-180, -60)
    wx1, wy1 = merc(180, 80)
    try:
        wimg, wext, ok = _stitch(base_url, -179.9, -60, 179.9, 80, 1)
        if ok:
            ins.imshow(wimg, extent=wext, origin="upper", interpolation="bilinear")
        else:
            ins.set_facecolor("#e9f0f7")
    except Exception:
        ins.set_facecolor("#e9f0f7")
    ins.plot(cx, cy, marker="s", ms=6, mfc="#d7191c", mec="white")
    ins.set_xlim(wx0, wx1)
    ins.set_ylim(wy0, wy1)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_title("Location", fontsize=8, color="#6b7686")

    foot = " · ".join(x for x in [credits, base_credit, "Made with GeoAI"] if x)
    fig.text(0.04, 0.035, foot, fontsize=7, color="#6b7686", wrap=True)

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200 if fmt == "png" else None, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ===========================================================================
# Professional area maps: country / region surface or region (county) choropleth
# ===========================================================================
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

CLASS_COLOURS = ["#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027", "#7b3294"]


def _fill_nan(z):
    z = z.copy()
    for _ in range(50):
        m = np.isnan(z)
        if not m.any():
            break
        p = np.pad(z, 1, constant_values=np.nan)
        nb = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        with np.errstate(all="ignore"):
            mean = np.nanmean(nb, axis=0)
        z[m] = mean[m]
    return z


def upsample(lats, lons, z, factor=10):
    lats, lons = np.asarray(lats), np.asarray(lons)
    z = _fill_nan(np.asarray(z, dtype=float))
    fl = np.linspace(lats.min(), lats.max(), (len(lats) - 1) * factor + 1)
    fo = np.linspace(lons.min(), lons.max(), (len(lons) - 1) * factor + 1)
    tmp = np.array([np.interp(fo, lons, row) for row in z])
    fine = np.array([np.interp(fl, lats, col) for col in tmp.T]).T
    return fl, fo, fine


# WHO 2021 guideline levels used as class limits (µg/m³), keyed by the short-term guideline value:
# PM2.5: annual 5, 24-h 15, interim targets 25 / 37.5 / 50 / 75
# PM10:  annual 15, 24-h 45, interim targets 50 / 75 / 100 / 150
# NO2:   annual 10, 24-h 25, interim targets 50 / 120
# O3:    peak season 60, 8-h 100, interim targets 120 / 160
WHO_STEPS = {15.0: [0, 5, 10, 15, 25, 37.5, 50, 75], 45.0: [0, 15, 30, 45, 50, 75, 100, 150],
             25.0: [0, 10, 25, 50, 120], 100.0: [0, 60, 100, 120, 160]}
EIGHT = ["#1a9850", "#66bd63", "#a6d96a", "#fee08b", "#fdae61", "#f46d43", "#d73027", "#7b3294"]


def class_breaks(values, mode, who):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if mode == "who":
        steps = WHO_STEPS.get(float(who), [0, who / 3, 2 * who / 3, who, 2 * who, 3 * who, 5 * who])
        breaks = list(steps) + [max(steps[-1] * 1.5, float(v.max()) + 1)]
        n = len(breaks) - 1
        cols = [EIGHT[int(round(i * 7 / max(n - 1, 1)))] for i in range(n)]
        return breaks, cols
    qs = np.unique(np.round(np.quantile(v, [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1]), 1))
    if len(qs) < 3:
        qs = np.round(np.linspace(v.min(), v.max() + 0.1, 4), 1)
    n = len(qs) - 1
    cols = [CLASS_COLOURS[int(round(i * 5 / max(n - 1, 1)))] for i in range(n)]
    return list(qs), cols


def _ccw(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


def _merc_ring(r):
    x, y = merc(r[:, 0], r[:, 1])
    return np.column_stack([x, y])


def _inverse_mask(ax, outer_rings, box, color="white", alpha=0.93):
    (x0, y0, x1, y1) = box
    verts, codes = [], []
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    verts += list(rect)
    codes += [MPath.MOVETO] + [MPath.LINETO] * 3 + [MPath.CLOSEPOLY]
    for r in outer_rings:
        m = _merc_ring(r)
        if _ccw(m):
            m = m[::-1]
        verts += list(m)
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(m) - 2) + [MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", alpha=alpha, zorder=3))


def _draw_rings(ax, rings, **kw):
    for r in rings:
        m = _merc_ring(r)
        ax.plot(m[:, 0], m[:, 1], **kw)


def zonal_stats(regs, fl, fo, fine, contains):
    gx, gy = np.meshgrid(fo, fl)
    out = {}
    for reg in regs:
        m = contains(reg["outer"], gx, gy)
        if m.sum() == 0:  # very small region: take the nearest grid value at its centre
            c = np.mean(np.vstack(reg["outer"]), axis=0)
            i, j = np.abs(fl - c[1]).argmin(), np.abs(fo - c[0]).argmin()
            vals = np.array([fine[i, j]])
        else:
            vals = fine[m]
        out[reg["name"]] = {"mean": float(np.nanmean(vals)), "min": float(np.nanmin(vals)),
                            "max": float(np.nanmax(vals))}
    return out


def make_area_map(*, area, grid, title, subtitle, legend_label, credits, who, stats=None, places=(),
                  mode="surface", classes="quantile", basemap="Light (clean, for printing)", fmt="png",
                  image=None, image_range=None, palette=None, contains=None, label_point=None):
    """area = dict(name, bbox=(lon0,lat0,lon1,lat1), outer=[rings] or None, regions=[...]).
       grid = dict(lats, lons, values) in display units (None when an image is given).
       Returns (bytes, zonal_stats or None)."""
    lon0, lat0, lon1, lat1 = area["bbox"]
    pad = 0.04 if area.get("outer") else 0.0
    pad_x, pad_y = (lon1 - lon0) * pad, (lat1 - lat0) * pad
    lon0, lon1, lat0, lat1 = lon0 - pad_x, lon1 + pad_x, lat0 - pad_y, lat1 + pad_y
    mx0, my0 = merc(lon0, lat0)
    mx1, my1 = merc(lon1, lat1)
    fig = plt.figure(figsize=(11.69, 8.27), dpi=200)
    ax = fig.add_axes([0.04, 0.10, 0.64, 0.75])
    ax.set_xlim(mx0, mx1)
    ax.set_ylim(my0, my1)
    ax.set_aspect("equal", adjustable="box", anchor="W")

    base_url, label_url, base_credit = BASEMAPS[basemap]
    z = _zoom_for(max(lon1 - lon0, (lat1 - lat0) * 1.4))
    try:
        bimg, bext, ok = _stitch(base_url, lon0, lat0, lon1, lat1, z)
        if ok:
            ax.imshow(bimg, extent=bext, origin="upper", zorder=0, interpolation="bilinear")
        else:
            base_credit = ""
    except Exception:
        base_credit = ""

    regs = area.get("regions") or []
    zstats = None
    legend_handles = []
    if grid is not None:
        fl, fo, fine = upsample(grid["lats"], grid["lons"], grid["values"])
        inside_vals = fine
        if area.get("outer") and contains:
            gx, gy = np.meshgrid(fo, fl)
            msk = contains(area["outer"], gx, gy)
            if msk.any():
                inside_vals = fine[msk]
        if regs and contains:
            zstats = zonal_stats(regs, fl, fo, fine, contains)
        source_vals = [s["mean"] for s in zstats.values()] if (mode == "regions" and zstats) else inside_vals
        breaks, cols = class_breaks(source_vals, classes, who)
        cmap = matplotlib.colors.ListedColormap(cols)
        norm = matplotlib.colors.BoundaryNorm(breaks, len(cols))
        if mode == "regions" and zstats:
            for reg in regs:
                s = zstats[reg["name"]]
                k = int(np.clip(np.searchsorted(breaks, s["mean"], side="right") - 1, 0, len(cols) - 1))
                s["class"] = f"{breaks[k]:g}–{breaks[k + 1]:g}"
                for r in reg["outer"]:
                    m = _merc_ring(r)
                    ax.fill(m[:, 0], m[:, 1], fc=cols[k], ec="white", lw=0.8, zorder=2)
            span = max(lon1 - lon0, lat1 - lat0)
            for reg in regs:
                if label_point is None:
                    break
                r0 = max(reg["outer"], key=lambda r: np.ptp(r[:, 0]) * np.ptp(r[:, 1]))
                if max(np.ptp(r0[:, 0]), np.ptp(r0[:, 1])) < span * 0.06:
                    continue  # too small to label
                c = label_point(reg["outer"])
                cx, cy = merc(c[0], c[1])
                ax.text(cx, cy, f"{zstats[reg['name']]['mean']:.1f}", ha="center", va="center", fontsize=7,
                        fontweight="bold", color="#1d2733", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        else:
            ex, _ = merc(np.append(fo - (fo[1] - fo[0]) / 2, fo[-1] + (fo[1] - fo[0]) / 2), 0)
            _, ey = merc(0, np.append(fl - (fl[1] - fl[0]) / 2, fl[-1] + (fl[1] - fl[0]) / 2))
            ax.pcolormesh(ex, ey, fine, cmap=cmap, norm=norm, alpha=0.85, shading="flat", zorder=1,
                          edgecolors="none", rasterized=True)
            if zstats:
                for reg in regs:
                    s = zstats[reg["name"]]
                    k = int(np.clip(np.searchsorted(breaks, s["mean"], side="right") - 1, 0, len(cols) - 1))
                    s["class"] = f"{breaks[k]:g}–{breaks[k + 1]:g}"
        for i, c in enumerate(cols):
            hi = breaks[i + 1]
            lab = f"{breaks[i]:g} – {hi:g}" if i < len(cols) - 1 or classes != "who" else f"> {breaks[i]:g}"
            if classes == "who" and abs(breaks[i + 1] - who) < 1e-9:
                lab += "  (WHO 24-h guideline)" if who != 100 else "  (WHO 8-h guideline)"
            legend_handles.append(Patch(fc=c, ec="#9aa4b5", label=lab))
    elif image is not None:
        im = Image.open(io.BytesIO(image)).convert("RGBA")
        bx0, by0 = merc(area["bbox"][0], area["bbox"][1])
        bx1, by1 = merc(area["bbox"][2], area["bbox"][3])
        ax.imshow(im, extent=(bx0, bx1, by0, by1), origin="upper", alpha=0.85, zorder=1, interpolation="bilinear")
        n = len(palette)
        for i, c in enumerate(palette):
            v = image_range[0] + (image_range[1] - image_range[0]) * i / (n - 1)
            if i % 2 == 0 or i == n - 1:
                legend_handles.append(Patch(fc=c, ec="#9aa4b5", label=f"{v:.3g}"))

    if area.get("outer"):
        _inverse_mask(ax, area["outer"], (mx0 - 1e5, my0 - 1e5, mx1 + 1e5, my1 + 1e5))
    try:
        limg, lext, ok = _stitch(label_url, lon0, lat0, lon1, lat1, z)
        if ok and mode != "regions":
            ax.imshow(limg, extent=lext, origin="upper", zorder=4, interpolation="bilinear", alpha=0.9)
    except Exception:
        pass
    for r in area.get("borders") or []:
        m = _merc_ring(r)
        ax.plot(m[:, 0], m[:, 1], color="#4a5565", lw=0.8, zorder=5, alpha=0.8)
    if mode != "bubbles":
        for reg in regs:
            _draw_rings(ax, reg["outer"], color="#3d4654", lw=0.6, zorder=5)
    if area.get("outer"):
        _draw_rings(ax, area["outer"], color="#111827", lw=1.6, zorder=6)
    if mode == "bubbles" and grid is not None:
        import matplotlib.patheffects as pe
        halo = [pe.withStroke(linewidth=2.5, foreground="#1d2733")]
        for name, lo, la, main in places:
            if not (lon0 <= lo <= lon1 and lat0 <= la <= lat1):
                continue
            i = np.clip(np.searchsorted(fl, la), 1, len(fl) - 1)
            j = np.clip(np.searchsorted(fo, lo), 1, len(fo) - 1)
            v = float(np.nanmean(fine[i - 1:i + 1, j - 1:j + 1]))
            k = int(np.clip(np.searchsorted(breaks, v, side="right") - 1, 0, len(cols) - 1))
            px, py = merc(lo, la)
            if main:
                ax.plot(px, py, marker="o", ms=30, mfc=cols[k], mec="#1d2733", mew=1, alpha=0.55, zorder=8)
            ax.plot(px, py, marker="o", ms=13 if main else 8, mfc=cols[k], mec="#1d2733", mew=1.2, zorder=9)
            ax.plot(px, py, marker="o", ms=4 if main else 2.5, mfc="white", mec="none", zorder=10)
            ax.annotate(f"{name}  {v:.0f}" if main else name, (px, py), xytext=(9, 5), textcoords="offset points",
                        fontsize=12 if main else 8, fontweight="bold", color="white", zorder=11, path_effects=halo,
                        clip_on=True)
    for name, lo, la, main in places:
        if mode in ("regions", "bubbles") and not (main and mode == "regions"):
            continue
        if lon0 <= lo <= lon1 and lat0 <= la <= lat1:
            px, py = merc(lo, la)
            ax.plot(px, py, marker="o", ms=7 if main else 4, mfc="#d7191c" if main else "#1d2733", mec="white",
                    mew=1.2, zorder=8)
            ax.annotate(name, (px, py), xytext=(5, 4), textcoords="offset points", fontsize=9 if main else 7,
                        fontweight="bold" if main else "normal", color="#1d2733", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_xlim(mx0, mx1)
    ax.set_ylim(my0, my1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#9aa4b5")

    # scale bar + north arrow
    latc = (lat0 + lat1) / 2
    k_ = math.cos(math.radians(latc))
    km = _nice_km((mx1 - mx0) * k_ / 1000 / 5)
    bar = km * 1000 / k_
    bx, by = mx0 + (mx1 - mx0) * 0.04, my0 + (my1 - my0) * 0.05
    ax.add_patch(plt.Rectangle((bx - bar * 0.06, by - (my1 - my0) * 0.025), bar * 1.12, (my1 - my0) * 0.07,
                               fc="white", ec="none", alpha=0.9, zorder=10))
    ax.plot([bx, bx + bar], [by, by], color="black", lw=3, zorder=11, solid_capstyle="butt")
    ax.plot([bx, bx + bar / 2], [by, by], color="white", lw=1.2, zorder=12, solid_capstyle="butt")
    ax.text(bx + bar / 2, by + (my1 - my0) * 0.016, f"{km} km", ha="center", fontsize=8, zorder=12)
    ax.annotate("N", xy=(0.95, 0.94), xytext=(0.95, 0.84), xycoords="axes fraction", ha="center", va="center",
                fontsize=13, fontweight="bold", zorder=12,
                arrowprops=dict(facecolor="black", width=4, headwidth=12, headlength=12))

    fig.text(0.04, 0.935, title, fontsize=20, fontweight="bold", color="#1d2733")
    fig.text(0.04, 0.895, subtitle, fontsize=11, color="#4a5565")
    ax.apply_aspect()
    pos = ax.get_position()
    xr = float(np.clip(pos.x1 + 0.035, 0.45, 0.71))   # side panel starts right next to the map
    xe = min(xr + 0.26, 0.965)
    lg = fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(xr, 0.86), frameon=False,
                    title=legend_label, title_fontsize=10, fontsize=9, alignment="left")
    lg.get_title().set_fontweight("bold")
    y = 0.56
    if stats:
        fig.text(xr, y, "Key numbers", fontsize=12, fontweight="bold", color="#1d2733")
        for kk, v in stats:
            y -= 0.04
            fig.text(xr, y, kk, fontsize=9, color="#6b7686")
            fig.text(xe, y, v, fontsize=10, fontweight="bold", color="#1d2733", ha="right")
    if zstats and mode == "regions":
        top = sorted(zstats.items(), key=lambda kv: -kv[1]["mean"])[:3]
        y -= 0.06
        fig.text(xr, y, "Highest regions", fontsize=10, fontweight="bold", color="#1d2733")
        for nm, s in top:
            y -= 0.032
            fig.text(xr, y, nm[:28], fontsize=8.5, color="#4a5565")
            fig.text(xe, y, f"{s['mean']:.1f}", fontsize=9, fontweight="bold", ha="right")

    ins = fig.add_axes([xe - 0.16, 0.07, 0.16, 0.11])
    wx0, wy0 = merc(-180, -60)
    wx1, wy1 = merc(180, 80)
    try:
        wimg, wext, ok = _stitch(base_url, -179.9, -60, 179.9, 80, 1)
        if ok:
            ins.imshow(wimg, extent=wext, origin="upper", interpolation="bilinear")
        else:
            ins.set_facecolor("#e9f0f7")
    except Exception:
        ins.set_facecolor("#e9f0f7")
    ins.add_patch(plt.Rectangle((mx0, my0), max(mx1 - mx0, 6e5), max(my1 - my0, 6e5), fc="none", ec="#d7191c", lw=1.5))
    ins.set_xlim(wx0, wx1)
    ins.set_ylim(wy0, wy1)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_title("Location", fontsize=8, color="#6b7686")

    foot = " · ".join(x for x in [credits, base_credit, "Made with GeoAI"] if x)
    fig.text(0.04, 0.035, foot, fontsize=7, color="#6b7686")
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200 if fmt == "png" else None, facecolor="white")
    plt.close(fig)
    return buf.getvalue(), zstats
