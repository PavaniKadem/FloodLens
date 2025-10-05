# score_patches.py - Ranking using AI
from pathlib import Path
import math, random
import numpy as np
import pandas as pd
import folium
import rasterio
from skimage.morphology import closing, opening, square, dilation, erosion
from skimage.measure import find_contours, label, regionprops

# XGBoost; else fallback to RandomForest
ModelType = "xgb"
try:
    from xgboost import XGBRegressor  # noqa: F401
except Exception:
    ModelType = "rf"
    from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MASK_PATH = DATA / "flood_mask.tif"
OUT_CSV = DATA / "flood_scores.csv"
OUT_HTML = DATA / "floodlens_ranked.html"

# Map / AOI
AOI_BBOX = (-95.90, 29.45, -95.00, 30.15)  # minx, miny, maxx, maxy
CENTER = (29.7604, -95.3698)
ZOOM = 10

# POIs for distance features
POIS = {
    "CBD": (29.7604, -95.3698),
    "IAH": (29.9902, -95.3368),
    "HOU": (29.6454, -95.2789),
    "I10x610W": (29.7853, -95.4200),
}

# Cleaning thresholds (same as overlay)
BORDER_FR = 0.06
CLOSE_SZ  = 7
OPEN_SZ   = 5
MIN_AREA_PX = 30_000
MIN_AXIS_PX = 40
MAX_ASPECT  = 12.0
MIN_CONTOUR_PTS = 16

def log(*a): print("[score]", *a)

#helpers
def deg_to_m_scale(lat_deg: float):
    lat = math.radians(lat_deg)
    return 110_574.0, 111_320.0 * math.cos(lat)

def px_to_latlon(y, x, H, W, bbox):
    minx, miny, maxx, maxy = bbox
    lat = maxy - (maxy - miny) * (y / H)
    lon = minx + (maxx - minx) * (x / W)
    return float(lat), float(lon)

def poly_area_perimeter_km2(coords):
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    lats = np.array([c[0] for c in coords], float)
    lons = np.array([c[1] for c in coords], float)
    lat0 = float(np.mean(lats))
    sy, sx = deg_to_m_scale(lat0)
    X = (lons - lons[0]) * sx
    Y = (lats - lats[0]) * sy
    area_m2 = 0.5 * abs(np.dot(X[:-1], Y[1:]) - np.dot(Y[:-1], X[1:]))
    perim_m = np.sum(np.hypot(np.diff(X), np.diff(Y)))
    return area_m2 / 1e6, perim_m / 1000.0  # km², km

def haversine_km(a, b):
    R = 6371.0
    la1, lo1 = map(math.radians, a)
    la2, lo2 = map(math.radians, b)
    dlat = la2 - la1; dlon = lo2 - lo1
    s = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(s))

def remove_seams(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    col_dense = out.sum(axis=0) > 0.55 * H
    row_dense = out.sum(axis=1) > 0.55 * W
    if col_dense.any():
        out[:, col_dense] = 0
        log("Removed seam columns:", int(col_dense.sum()))
    if row_dense.any():
        out[row_dense, :] = 0
        log("Removed seam rows:", int(row_dense.sum()))
    return out

def ellipse_points_pix(cx, cy, a, b, angle_rad, n=72):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = a * np.cos(t); ys = b * np.sin(t)
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    xr = xs * ca - ys * sa
    yr = xs * sa + ys * ca
    x = cx + xr; y = cy + yr
    return np.stack([y, x], axis=1)

def contour_area_px(contour_yx: np.ndarray) -> float:
    x = contour_yx[:, 1]; y = contour_yx[:, 0]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))

#ellipse extraction
def extract_real_ellipses():
    if not MASK_PATH.exists():
        return []
    with rasterio.open(MASK_PATH) as ds:
        arr = ds.read(1)

    H, W = arr.shape
    m = (arr > 0).astype(np.uint8)

    b = int(round(BORDER_FR * min(H, W)))
    if b > 0:
        m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0
    m = remove_seams(m)
    m = closing(m, square(CLOSE_SZ))
    m = opening(m, square(OPEN_SZ)).astype(np.uint8)
    m = dilation(m, square(2))
    m = erosion(m, square(2))

    L = label(m, connectivity=1)
    regs = regionprops(L)

    named = []
    idx = 0
    for r in regs:
        minr, minc, maxr, maxc = r.bbox
        h = maxr - minr; w = maxc - minc
        if h < MIN_AXIS_PX or w < MIN_AXIS_PX:
            continue
        sub = (L[minr:maxr, minc:maxc] == r.label).astype(float)
        cs = find_contours(sub, 0.5)
        if not cs:
            continue
        c0 = max(cs, key=lambda c: c.shape[0])
        if c0.shape[0] < MIN_CONTOUR_PTS:
            continue
        c0[:, 0] += minr; c0[:, 1] += minc
        area_px = contour_area_px(np.vstack([c0, c0[0]]))
        if area_px < MIN_AREA_PX:
            continue

        cy, cx = r.centroid
        a = float(r.axis_major_length) / 2.0
        b = float(r.axis_minor_length) / 2.0
        if a < MIN_AXIS_PX or b < MIN_AXIS_PX:
            continue
        asp = max(a / max(1e-6, b), b / max(1e-6, a))
        if asp > MAX_ASPECT:
            continue

        angle = float(r.orientation)
        yx = ellipse_points_pix(cx, cy, a, b, angle, n=72)
        ring = [px_to_latlon(y, x, H, W, AOI_BBOX) for (y, x) in yx]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        idx += 1
        named.append((f"FloodLens ellipse {idx}", ring))

    log(f"Real ellipses kept: {len(named)}")
    return named

# synthetic ellipses fallback
def synth_ellipse_ring(lat_c, lon_c, rlat, rlon, angle_deg=0.0, jitter=0.08, n=72):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    rl = rlat * (1.0 + np.random.uniform(-jitter, jitter, size=n))
    rr = rlon * (1.0 + np.random.uniform(-jitter, jitter, size=n))
    x = rl * np.cos(t); y = rr * np.sin(t)
    a0 = np.deg2rad(angle_deg)
    xr = x * np.cos(a0) - y * np.sin(a0)
    yr = x * np.sin(a0) + y * np.cos(a0)
    lat = lat_c + xr; lon = lon_c + yr
    ring = list(zip(lat.tolist(), lon.tolist()))
    ring.append(ring[0])
    return ring

def synth_ellipses():
    specs = [
        ("Addicks Reservoir (demo)", (29.80, -95.63), 0.050, 0.060, -18),
        ("Barker Reservoir (demo)",  (29.76, -95.63), 0.042, 0.050, -12),
        ("Greens Bayou (demo)",      (29.90, -95.27), 0.025, 0.095,  -8),
        ("Clear Lake (demo)",        (29.56, -95.07), 0.035, 0.065,  10),
    ]
    return [(name, synth_ellipse_ring(lat, lon, rlat, rlon, angle_deg=ang))
            for name,(lat,lon),rlat,rlon,ang in specs]

#features & AI ranking
def features_from_rings(named_rings):
    rows = []
    for i, (name, ring) in enumerate(named_rings, 1):
        lat = float(np.mean([p[0] for p in ring[:-1]]))
        lon = float(np.mean([p[1] for p in ring[:-1]]))
        area_km2, perim_km = poly_area_perimeter_km2(ring)
        compactness = (perim_km**2) / max(1e-6, area_km2)
        dist_cbd = haversine_km((lat, lon), POIS["CBD"])
        dist_airport = min(haversine_km((lat, lon), POIS["IAH"]),
                           haversine_km((lat, lon), POIS["HOU"]))
        dist_roadhub = haversine_km((lat, lon), POIS["I10x610W"])
        rows.append({
            "id": i, "name": name, "lat": lat, "lon": lon,
            "area_km2": area_km2, "perim_km": perim_km, "compactness": compactness,
            "dist_cbd_km": dist_cbd, "dist_airport_km": dist_airport, "dist_roadhub_km": dist_roadhub,
            "ring": ring,
        })
    return pd.DataFrame(rows)

def heuristic_label(df: pd.DataFrame):
    return (
        1.0 * df["area_km2"] +
        0.4 * np.clip(30 - df["dist_cbd_km"], 0, None) +
        0.3 * np.clip(25 - df["dist_airport_km"], 0, None) +
        0.2 * np.clip(20 - df["dist_roadhub_km"], 0, None) +
        0.1 * (1.0 / np.clip(df["compactness"], 1e-3, None))
    ).values

def train_and_rank(df: pd.DataFrame):
    feats = ["area_km2","perim_km","compactness","dist_cbd_km","dist_airport_km","dist_roadhub_km"]
    X = df[feats].values.astype(np.float32)
    y = heuristic_label(df)

    if ModelType == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.08,
            subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
            random_state=42, n_jobs=1
        )
        log("Using XGBoost")
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)
        log("XGBoost not available → using RandomForest")

    model.fit(X, y)
    df["impact_score"] = model.predict(X)
    df = df.sort_values("impact_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df)+1)
    return df

def color_for_rank(rank):
    palette = ["#b30000","#d7301f","#ef6548","#fc8d59","#fdbb84","#fdd49e"]
    return palette[min(rank-1, len(palette)-1)]

def make_map(df: pd.DataFrame):
    m = folium.Map(location=CENTER, zoom_start=ZOOM, tiles="OpenStreetMap", control_scale=True)
    for _, r in df.iterrows():
        folium.Polygon(
            locations=r["ring"], color="#000", weight=2,
            fill=True, fill_color=color_for_rank(int(r["rank"])), fill_opacity=0.45,
            tooltip=f'#{int(r["rank"])} • {r["name"]} • score={r["impact_score"]:.2f} • area={r["area_km2"]:.2f} km²',
        ).add_to(m)
        folium.Marker(
            location=(float(r["lat"]), float(r["lon"])),
            tooltip=f'#{int(r["rank"])}',
            icon=folium.DivIcon(html=f'<div style="font-weight:700;color:#000;background:#fff;padding:2px 4px;border-radius:4px;border:1px solid #000">#{int(r["rank"])}</div>')
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUT_HTML)

def main():
    random.seed(42); np.random.seed(42)
    DATA.mkdir(parents=True, exist_ok=True)

    real = extract_real_ellipses()
    if real:
        named = real
        log(f"Using REAL ellipses: {len(named)}")
    else:
        named = synth_ellipses()
        log(f"Using SYNTHETIC ellipses: {len(named)}")

    df = features_from_rings(named)
    if df.empty:
        log("No ellipses to score.")
        return

    df_scored = train_and_rank(df)

    keep = ["rank","impact_score","name","area_km2","compactness",
            "dist_cbd_km","dist_airport_km","dist_roadhub_km","lat","lon"]
    df_scored[keep].to_csv(OUT_CSV, index=False)
    make_map(df_scored)

    print(f"[score] Saved table: {OUT_CSV}")
    print(f"[score] Saved map  : {OUT_HTML}")
    print("\nOpen with:\n  open", OUT_HTML)

if __name__ == "__main__":
    main()