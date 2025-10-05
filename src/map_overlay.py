# map_overlay.py
from pathlib import Path
import math, random
import numpy as np
import folium
import rasterio
from skimage.morphology import closing, opening, square, dilation, erosion
from skimage.measure import find_contours, label, regionprops

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MASK = DATA / "flood_mask.tif"
OUT_HTML = DATA / "map_overlay.html"

# Houston AOI (min_lon, min_lat, max_lon, max_lat)
AOI_BBOX = (-95.90, 29.45, -95.00, 30.15)
CENTER = (29.76, -95.37)
ZOOM = 10

# Styling
FILL = "#ff0033"
EDGE = "#000000"
FILL_OPACITY = 0.42
EDGE_WEIGHT = 2

# Cleaning / filtering thresholds
BORDER_FR = 0.06            
CLOSE_SZ  = 7               
OPEN_SZ   = 5               
MIN_AREA_PX = 30_000        
MIN_AXIS_PX = 40           
MAX_ASPECT  = 12.0         
MIN_CONTOUR_PTS = 16        

def log(*a): print("[map]", *a)

#utilities
def px_to_latlon(y, x, H, W, bbox):
    minx, miny, maxx, maxy = bbox
    lat = maxy - (maxy - miny) * (y / H)
    lon = minx + (maxx - minx) * (x / W)
    return float(lat), float(lon)

def remove_seams(mask: np.ndarray) -> np.ndarray:
    """Zero out abnormally dense rows/cols (strip seams)."""
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
    """Ellipse in pixel coords centered at (cx,cy), semi-axes a,b, rotated by angle_rad (radians).
       Returns array of shape (n, 2) with (y, x) pixel coords (row,col)."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    cos_t, sin_t = np.cos(t), np.sin(t)
    xs = a * cos_t
    ys = b * sin_t
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    xr = xs * ca - ys * sa
    yr = xs * sa + ys * ca
    x = cx + xr
    y = cy + yr
    return np.stack([y, x], axis=1) 

def contour_area_px(contour_yx: np.ndarray) -> float:
    """Shoelace area from a (y,x) pixel contour."""
    x = contour_yx[:, 1]
    y = contour_yx[:, 0]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))

#ellipses from mask
def extract_ellipses_from_mask(arr: np.ndarray):
    """Return list of ellipse rings in lat/lon. Each ring is list[(lat,lon), ...]."""
    H, W = arr.shape
    m = (arr > 0).astype(np.uint8)

    # border trim
    b = int(round(BORDER_FR * min(H, W)))
    if b > 0:
        m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0

    # remove seam artifacts
    m = remove_seams(m)

    # morphology to tidy shapes
    m = closing(m, square(CLOSE_SZ))
    m = opening(m, square(OPEN_SZ)).astype(np.uint8)
    m = dilation(m, square(2))
    m = erosion(m, square(2))

    # label & regionprops for ellipse params
    L = label(m, connectivity=1)
    regs = regionprops(L)

    rings = []
    for r in regs:
        # quick bounding-box filter + pixel area
        minr, minc, maxr, maxc = r.bbox
        h = maxr - minr; w = maxc - minc
        if h < MIN_AXIS_PX or w < MIN_AXIS_PX:
            continue

        # contour-based area check
        sub = (L[minr:maxr, minc:maxc] == r.label).astype(float)
        cs = find_contours(sub, 0.5)
        if not cs:
            continue
        c0 = max(cs, key=lambda c: c.shape[0])  # largest contour
        if c0.shape[0] < MIN_CONTOUR_PTS:
            continue
        c0[:, 0] += minr  # offset back to full image coords
        c0[:, 1] += minc
        area_px = contour_area_px(np.vstack([c0, c0[0]]))
        if area_px < MIN_AREA_PX:
            continue

        # ellipse params from regionprops
        cy, cx = r.centroid
        a = float(r.axis_major_length) / 2.0
        b = float(r.axis_minor_length) / 2.0
        if a < MIN_AXIS_PX or b < MIN_AXIS_PX:
            continue
        asp = max(a / max(1e-6, b), b / max(1e-6, a))
        if asp > MAX_ASPECT:
            continue
        # skimage orientation
        angle = float(r.orientation)

        # sample ellipse as many points, map to lat/lon
        yx = ellipse_points_pix(cx, cy, a, b, angle, n=72)
        ring = [px_to_latlon(y, x, H, W, AOI_BBOX) for (y, x) in yx]
        # close ring
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        rings.append(ring)

    log(f"Real ellipses kept: {len(rings)}")
    return rings

# synthetic ellipses (fallback only)
def synth_ellipse_ring(lat_c, lon_c, rlat, rlon, angle_deg=0.0, jitter=0.08, n=72):
    """Generate a slightly irregular ellipse in lat/lon around a center."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    # small radial jitter to avoid perfect ovals
    rl = rlat * (1.0 + np.random.uniform(-jitter, jitter, size=n))
    rr = rlon * (1.0 + np.random.uniform(-jitter, jitter, size=n))
    x = rl * np.cos(t)
    y = rr * np.sin(t)
    a0 = np.deg2rad(angle_deg)
    xr = x * np.cos(a0) - y * np.sin(a0)
    yr = x * np.sin(a0) + y * np.cos(a0)
    lat = lat_c + xr
    lon = lon_c + yr
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

def add_ring(m, ring, label):
    folium.Polygon(
        locations=ring,
        color=EDGE, weight=EDGE_WEIGHT,
        fill=True, fill_color=FILL, fill_opacity=FILL_OPACITY,
        tooltip=label,
    ).add_to(m)

def main():
    random.seed(42); np.random.seed(42)
    DATA.mkdir(parents=True, exist_ok=True)
    m = folium.Map(location=CENTER, zoom_start=ZOOM, tiles="OpenStreetMap", control_scale=True)

    rings = []
    source = ""
    if MASK.exists():
        try:
            with rasterio.open(MASK) as ds:
                arr = ds.read(1)
            rings = extract_ellipses_from_mask(arr)
            source = "FloodLens (real ellipses)"
        except Exception as e:
            log("Failed to read/parse mask:", e)
            rings = []
    if not rings:
        source = "FloodLens (demo ellipses)"
        rings = [r for _, r in synth_ellipses()]
        log("No valid real ellipses → using synthetic ellipses")

    for i, ring in enumerate(rings, 1):
        add_ring(m, ring, f"{source} — region {i}")

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUT_HTML)
    log("Saved:", OUT_HTML)
    print(f"\nOpen:\n  open {OUT_HTML}\n")

if __name__ == "__main__":
    main()