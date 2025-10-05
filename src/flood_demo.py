# synth_flood_demo.py 
from pathlib import Path
import math
import numpy as np
import folium
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_HTML = DATA / "map_overlay_demo.html"

# Base map center
CENTER = (29.76, -95.37)  # lat, lon
ZOOM = 10

# parametric rotated ellipse -> lat/lon ring
def ellipse_ring(lat_c, lon_c, dlat, dlon, angle_deg=0.0, n=120):
    """
    lat_c/lon_c: center in degrees
    dlat/dlon: semi-axes in degrees (rough footprint)
    angle_deg: rotation clockwise (visual alignment)
    n: points around the ellipse
    """
    ang = math.radians(angle_deg)
    t = np.linspace(0, 2*math.pi, n, endpoint=True)
    # parametric ellipse in local deltas
    x = dlat * np.cos(t)
    y = dlon * np.sin(t)
    # rotate
    xr = x * math.cos(ang) - y * math.sin(ang)
    yr = x * math.sin(ang) + y * math.cos(ang)
    lat = lat_c + xr
    lon = lon_c + yr
    return list(zip(lat.tolist(), lon.tolist()))

def add_flood_polygon(m, ring, name="Flooded area", color="#ff0033", edge="#000000"):
    folium.Polygon(
        locations=ring,
        color=edge, weight=2,
        fill=True, fill_color=color, fill_opacity=0.40,
        tooltip=name,
    ).add_to(m)

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    # Base map
    m = folium.Map(location=CENTER, zoom_start=ZOOM, tiles="OpenStreetMap", control_scale=True)

    # --- Synthetic but plausible patches (hand-picked centers & shapes) ---
    patches = [
        # Addicks Reservoir west of Houston
        {
            "name": "Addicks Reservoir overflow (demo)",
            "center": (29.80, -95.63),
            "dlat": 0.045, "dlon": 0.055, "angle": -20,
        },
        # Barker Reservoir
        {
            "name": "Barker Reservoir overflow (demo)",
            "center": (29.76, -95.63),
            "dlat": 0.038, "dlon": 0.050, "angle": -15,
        },
        # Greens Bayou (north/east side) — narrow, elongated
        {
            "name": "Greens Bayou segment (demo)",
            "center": (29.90, -95.27),
            "dlat": 0.020, "dlon": 0.090, "angle": -5,
        },
        # Clear Lake / Bay Area
        {
            "name": "Clear Lake low-lying (demo)",
            "center": (29.56, -95.07),
            "dlat": 0.030, "dlon": 0.060, "angle": 10,
        },
    ]

    for p in patches:
        ring = ellipse_ring(p["center"][0], p["center"][1], p["dlat"], p["dlon"], p["angle"])
        add_flood_polygon(m, ring, p["name"])

    # Optional: add labeled markers at centers
    for p in patches:
        folium.Marker(location=p["center"], tooltip=p["name"]).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUT_HTML)
    print(f"[demo] Saved synthetic flood map: {OUT_HTML}\nOpen it with:\n  open {OUT_HTML}\n")

if __name__ == "__main__":
    main()
