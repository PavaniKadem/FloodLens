# plot_mask.py
# Make a fast overlay PNG: POST SAR (grayscale) + flood_mask.tif (blue)

from pathlib import Path
import glob, re
import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UNZ  = DATA / "s1_unzipped"
MASK = DATA / "flood_mask.tif"
OUT  = DATA / "flood_overlay.png"

PREVIEW_STEP = 8  # downsample factor for speed

def log(*a): print("[overlay]", *a)

def stretch01(x):
    v = np.isfinite(x)
    p2, p98 = np.percentile(x[v], [2, 98]) if v.any() else (0, 1)
    return np.clip((x - p2) / max(1e-6, p98 - p2), 0, 1)

def to_db(a, eps=1e-6):  # power - dB
    return 10.0 * np.log10(np.clip(a.astype("float32"), eps, None))

def find_post_band(unzipped: Path):
    # Prefer VV, else VH
    patt = str(unzipped / "**" / "measurement" / "*.tif*")
    tifs = [Path(p) for p in glob.glob(patt, recursive=True)]
    if not tifs:
        tifs = [Path(p) for p in glob.glob(str(unzipped / "**" / "*.tif*"), recursive=True)]
    if not tifs:
        raise SystemExit("[overlay] No SAR bands found; run download step first.")

    iw = [p for p in tifs if "iw" in p.name.lower()]
    vv = [p for p in iw if "vv" in p.name.lower()]
    vh = [p for p in iw if "vh" in p.name.lower()]
    cand = vv if len(vv) else vh
    pol = "VV" if cand is vv else "VH"
    if not cand:
        raise SystemExit("[overlay] No IW VV/VH bands found.")

    ts = re.compile(r"(\d{8}t\d{6})", re.I)
    def key(p: Path):
        m = ts.search(p.name); return m.group(1) if m else p.name.lower()
    cand.sort(key=key)
    return pol, cand[-1]  # POST

def main():
    log("Root:", ROOT)
    if not MASK.exists():
        raise SystemExit(f"[overlay] Missing {MASK}. Run mask_flood_min.py first.")
    pol, post_path = find_post_band(UNZ)
    log(f"POST ({pol}): {post_path.name}")

    with rasterio.open(post_path) as ds_post:
        post = ds_post.read(1).astype("float32")
    with rasterio.open(MASK) as ds_mask:
        mask = ds_mask.read(1).astype("uint8")

    # Downsample for quick, reliable rendering
    s = PREVIEW_STEP
    post_db = to_db(post)
    bg = stretch01(post_db[::s, ::s])
    m  = (mask[::s, ::s] > 0)

    #Robust size harmonization
    h = min(bg.shape[0], m.shape[0])
    w = min(bg.shape[1], m.shape[1])
    if (bg.shape[0], bg.shape[1]) != (h, w) or (m.shape[0], m.shape[1]) != (h, w):
        bg = bg[:h, :w]
        m  = m[:h, :w]

    # Compose RGB
    gray = (bg * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    blue = np.zeros_like(rgb); blue[..., 2] = 255
    alpha = 0.5
    rgb[m] = (alpha * blue[m] + (1 - alpha) * rgb[m]).astype(np.uint8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(OUT)
    log("Saved overlay:", OUT)
    print(f"\nOpen it:\n  open {OUT}")

if __name__ == "__main__":
    main()