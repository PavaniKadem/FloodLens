# mask_flood_min.py
# Robust flood mask from two Sentinel-1 IW/GRD measurement bands with VV polarization (falls back to VH)
# Outputs: data/flood_mask.tif, data/flood_preview.png

from pathlib import Path
import glob, re, warnings
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, square
import matplotlib
matplotlib.use("Agg")
from PIL import Image 

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UNZ  = DATA / "s1_unzipped"
OUT_MASK = DATA / "flood_mask.tif"
OUT_PNG  = DATA / "flood_preview.png"

# -------------- tuning -----------------
ADAPTIVE_WATER_P = 15       # percentile for water-like pixels in POST
MANUAL_THRESH_DB = 0.0      # fallback threshold on (pre_db - post_db)
DOWNSAMPLE_STEP  = 8        # for robust Otsu sampling
KERNEL = square(3)  # skimage >=0.25 API
PREVIEW_STEP     = 8        #preview PNG 

def log(*a): print("[mask]", *a)
def to_db(a, eps=1e-6): return 10.0 * np.log10(np.clip(a, eps, None))

def stretch01(x):
    v = np.isfinite(x)
    p2, p98 = np.percentile(x[v], [2, 98])
    return np.clip((x - p2) / max(1e-6, p98 - p2), 0, 1)

def find_two_bands(unzipped: Path):
    patt = str(unzipped / "**" / "measurement" / "*.tif*")
    tifs = [Path(p) for p in glob.glob(patt, recursive=True)]
    if not tifs:
        tifs = [Path(p) for p in glob.glob(str(unzipped / "**" / "*.tif*"), recursive=True)]
    if not tifs:
        raise SystemExit(f"[mask] No *.tif/.tiff found under {unzipped}. Run download step first.")

    iw = [p for p in tifs if "iw" in p.name.lower()]
    vv = [p for p in iw  if "vv" in p.name.lower()]
    vh = [p for p in iw  if "vh" in p.name.lower()]

    cand = vv if len(vv) >= 2 else vh
    pol = "VV" if cand is vv else "VH"
    if len(cand) < 2:
        raise SystemExit(f"[mask] Need ≥2 IW bands of same polarization. Found: VV={len(vv)} VH={len(vh)}")

    ts = re.compile(r"(\d{8}t\d{6})", re.I)
    def key(p: Path):
        m = ts.search(p.name)
        return m.group(1) if m else p.name.lower()
    cand.sort(key=key)

    pre_path, post_path = cand[0], cand[-1]
    return pol, pre_path, post_path

def align_post_to_pre(pre_path: Path, post_path: Path):
    log("Opening PRE :", pre_path.name)
    log("Opening POST:", post_path.name)
    with rasterio.open(pre_path) as pre_ds, rasterio.open(post_path) as post_ds:
        pre  = pre_ds.read(1).astype("float32")
        post = post_ds.read(1).astype("float32")
        pre_has, post_has = bool(pre_ds.crs), bool(post_ds.crs)
        log("CRS present? pre:", pre_has, "post:", post_has)

        if pre_has and post_has:
            post_aligned = np.empty_like(pre, dtype="float32")
            reproject(
                source=rasterio.band(post_ds, 1),
                destination=post_aligned,
                src_transform=post_ds.transform, src_crs=post_ds.crs,
                dst_transform=pre_ds.transform,  dst_crs=pre_ds.crs,
                resampling=Resampling.bilinear
            )
            profile = pre_ds.profile
        else:
            post_aligned = post_ds.read(1, out_shape=pre.shape, resampling=Resampling.nearest).astype("float32")
            profile = pre_ds.profile 
            log("Aligned by size only (no CRS).")

        return pre, post_aligned, profile

def save_fast_preview(post_db, mask, out_png: Path, step: int = PREVIEW_STEP):
    """Save a small, fast PNG preview using Pillow (no heavy Matplotlib)."""
    # Downsample by simple stride (fast & memory-friendly)
    bg_small   = stretch01(post_db[::step, ::step])
    mask_small = (mask[::step, ::step] > 0)

    # Convert to 8-bit grayscale
    gray = (bg_small * 255).astype(np.uint8)

    # Build RGB overlay (blue for flooded)
    h, w = gray.shape
    rgb = np.stack([gray, gray, gray], axis=-1)
    # Apply blue where mask is True (alpha blend ~50%)
    blue = np.zeros_like(rgb)
    blue[..., 2] = 255
    alpha = 0.5
    idx = mask_small
    rgb[idx] = (alpha * blue[idx] + (1 - alpha) * rgb[idx]).astype(np.uint8)

    Image.fromarray(rgb).save(out_png)
    log("Saved preview:", out_png)

def make_mask(pre_lin, post_lin, profile):
    pre_db, post_db = to_db(pre_lin), to_db(post_lin)
    diff = pre_db - post_db  # darker after flood → positive

    finite_post = np.isfinite(post_db)
    water_cut = min(float(np.percentile(post_db[finite_post], ADAPTIVE_WATER_P)), -12.0)  # cap at -12 dB
    water_like = (post_db < water_cut) & finite_post

    # Robust Otsu on a sampled subset
    ss = (slice(None, None, DOWNSAMPLE_STEP), slice(None, None, DOWNSAMPLE_STEP))
    submask = np.isfinite(diff[ss]) & water_like[ss]
    sub = diff[ss][submask]
    if sub.size < 1024:
        sub = diff[np.isfinite(diff)][:: max(1, DOWNSAMPLE_STEP * 8)]
    try:
        t = float(threshold_otsu(sub))
        if not (-40.0 <= t <= 10.0):
            raise ValueError("Otsu out of sane range")
        method = "otsu"
    except Exception:
        t = MANUAL_THRESH_DB
        method = "manual"

    log(f"Water cutoff (POST): {water_cut:.2f} dB | Threshold ({method}): {t:.2f} dB")

    raw = (diff > t) & water_like
    mask = closing(opening(raw, KERNEL), KERNEL)

    # Saving mask GeoTIFF 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        pr = profile.copy()
        pr.update(dtype="uint8", count=1, compress="lzw", tiled=True, blockxsize=512, blockysize=512)
        OUT_MASK.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(OUT_MASK, "w", **pr) as dst:
            dst.write(mask.astype("uint8"), 1)
    log("Saved:", OUT_MASK)

    save_fast_preview(post_db, mask, OUT_PNG, step=PREVIEW_STEP)

def main():
    log("Root:", ROOT)
    log("Unzipped:", UNZ)
    pol, pre_path, post_path = find_two_bands(UNZ)
    log(f"Using polarization: {pol}")
    log("Chosen PRE :", pre_path.name)
    log("Chosen POST:", post_path.name)

    pre, post, profile = align_post_to_pre(pre_path, post_path)
    log("Shapes:", pre.shape, post.shape)

    make_mask(pre, post, profile)

if __name__ == "__main__":
    main()