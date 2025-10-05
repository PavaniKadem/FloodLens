# download_two_iw.py — FloodLens
import asf_search as asf
from shapely.geometry import box
from pathlib import Path
import subprocess
import os

# Root & folders
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UNZIP = DATA / "s1_unzipped"
DATA.mkdir(parents=True, exist_ok=True)
UNZIP.mkdir(parents=True, exist_ok=True)

# Houston bounding box
aoi_wkt = box(-95.90, 29.45, -95.00, 30.15).wkt
date_start, date_end = "2024-06-01", "2024-12-31"

print("[dl] Searching Sentinel-1 IW/GRD_HD over Houston…")
results = asf.search(
    platform=[asf.PLATFORM.SENTINEL1A, asf.PLATFORM.SENTINEL1B],
    processingLevel="GRD_HD",
    beamMode="IW",
    intersectsWith=aoi_wkt,
    start=date_start,
    end=date_end,
)

lst = list(results)
if len(lst) < 2:
    raise SystemExit(f"[dl] Need >=2 IW/GRD_HD scenes, got {len(lst)}. Expand dates.")

# Sort by start time
def key(granule):
    return granule.properties.get("startTime") or ""

lst.sort(key=key)
picks = [lst[0], lst[-1]]

print("[dl] Selected:")
for g in picks:
    print("   ", key(g), g.properties.get("granuleName"))

# Auth with Earthdata token
token = os.environ.get("EARTHDATA_TOKEN")
if not token:
    raise SystemExit("Set EARTHDATA_TOKEN first (export EARTHDATA_TOKEN=...)")

session = asf.ASFSession().auth_with_token(token)

# Download
for g in picks:
    name = g.properties.get("granuleName")
    print("[dl] Downloading:", name)
    g.download(path=str(DATA), session=session)

# Unzip
print("[dl] Unzipping into:", UNZIP)
subprocess.run(f'unzip -n "{DATA}/*.zip" -d "{UNZIP}"', shell=True, check=False)
print("[dl] Done.")
