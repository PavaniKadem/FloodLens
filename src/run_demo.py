# run_demo.py — FloodLens end-to-end run
from __future__ import annotations
import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
DATA = ROOT / "data"
UNZ  = DATA / "s1_unzipped"

PY_EXE = sys.executable  

SCRIPTS = {
    "download": SRC / "download_two_iw.py",
    "mask":     SRC / "mask_flood_min.py",
    "preview":  SRC / "plot_mask.py",
    "overlay":  SRC / "map_overlay.py",
    "score":    SRC / "score_patches.py",
    "demo":     SRC / "run_demo.py",
}

OUTPUTS = {
    "mask_tif": DATA / "flood_mask.tif",
    "preview_png": DATA / "flood_preview.png",
    "overlay_html": DATA / "map_overlay.html",
    "ranked_html": DATA / "floodlens_ranked.html",
    "scores_csv": DATA / "flood_scores.csv",
}

def run_step(args: List[str], label: str) -> None:
    print(f"\n[run] ▶ {label}: {' '.join(args)}")
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[run] Step failed: {label}\n    {e}")
        sys.exit(2)

def needs_download() -> bool:
    """Heuristic: if no SAFE or ZIP under data/s1_unzipped and no mask file, try to download."""
    if OUTPUTS["mask_tif"].exists():
        return False
    if not UNZ.exists():
        return True
    # look for any Sentinel-1 SAFE directories or zips
    any_safe = any(p.suffix.lower() == ".safe" or p.name.endswith(".SAFE") for p in UNZ.iterdir()) if UNZ.exists() else False
    any_zip  = any(p.suffix.lower() == ".zip" for p in UNZ.iterdir()) if UNZ.exists() else False
    return not (any_safe or any_zip)

def summarize():
    print("\n[run] ✅ Pipeline finished. Artifacts:")
    for k, p in OUTPUTS.items():
        status = "✓" if p.exists() else "—"
        print(f"   {status} {p.relative_to(ROOT)}")
    print("\nOpen in a browser (macOS):")
    for key in ("overlay_html", "ranked_html"):
        p = OUTPUTS[key]
        if p.exists():
            print(f"  open {p}")

def main():
    parser = argparse.ArgumentParser(description="FloodLens end-to-end runner")
    parser.add_argument("--download", action="store_true",
                        help="Force run download step even if data/s1_unzipped is populated")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download step even if no data is present")
    parser.add_argument("--demo-fallback", action="store_true", default=True,
                        help="If real mask not produced, render demo polygons (default on)")
    parser.add_argument("--no-demo-fallback", dest="demo_fallback", action="store_false",
                        help="Disable demo fallback")
    args = parser.parse_args()

    print("[run] FloodLens root:", ROOT)

    # ensure folders
    DATA.mkdir(parents=True, exist_ok=True)
    UNZ.mkdir(parents=True, exist_ok=True)

    # Download if appropriate
    run_dl = False
    if args.download:
        run_dl = True
    elif args.no_download:
        run_dl = False
    else:
        run_dl = needs_download()

    if run_dl:
        print("[run] Decided to download Sentinel-1 IW scenes (VV/VH).")
        print("      Tip: to avoid interactive login, export EARTHDATA_USER and EARTHDATA_PASS beforehand.")
        run_step([PY_EXE, "-u", str(SCRIPTS["download"])], "download_two_iw")
    else:
        print("[run] Skipping download step (data appears present).")

    # Build flood mask (real pipeline)
    run_step([PY_EXE, "-u", str(SCRIPTS["mask"])], "mask_flood_min")

    # Generate quicklook preview PNG
    run_step([PY_EXE, "-u", str(SCRIPTS["preview"])], "plot_mask")

    # Overlay on interactive basemap
    run_step([PY_EXE, "-u", str(SCRIPTS["overlay"])], "map_overlay")

    #AI score & rank patches (works with real polygons or synthetic fallback)
    run_step([PY_EXE, "-u", str(SCRIPTS["score"])], "score_patches")

    # If real overlay didn’t appear and user allows fallback, generate demo page too
    if args.demo_fallback and not OUTPUTS["overlay_html"].exists():
        print("[run] Real overlay not found. Creating demo page with synthetic ellipses…")
        run_step([PY_EXE, "-u", str(SCRIPTS["demo"])], "run_demo")

    summarize()

if __name__ == "__main__":
    main()