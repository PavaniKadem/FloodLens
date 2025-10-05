                                         About this Project

## FloodLens:

Floods are one of the common natural disasters that occur worldwide. Rapid identification of heavily affected areas is crucial for emergency response and resources allocation.
FloodLens is a geospatial AI tool that detects flooded areas from Radar imagery, visualizes and prioritizes the flood-affected areas using Sentinel-1 SAR data and machine Learning techniques.This empowers decision-makers by combining Radar remote sensing and AI-driven impact scoring into a seamless end-to-end pipeline.


## Project Overview:

FloodLens automates the following workflow:

1. Downloads Sentinel-1 SAR data (VV/VH polarization).
2. Detects flood zones using pre and post-flood radar images
3. Generate masks of affected areas
4. Visualizes results on an interactive map (Folium)
5. Score and rank patches using AI (XGBoost) based on area and impact.

## Project Structure

```
FloodLens
   — data                                   # results and output
         — flood_mask.tif 			        # binary mask of flooded pixels
	 — flood_preview.png                    # Grayscale images of flood areas
         — flood_overlay.png                # overlay of mask on baseman
         — flood_scores.csv                 # AI ranking scores      
         — floodlens_demo.html              # Interactive demo map
         — floodlens_ranked.html            # Ranked flood map with AI scoring 
         — map_overlay_demo.html            # overlay demo map (fallback)
         — map_overlay.html                 # overlay visualization  
	 — s1_unzipped                          # Unzipped sentinel-1 SAR product

   — src                                    # soruce code
         — download_two_iw.py               # Download 2 sentinel-1 IW images
         — mask_flood_min.py                # Apply thresholding to create flood mask
         — plot_mask.py                     # plot flood mask as PNG preview
         — map_overlay.py                   # overlay mask on interactive map
         — score_patches.py                 # AI scoring of flood patches
         — flood_demo.py                    # Synthetic demo flood generator
         — run_demo.py                      # one step execution of all files

   — requirements.txt                       # Python dependencies for project
   — README.md                              # project description
```

## Quick Start
1. Clone this repository:
   ```bash
   git clone https://github.com/PavaniKadem/FloodLens.git
   cd FloodLens
2. Create conda environment:
   ```bash
   conda create -n floodlens-py312 python=3.12
   conda activate floodlens-py312
3. Install dependencies
   ```bash
   pip install -r requirements.txt
4. Run demo pipeline:
   ```bash
   python src/run_demo.py
   
5. Open results:
   ```
   data/floodlens_demo.html -> Interactive demo map
   data/floodlens_ranked.html -> AI-ranked flood map
   

## Live Demo:
```
Interactive Flood Map: https://pavanikadem.github.io/FloodLens/
Ranked Flood Map (AI-Scoring): https://pavanikadem.github.io/FloodLens/floodlens_ranked.html