# OCTA Microvascular Network Analysis Pipeline

## Overview

This repository contains an automated Python data science pipeline designed to extract, quantify, and visualize topological and structural metrics from *en face* Optical Coherence Tomography Angiography (OCTA) images.

Originally developed to analyze *in vivo* cutaneous microvascular responses to ultraviolet B (UVB) irradiation and topical treatments, this tool converts raw microvascular scans into mathematical graphs. By measuring vascular density, branching complexity, and vessel thickness, it provides a robust, reproducible framework for assessing microvascular remodeling and hyperperfusion in biological tissues.

## Core Features

* **Vessel Enhancement & Segmentation:** Utilizes a Frangi vesselness filter and Otsu's thresholding to accurately isolate vascular structures from background tissue noise.
* **Graph Extraction & Topology:** Converts skeletonized vessel maps into `NetworkX` graphs using the `sknw` library. Automatically extracts key topological metrics including:
* Total Node Count (Branching points)
* Total Edge Count (Vessel segments)
* Total Vessel Length (Perfusion area)
* Average Node Degree (Network complexity)


* **Thickness Quantification:** Employs the Euclidean Distance Transform (EDT) via `SciPy` to calculate precise radii along the skeletonized centerlines, outputting the average, median, and maximum vessel thicknesses.
* **Visualization Modules:** * **Image Overlays:** Automatically generates a side-by-side 1x3 plot showing the original OCTA scan, the skeletonized map, and a high-contrast network graph overlay.
* **Statistical Charts:** Includes a secondary module built with `matplotlib` to generate publication-ready grouped bar charts for comparing temporal percentage changes across multiple experimental cohorts.



## Dependencies

This pipeline requires Python 3.x. Install the required libraries using `pip`:

```bash
pip install numpy matplotlib scikit-image networkx scipy sknw

```

## Usage

### 1. Single Image Analysis

You can process an individual OCTA image by calling the main analysis function. This will print the computed metrics to your console and display the extraction visualization.

```python
from uv_vascular_analysis import analyze_octa_network

# Analyze a single image and return the generated graph
graph = analyze_octa_network('path/to/OCTA_image.png')

```

### 2. Batch Processing

The script is lightweight and can be easily wrapped in a loop for sequential processing of an entire dataset or time-series directory:

```python
# Example: Processing baseline and 48-hour follow-up images
analyze_octa_network('data/baseline_control.png')
analyze_octa_network('data/48h_control.png')

```

### 3. Generating Temporal Change Charts

To visualize the percentage changes of the extracted metrics across different experimental groups (e.g., Control, UV, L-NAME, UV + L-NAME), use the included charting script.

1. Update the `pct_nodes`, `pct_edges`, and `pct_vessel_length` arrays in the script with the quantitative results derived from your image analysis.
2. Execute the block to automatically render and export a high-resolution (300 DPI) chart (`percentage_changes_chart.png`) suitable for academic publication.

```python
# Simply run the charting section of the script after updating your data arrays
python generate_charts.py

```

---

Would you like me to add a "Data Structure" or "Output format" section to explain how to save the terminal outputs directly into a CSV file for your statistical analysis software?
