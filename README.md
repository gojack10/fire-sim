# Wildfire Simulation Project

This project simulates wildfire spread across a 2D grid, incorporating environmental factors such as terrain elevation (slope effects) and wind.

## Features

- **Grid-Based Simulation:** Fire spreads across a 2D grid of cells.
- **Cell States:** Cells can be unburned, burning, or burned out.
- **Elevation Data:**
    - Load terrain elevation from GeoTIFF (.tif) DEM files.
    - Generate random terrain using Perlin noise for testing.
    - Support for flat terrain.
- **Slope Effects:** Fire spread probability is influenced by the terrain slope between adjacent cells (fire spreads faster uphill).
- **Wind Effects:** Fire spread is influenced by wind speed and direction.
- **Configurable Parameters:** Grid size, ignition point, simulation duration, base spread probability, wind parameters, slope coefficient, and cell resolution can be configured.
- **Visualization:** Outputs a PNG image showing the final grid state (burn status) and the elevation map for each simulation run.
- **Command-Line Interface:** A script (`tests/sample_notebook_script.py`) allows for running simulations with various configurations.

For a detailed project plan, including future development ideas and a breakdown of components, please see `planning.md`.

## Setup

**Prerequisites:**
- Python 3.10+ (as per testing environment)
- [Homebrew](https://brew.sh/) (for macOS users, to install `uv`)
- `uv` (Python package manager - recommended)

**1. Clone the Repository (if you haven't already):**
```bash
git clone <your-repository-url>
cd weather-sim
```

**2. Install `uv` (if not already installed):**
```bash
# Using Homebrew on macOS
brew install uv

# Other systems (see uv documentation: https://github.com/astral-sh/uv)
pip install uv
```

**3. Create and Activate a Virtual Environment (using `uv`):**
```bash
# Create a virtual environment named .venv in the project root
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (Git Bash or similar):
# source .venv/Scripts/activate
```

**4. Install Dependencies (using `uv`):**
With the virtual environment activated, install the required packages:
```bash
uv pip install -r requirements.txt
```

## Running Simulations