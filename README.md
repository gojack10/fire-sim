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

The primary way to run simulations is via the `tests/sample_notebook_script.py` script. Ensure your virtual environment is activated before running.

**Basic Command Structure:**
```bash
.venv/bin/python tests/sample_notebook_script.py [TERRAIN_OPTION] [OTHER_PARAMETERS]
```

**Terrain Options (choose one, required):**
- `--dem_file /path/to/your/dem.tif`: Use a specific DEM file.
- `--all_dems`: Run simulations for all `.tif` files in the `data/` directory (default: `/Users/jack10/coding-projects/weather-sim/data/`, configurable with `--data_dir`).
- `--random_terrain`: Use randomly generated Perlin noise terrain.
- `--flat_terrain`: Use a completely flat terrain (all elevations are 0).

**Common Parameters (Optional):**
- `--grid_rows <int>`: Number of rows in the grid (default: 100).
- `--grid_cols <int>`: Number of columns in the grid (default: 100).
- `--max_steps <int>`: Maximum simulation steps (default: 50).
- `--ignition_row <int>`: Row index for initial ignition (default: 50, or center).
- `--ignition_col <int>`: Column index for initial ignition (default: 50, or center).
- `--base_prob <float>`: Base probability of spread (default: 0.58).
- `--wind_speed <float>`: Wind speed (e.g., m/s). Requires `--wind_direction`.
- `--wind_direction <float>`: Wind direction in degrees (0=N, 90=E). Requires `--wind_speed`.
- `--wind_strength <float>`: Factor to modify wind effect (default: 1.0).
- `--slope_coefficient <float>`: Coefficient for slope effect (default: 0.1).
- `--cell_resolution <float>`: Cell resolution in meters (default: 30.0).
- `--output_prefix <str>`: Prefix for output plot filenames (default: `simulation_output`).

**Examples:**

1.  **Run with a specific DEM file from the `data` directory:**
    ```bash
    .venv/bin/python tests/sample_notebook_script.py --dem_file data/n00_e009_1arc_v3.tif --grid_rows 150 --grid_cols 150 --ignition_row 75 --ignition_col 75 --max_steps 100
    ```

2.  **Run with randomly generated terrain:**
    ```bash
    .venv/bin/python tests/sample_notebook_script.py --random_terrain --grid_rows 50 --grid_cols 50 --max_steps 30
    ```

3.  **Run for all DEM files in the default `data` directory:**
    ```bash
    .venv/bin/python tests/sample_notebook_script.py --all_dems --grid_rows 120 --grid_cols 120 --max_steps 60 --slope_coefficient 0.15
    ```

4.  **Run with flat terrain and wind:**
    ```bash
    .venv/bin/python tests/sample_notebook_script.py --flat_terrain --grid_rows 100 --grid_cols 100 --max_steps 50 --wind_speed 10 --wind_direction 90 --wind_strength 1.5
    ```

Output plots (PNG images) will be saved in the project's root directory (`/Users/jack10/coding-projects/weather-sim/`).

## Development

- **Simulation Logic:** `src/simulation.py`
- **Tests:** `tests/test_simulation.py` (run with `pytest`)
- **Main Script:** `tests/sample_notebook_script.py`

## Future Work (see `planning.md`)

- Interactive Jupyter Notebook interface using `ipywidgets`.
- Dynamic animations of fire spread.
- More complex fuel modeling and weather inputs.
