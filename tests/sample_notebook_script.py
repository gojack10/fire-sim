import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from ..src import simulation # Use relative import if this script is a module

# If running as a standalone script, you might need to adjust sys.path
# import sys
# if ".." not in sys.path:
#     sys.path.append("..")
# from src import simulation

DEFAULT_DATA_DIR = "/Users/jack10/coding-projects/weather-sim/data"

def run_single_simulation(elevation_src, grid_rows, grid_cols, ignition_row, ignition_col, 
                          max_steps, base_prob, wind_speed, wind_direction, wind_strength, 
                          slope_coefficient, cell_resolution, output_filename_prefix):
    """Runs a single simulation and saves plots of final state and elevation."""
    print(f"\n--- Running simulation for: {elevation_src if isinstance(elevation_src, str) else 'custom_array_elevation'} ---")
    print(f"Parameters: Grid={grid_rows}x{grid_cols}, Ignition=({ignition_row},{ignition_col}), Steps={max_steps}")
    print(f"Wind: Speed={wind_speed}, Dir={wind_direction}, Strength={wind_strength}")
    print(f"Slope Coeff={slope_coefficient}, Cell Res={cell_resolution}m")

    grid_size = (grid_rows, grid_cols)

    try:
        initial_grid = simulation.initialize_grid(
            grid_size,
            elevation_source=elevation_src,
            cell_resolution=cell_resolution
        )
    except Exception as e:
        print(f"Error initializing grid with {elevation_src}: {e}")
        return

    simulation.ignite(initial_grid, ignition_row, ignition_col, time=0)

    wind_tuple = (wind_speed, wind_direction) if wind_speed is not None and wind_direction is not None else None

    history = simulation.run_simulation(
        initial_grid,
        max_steps=max_steps,
        base_prob=base_prob,
        wind=wind_tuple,
        wind_strength=wind_strength,
        slope_coefficient=slope_coefficient,
        cell_resolution=cell_resolution,
        random_seed=42 # Consistent seed for comparable runs
    )

    final_grid = history[-1]

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    simulation.plot_grid_state(final_grid, ax=axs[0], title=f"Final State ({os.path.basename(str(elevation_src))})")
    simulation.plot_grid_property(final_grid, 'elevation', ax=axs[1], title=f"Elevation ({os.path.basename(str(elevation_src))})", cmap='terrain')

    plt.tight_layout()
    output_plot_filename = f"{output_filename_prefix}_{os.path.basename(str(elevation_src)).replace('.tif','')}_results.png"
    if elevation_src == "random":
        output_plot_filename = f"{output_filename_prefix}_random_results.png"
    elif not isinstance(elevation_src, str):
        output_plot_filename = f"{output_filename_prefix}_custom_array_results.png"

    plt.savefig(output_plot_filename)
    print(f"Saved plot to {output_plot_filename}")
    plt.close(fig)

    burned_count = np.count_nonzero(final_grid['state'] == simulation.CellState.BURNED)
    print(f"Total cells burned: {burned_count}")
    print(f"--- Simulation complete for: {elevation_src if isinstance(elevation_src, str) else 'custom_array_elevation'} ---")

def main():
    parser = argparse.ArgumentParser(description="Run wildfire simulations with various terrain sources.")

    terrain_group = parser.add_mutually_exclusive_group(required=True)
    terrain_group.add_argument("--dem_file", type=str, help="Path to a specific DEM .tif file.")
    terrain_group.add_argument("--all_dems", action="store_true", help=f"Run simulation for all .tif files in {DEFAULT_DATA_DIR}.")
    terrain_group.add_argument("--random_terrain", action="store_true", help="Use randomly generated terrain.")
    terrain_group.add_argument("--flat_terrain", action="store_true", help="Use flat terrain (all zeros elevation).")

    parser.add_argument("--grid_rows", type=int, default=100, help="Number of rows in the grid.")
    parser.add_argument("--grid_cols", type=int, default=100, help="Number of columns in the grid.")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum simulation steps.")
    parser.add_argument("--ignition_row", type=int, default=50, help="Row index for initial ignition.")
    parser.add_argument("--ignition_col", type=int, default=50, help="Column index for initial ignition.")
    
    parser.add_argument("--base_prob", type=float, default=0.58, help="Base probability of spread.") # Adjusted to match default in simulation.py step
    parser.add_argument("--wind_speed", type=float, default=None, help="Wind speed (e.g., m/s). Requires --wind_direction.")
    parser.add_argument("--wind_direction", type=float, default=None, help="Wind direction in degrees (0=N, 90=E). Requires --wind_speed.")
    parser.add_argument("--wind_strength", type=float, default=1.0, help="Factor to modify wind effect.")
    
    parser.add_argument("--slope_coefficient", type=float, default=0.1, help="Coefficient for slope effect.")
    parser.add_argument("--cell_resolution", type=float, default=30.0, help="Cell resolution in meters (e.g., for DEMs).")

    parser.add_argument("--output_prefix", type=str, default="simulation_output", help="Prefix for output plot filenames.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing DEM files for --all_dems.")

    args = parser.parse_args()

    # Validate ignition points against grid size
    if not (0 <= args.ignition_row < args.grid_rows and 0 <= args.ignition_col < args.grid_cols):
        parser.error(f"Ignition point ({args.ignition_row},{args.ignition_col}) is outside grid dimensions ({args.grid_rows}x{args.grid_cols}).")

    elevation_sources_to_run = []

    if args.dem_file:
        if not os.path.isfile(args.dem_file):
            parser.error(f"Specified DEM file not found: {args.dem_file}")
        elevation_sources_to_run.append(args.dem_file)
    elif args.all_dems:
        dem_files = glob.glob(os.path.join(args.data_dir, "*.tif"))
        if not dem_files:
            print(f"No .tif files found in {args.data_dir}. Consider using --dem_file or --random_terrain.")
            return
        elevation_sources_to_run.extend(sorted(dem_files))
        print(f"Found DEM files to process: {elevation_sources_to_run}")
    elif args.random_terrain:
        elevation_sources_to_run.append("random")
    elif args.flat_terrain:
        elevation_sources_to_run.append(None) # None results in flat terrain in initialize_grid

    for elev_src in elevation_sources_to_run:
        run_single_simulation(
            elevation_src=elev_src,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            ignition_row=args.ignition_row,
            ignition_col=args.ignition_col,
            max_steps=args.max_steps,
            base_prob=args.base_prob,
            wind_speed=args.wind_speed,
            wind_direction=args.wind_direction,
            wind_strength=args.wind_strength,
            slope_coefficient=args.slope_coefficient,
            cell_resolution=args.cell_resolution,
            output_filename_prefix=args.output_prefix
        )

if __name__ == "__main__":
    # Quick check for relative import if running as script.
    # This assumes 'src' and 'tests' (or wherever this script is) are siblings.
    if __package__ is None or __package__ == '':
        import sys
        module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if module_path not in sys.path:
            sys.path.append(module_path)
        from src import simulation
    main()
