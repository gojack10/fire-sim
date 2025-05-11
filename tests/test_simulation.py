import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src import simulation

def test_initialize_grid():
    shape = (4, 4)
    grid = simulation.initialize_grid(shape, fuel_load=2.0, moisture=0.5)
    assert grid.shape == shape
    assert np.all(grid['state'] == simulation.CellState.UNBURNED)
    assert np.allclose(grid['fuel'], 2.0)
    assert np.allclose(grid['moisture'], 0.5)
    assert np.all(grid['ignition_time'] == -1)

def test_initialize_grid_arrays():
    shape = (2, 2)
    fuel = np.array([[1, 2],[3, 4]], dtype=np.float32)
    moist = np.array([[0.1, 0.2],[0.3, 0.4]], dtype=np.float32)
    grid = simulation.initialize_grid(shape, fuel_load=fuel, moisture=moist)
    assert np.allclose(grid['fuel'], fuel)
    assert np.allclose(grid['moisture'], moist)

def test_ignite():
    grid = simulation.initialize_grid((3, 3))
    simulation.ignite(grid, 1, 1, time=5)
    assert grid['state'][1, 1] == simulation.CellState.BURNING
    assert grid['ignition_time'][1, 1] == 5

def test_step_basic_spread():
    grid = simulation.initialize_grid((3, 3))
    simulation.ignite(grid, 1, 1, time=0)
    n_new = simulation.step(grid, time=1)
    # Should ignite 8 neighbors (cardinal and diagonal)
    assert n_new == 8
    burning = np.argwhere(grid['state'] == simulation.CellState.BURNING)
    assert set(map(tuple, burning)) == {(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)}
    # Center should be burned
    assert grid['state'][1, 1] == simulation.CellState.BURNED

def test_step_high_moisture_blocks_spread():
    grid = simulation.initialize_grid((3, 3), fuel_load=1.0, moisture=0.9)
    simulation.ignite(grid, 1, 1, time=0)
    n_new = simulation.step(grid, time=1, moisture_threshold=0.3)
    # No new ignitions because moisture is too high
    assert n_new == 0
    # Center should be burned
    assert grid['state'][1, 1] == simulation.CellState.BURNED

def test_step_low_fuel_blocks_spread():
    grid = simulation.initialize_grid((3, 3), fuel_load=0.0, moisture=0.0)
    # Set center cell to have fuel so it can ignite
    grid['fuel'][1, 1] = 1.0
    simulation.ignite(grid, 1, 1, time=0)
    n_new = simulation.step(grid, time=1)
    # No new ignitions because neighbors have zero fuel
    assert n_new == 0
    # Center should be burned
    assert grid['state'][1, 1] == simulation.CellState.BURNED

def test_run_simulation():
    grid = simulation.initialize_grid((3, 3))
    simulation.ignite(grid, 1, 1, time=0)
    history = simulation.run_simulation(grid, max_steps=10)
    # It should run until all cells are burned or max_steps reached
    # In a 3x3 grid, fire should spread to all cells
    final = history[-1]
    assert np.all(final['state'] != simulation.CellState.BURNING)
    # All cells except corners should be burned
    assert np.count_nonzero(final['state'] == simulation.CellState.BURNED) >= 5
    # History should have at least as many steps as needed to burn out
    assert len(history) > 1

def test_burned_area_over_time():
    grid = simulation.initialize_grid((3, 3))
    simulation.ignite(grid, 1, 1, time=0)
    history = simulation.run_simulation(grid, max_steps=10)
    burned = simulation.burned_area_over_time(history)
    # Burned area should be monotonically increasing
    assert np.all(np.diff(burned) >= 0)
    # First time step should have 0 burned, then increase
    assert burned[0] == 0
    assert burned[-1] == np.count_nonzero(history[-1]['state'] == simulation.CellState.BURNED)

def test_wind_effect_east():
    import numpy as np
    np.random.seed(42)
    grid_size = 11
    steps = 10
    ignition_x = grid_size // 2
    ignition_y = grid_size // 2
    grid_east_wind = simulation.initialize_grid((grid_size, grid_size), fuel_load=1.0, moisture=0.0)
    simulation.ignite(grid_east_wind, ignition_x, ignition_y, time=0)
    history_east_wind = simulation.run_simulation(grid_east_wind, max_steps=steps, wind=(3.0, 0.0), wind_strength=2.0)
    final_east_wind = history_east_wind[-1]
    # Count burned cells east vs. west of ignition point
    east_burned = np.count_nonzero(final_east_wind['state'][:, ignition_y+1:] == simulation.CellState.BURNED)
    west_burned = np.count_nonzero(final_east_wind['state'][:, :ignition_y] == simulation.CellState.BURNED)
    if east_burned <= west_burned:
        print("east_burned:", east_burned, "west_burned:", west_burned)
        print(final_east_wind['state'])
    assert east_burned > west_burned

def test_initialize_grid_with_random_elevation():
    grid_size = (5, 5)
    grid = simulation.initialize_grid(grid_size, elevation_source="random")
    assert 'elevation' in grid.dtype.names
    # Check that elevation is not flat (i.e., some variation exists)
    assert np.std(grid['elevation']) > 1e-6 # Allow for very minor variation if noise is weak
    # Check that elevation values are within a reasonable range (e.g., 0 to max_elevation used in _generate_random_terrain)
    assert np.all(grid['elevation'] >= 0)
    assert np.all(grid['elevation'] <= 1000.0) # Assuming default max_elevation

def test_initialize_grid_with_flat_elevation():
    grid_size = (5, 5)
    grid = simulation.initialize_grid(grid_size, elevation_source=None)
    assert 'elevation' in grid.dtype.names
    assert np.all(grid['elevation'] == 0.0)

# Placeholder for DEM file testing - requires a sample DEM file
# For now, we can skip this or use one of the user's files if path is known and stable
# def test_initialize_grid_with_dem_file():
#     grid_size = (10, 10) # Or match the DEM's aspect ratio if desired
#     # IMPORTANT: This path needs to be valid and the DEM small enough for quick test.
#     # Consider adding a small test DEM to the repo.
#     dem_path = "/Users/jack10/coding-projects/weather-sim/data/n00_e009_1arc_v3.tif" 
#     try:
#         grid = simulation.initialize_grid(grid_size, elevation_source=dem_path)
#         assert 'elevation' in grid.dtype.names
#         assert grid['elevation'].shape == grid_size
#         # Check that elevation data is not all zeros (assuming DEM has variation)
#         assert np.any(grid['elevation'] != 0.0) 
#     except Exception as e:
#         # This might happen if rasterio or its deps are not installed, or file is missing
#         pytest.skip(f"DEM loading test skipped: {e}")

def test_slope_effect_uphill_faster():
    # Create a 5x3 grid with a simple linear slope uphill (North)
    # Row 0: high elevation, Row 4: low elevation
    grid_size = (5, 3)
    np.random.seed(43) # For probabilistic spread consistency
    
    # Create elevation data: Row 0 = 40m, Row 1 = 30m, ..., Row 4 = 0m
    elevation_data = np.array([
        [40, 40, 40],
        [30, 30, 30],
        [20, 20, 20],
        [10, 10, 10],
        [0,  0,  0] 
    ], dtype='f4')

    grid = simulation.initialize_grid(grid_size, fuel_load=1.0, moisture=0.0, elevation_source=elevation_data, cell_resolution=10.0)
    
    # Ignite at the bottom of the slope (row 4, middle cell)
    simulation.ignite(grid, 4, 1, time=0)
    
    # Run for a few steps with slope effect enabled, no wind
    # High slope_coefficient to make effect obvious, low base_prob to prevent saturation
    history = simulation.run_simulation(grid, max_steps=3, base_prob=0.6, slope_coefficient=1.0, random_seed=43)
    final_grid_state = history[-1]['state']

    # Expected: Fire should have spread more uphill (towards row 0) than downhill or sideways
    # Burning cells at row 4 (initial): grid[4,1] was initial ignition
    # Burning cells at row 3 (uphill): should be burning
    # Burning cells at row 2 (further uphill): potentially burning
    # Sideways spread at row 4 (grid[4,0], grid[4,2]) should be less likely or not occurred compared to uphill

    burned_uphill_row3 = np.count_nonzero(final_grid_state[3,:] == simulation.CellState.BURNED) + \
                         np.count_nonzero(final_grid_state[3,:] == simulation.CellState.BURNING)
    burned_uphill_row2 = np.count_nonzero(final_grid_state[2,:] == simulation.CellState.BURNED) + \
                         np.count_nonzero(final_grid_state[2,:] == simulation.CellState.BURNING)
    
    # Initial ignition point will be BURNED
    initial_ignition_burned = (final_grid_state[4,1] == simulation.CellState.BURNED)
    assert initial_ignition_burned, "Initial ignition point should be burned"

    # Check that fire has spread significantly uphill
    # Given strong slope, it should have reached row 3
    assert burned_uphill_row3 > 0, "Fire should spread uphill to row 3"

    # Optional: Check that it spread more uphill than sideways or not at all downhill (beyond initial)
    # Sideways spread at initial elevation (row 4)
    burned_sideways_row4_left = (final_grid_state[4,0] == simulation.CellState.BURNED) or (final_grid_state[4,0] == simulation.CellState.BURNING)
    burned_sideways_row4_right = (final_grid_state[4,2] == simulation.CellState.BURNED) or (final_grid_state[4,2] == simulation.CellState.BURNING)
    
    # Sum of cells burned/burning uphill from initial ignition row
    total_uphill_spread_count = burned_uphill_row3 + burned_uphill_row2 # + burned_uphill_row1 + burned_uphill_row0 if checking further
    
    # Sum of cells burned/burning sideways from initial ignition cell
    total_sideways_spread_count = (1 if burned_sideways_row4_left else 0) + (1 if burned_sideways_row4_right else 0)
    
    # If the fire spreads to row 2, it's a strong indication of uphill preference.    
    # A simple check: more cells burned/burning above the initial row than beside it on the initial row.
    # This is a heuristic, exact numbers depend on base_prob and steps.
    print(f"Final grid state:\n{final_grid_state}")
    print(f"Uphill (R3): {burned_uphill_row3}, Uphill (R2): {burned_uphill_row2}, Sideways (R4 L/R): {burned_sideways_row4_left}/{burned_sideways_row4_right}")
    assert total_uphill_spread_count > total_sideways_spread_count, "Fire should spread more uphill than sideways on a steep slope with low base probability and high slope coefficient."
