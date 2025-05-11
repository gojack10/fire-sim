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
    # Should ignite 4 neighbors
    assert n_new == 4
    burning = np.argwhere(grid['state'] == simulation.CellState.BURNING)
    assert set(map(tuple, burning)) == {(0,1), (2,1), (1,0), (1,2)}
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
