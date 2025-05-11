import numpy as np

class CellState:
    UNBURNED = 0
    BURNING = 1
    BURNED = 2

def initialize_grid(shape, fuel_load=1.0):
    """Initialize the simulation grid with fuel and unburned state."""
    grid = np.zeros(shape, dtype=[('state', 'i1'), ('fuel', 'f4'), ('ignition_time', 'f4')])
    grid['state'] = CellState.UNBURNED
    grid['fuel'] = fuel_load
    grid['ignition_time'] = -1
    return grid

def ignite(grid, x, y, time=0):
    """Set ignition point at (x, y)."""
    grid['state'][x, y] = CellState.BURNING
    grid['ignition_time'][x, y] = time
