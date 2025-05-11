import numpy as np

class CellState:
    """Enum-like class for cell states."""
    UNBURNED = 0
    BURNING = 1
    BURNED = 2

def initialize_grid(shape, fuel_load=1.0):
    """
    Initialize the simulation grid with fuel and unburned state.
    Args:
        shape (tuple): (rows, cols) of the grid
        fuel_load (float): initial fuel value for all cells
    Returns:
        np.ndarray: structured array with fields 'state', 'fuel', 'ignition_time'
    """
    grid = np.zeros(shape, dtype=[('state', 'i1'), ('fuel', 'f4'), ('ignition_time', 'f4')])
    grid['state'] = CellState.UNBURNED
    grid['fuel'] = fuel_load
    grid['ignition_time'] = -1
    return grid

def ignite(grid, x, y, time=0):
    """
    Set ignition point at (x, y).
    Args:
        grid (np.ndarray): simulation grid
        x (int): row index
        y (int): column index
        time (int): ignition time step
    """
    grid['state'][x, y] = CellState.BURNING
    grid['ignition_time'][x, y] = time

def step(grid, time=0):
    """
    Perform one simulation step: burning cells ignite adjacent unburned cells.
    Args:
        grid (np.ndarray): simulation grid
        time (int): current time step
    Returns:
        int: number of new ignitions
    """
    new_ignitions = []
    rows, cols = grid.shape
    for x in range(rows):
        for y in range(cols):
            if grid['state'][x, y] == CellState.BURNING:
                # Ignite adjacent unburned cells (N, S, E, W)
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        if grid['state'][nx, ny] == CellState.UNBURNED and grid['fuel'][nx, ny] > 0:
                            new_ignitions.append((nx, ny))
                # Burned out after this step
                grid['state'][x, y] = CellState.BURNED
    for nx, ny in new_ignitions:
        grid['state'][nx, ny] = CellState.BURNING
        grid['ignition_time'][nx, ny] = time
    return len(new_ignitions)

def print_grid(grid):
    """
    Print the grid state to the terminal using ASCII symbols.
    """
    chars = {CellState.UNBURNED: '.', CellState.BURNING: '*', CellState.BURNED: 'x'}
    for row in grid['state']:
        print(''.join(chars[val] for val in row))
