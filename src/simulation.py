import numpy as np

class CellState:
    """Enum-like class for cell states."""
    UNBURNED = 0
    BURNING = 1
    BURNED = 2

def initialize_grid(shape, fuel_load=1.0, moisture=0.2):
    """
    Initialize the simulation grid with fuel and unburned state.
    Args:
        shape (tuple): (rows, cols) of the grid
        fuel_load (float or np.ndarray): initial fuel value(s) for all cells
        moisture (float or np.ndarray): initial moisture value(s) for all cells
    Returns:
        np.ndarray: structured array with fields 'state', 'fuel', 'moisture', 'ignition_time'
    """
    grid = np.zeros(shape, dtype=[('state', 'i1'), ('fuel', 'f4'), ('moisture', 'f4'), ('ignition_time', 'f4')])
    grid['state'] = CellState.UNBURNED
    if isinstance(fuel_load, np.ndarray):
        grid['fuel'] = fuel_load
    else:
        grid['fuel'] = fuel_load
    if isinstance(moisture, np.ndarray):
        grid['moisture'] = moisture
    else:
        grid['moisture'] = moisture
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

def step(grid, time=0, moisture_threshold=0.3):
    """
    Perform one simulation step: burning cells ignite adjacent unburned cells.
    Spread depends on both fuel and moisture (cell ignites only if fuel > 0 and moisture < threshold).
    Args:
        grid (np.ndarray): simulation grid
        time (int): current time step
        moisture_threshold (float): maximum moisture for ignition
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
                        if (
                            grid['state'][nx, ny] == CellState.UNBURNED
                            and grid['fuel'][nx, ny] > 0
                            and grid['moisture'][nx, ny] < moisture_threshold
                        ):
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

def run_simulation(grid, max_steps=100):
    """
    Run the simulation for up to max_steps or until no burning cells remain.
    Args:
        grid (np.ndarray): simulation grid (will be modified in place)
        max_steps (int): maximum number of time steps
    Returns:
        list of np.ndarray: snapshots of grid states at each time step
    """
    history = [grid.copy()]
    for t in range(1, max_steps+1):
        n_new = step(grid, time=t)
        history.append(grid.copy())
        if np.count_nonzero(grid['state'] == CellState.BURNING) == 0:
            break
    return history
