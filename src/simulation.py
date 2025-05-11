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

def burned_area_over_time(history):
    """
    Compute the burned area (number of burned cells) at each time step.
    Args:
        history (list of np.ndarray): simulation grid snapshots (from run_simulation)
    Returns:
        np.ndarray: burned area at each time step
    """
    return np.array([
        np.count_nonzero(grid['state'] == CellState.BURNED)
        for grid in history
    ])

def plot_burned_area(*burned_areas, labels=None, save_path=None):
    """
    Plot one or more burned area curves vs. time using matplotlib.
    Args:
        *burned_areas: one or more arrays of burned area (for comparison)
        labels (list of str, optional): labels for each curve
        save_path (str, optional): if provided, save plot as PNG to this path
    """
    import matplotlib.pyplot as plt
    plt.figure()
    for i, burned in enumerate(burned_areas):
        label = labels[i] if labels and i < len(labels) else None
        plt.plot(burned, marker='o', label=label)
    plt.xlabel('Time Step')
    plt.ylabel('Burned Area (cells)')
    plt.title('Burned Area vs. Time')
    plt.grid(True)
    if labels:
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def export_burned_area_csv(burned_area, filename):
    """
    Export burned area data to a CSV file.
    Args:
        burned_area (array-like): burned area at each time step
        filename (str): path to output CSV file
    """
    import numpy as np
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_step', 'burned_area'])
        for t, area in enumerate(burned_area):
            writer.writerow([t, area])
