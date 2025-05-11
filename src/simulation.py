import numpy as np
import enum
import math
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import zoom # Added for resampling

class CellState(enum.IntEnum):
    UNBURNED = 0
    BURNING = 1
    BURNED = 2

grid_dtype = np.dtype([
    ('fuel', 'f4'),         # kg/m^2
    ('moisture', 'f4'),     # % (0-1)
    ('state', 'i1'),        # CellState
    ('ignition_time', 'i4'), # time step of ignition
    ('elevation', 'f4')     # meters
])

def _generate_random_terrain(rows, cols, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, random_seed=None, max_elevation=1000.0):
    if random_seed is not None:
        # Note: perlin_noise library itself doesn't have a direct seed function for pnoise2.
        # For true reproducibility with pnoise2, one might need to use a seeded RNG 
        # to generate offsets for x, y if the library doesn't directly support it.
        # This is a placeholder for if we find a way to seed it or use an alternative that supports seeding.
        pass 
    world = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            world[i][j] = perlin_noise.pnoise2(i / scale,
                                               j / scale,
                                               octaves=octaves,
                                               persistence=persistence,
                                               lacunarity=lacunarity,
                                               repeatx=rows, # Should be larger than rows/scale for non-tiling
                                               repeaty=cols, # Should be larger than cols/scale for non-tiling
                                               base=random_seed if random_seed is not None else 0) # base is an integer seed
    # Normalize to 0-1 and then scale to max_elevation
    world_min = np.min(world)
    world_max = np.max(world)
    if world_max > world_min:
        world = (world - world_min) / (world_max - world_min)
        world *= max_elevation
    else: # Flat terrain if noise is uniform (unlikely but possible)
        world = np.full((rows,cols), max_elevation / 2) # or some other default flat elevation
    return world

def initialize_grid(size, fuel_load=1.0, moisture=0.2, elevation_source=None, cell_resolution=30.0):
    rows, cols = size
    grid = np.zeros(size, dtype=grid_dtype)
    grid['fuel'] = fuel_load
    grid['moisture'] = moisture
    grid['state'] = CellState.UNBURNED
    grid['ignition_time'] = -1 # -1 means not ignited

    if isinstance(elevation_source, str):
        if elevation_source.lower() == "random":
            grid['elevation'] = _generate_random_terrain(rows, cols)
        elif elevation_source.lower().endswith(('.tif', '.tiff')):
            try:
                with rasterio.open(elevation_source) as dataset:
                    # Read the first band into an array, resample to grid dimensions
                    grid['elevation'] = dataset.read(
                        1, 
                        out_shape=(rows, cols), 
                        resampling=Resampling.bilinear
                    ).astype('f4')
            except Exception as e:
                print(f"Error loading DEM {elevation_source}: {e}. Using flat terrain.")
                grid['elevation'] = 0.0
        else:
            print(f"Unknown elevation_source string: {elevation_source}. Using flat terrain.")
            grid['elevation'] = 0.0
    elif elevation_source is None:
        grid['elevation'] = 0.0 # Flat terrain
    else:
        # Assuming elevation_source is a pre-loaded numpy array
        try:
            source_array = np.array(elevation_source, dtype='f4')
            if source_array.shape == size:
                grid['elevation'] = source_array
            else:
                # Resample if shapes don't match
                print(f"Resampling elevation data from {source_array.shape} to {size}")
                zoom_factors = (size[0] / source_array.shape[0], size[1] / source_array.shape[1])
                grid['elevation'] = zoom(source_array, zoom_factors, order=1) # order=1 for bilinear
        except Exception as e:
            print(f"Error applying provided elevation data: {e}. Using flat terrain.")
            grid['elevation'] = 0.0
            
    # cell_resolution is not stored directly in the grid but passed to step function
    return grid

def ignite(grid, x, y, time=0):
    grid['state'][x, y] = CellState.BURNING
    grid['ignition_time'][x, y] = time

def step(grid, time=0, moisture_threshold=0.3, wind=None, base_prob=1.0, wind_strength=0.5, random_seed=None, slope_coefficient=0.1, cell_resolution=30.0):
    """
    Perform one simulation step: burning cells ignite adjacent unburned cells.
    Spread depends on fuel, moisture, wind, and slope.
    Args:
        grid (np.ndarray): simulation grid
        time (int): current time step
        moisture_threshold (float): maximum moisture for ignition
        wind (tuple or None): (u, v) wind vector (east, north), or (speed, direction_degrees)
        base_prob (float): base ignition probability
        wind_strength (float): how strongly wind affects spread
        random_seed (int, optional): seed for random number generator
        slope_coefficient (float): how strongly slope affects spread
        cell_resolution (float): physical size of a cell edge in meters
    Returns:
        int: number of new ignitions
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    new_ignitions = []
    rows, cols = grid.shape

    if wind is None:
        wind_u, wind_v = 0.0, 0.0
    elif len(wind) == 2:
        if isinstance(wind[1], (int, float)) and abs(wind[1]) > 2 * np.pi: # Check against 2*pi for radians
            speed, deg = wind
            rad = math.radians(deg)
            wind_u = speed * math.cos(rad)
            wind_v = speed * math.sin(rad)
        else:
            wind_u, wind_v = wind
    
    dirs = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)] # N, S, W, E, NW, NE, SW, SE
    # For distance calculation: diagonal neighbors are sqrt(2) * cell_resolution away
    # For simplicity, we can assume basic cardinal distances for slope calc or use actual
    # For now, slope gradient is always over 'cell_resolution' distance as an approximation even for diagonals,
    # or we can calculate true distance for diagonals.
    # Let's use true distance for slope calculation if neighbor is diagonal.

    wind_vec = np.array([wind_u, wind_v])
    wind_norm = np.linalg.norm(wind_vec)

    for r_curr in range(rows):
        for c_curr in range(cols):
            if grid['state'][r_curr, c_curr] == CellState.BURNING:
                for dr, dc in dirs:
                    r_next, c_next = r_curr + dr, c_curr + dc

                    if 0 <= r_next < rows and 0 <= c_next < cols:
                        if (
                            grid['state'][r_next, c_next] == CellState.UNBURNED
                            and grid['fuel'][r_next, c_next] > 0
                            and grid['moisture'][r_next, c_next] < moisture_threshold
                        ):
                            # Wind effect calculation
                            wind_align = 0.0
                            if wind_norm > 0:
                                spread_cartesian_vec = np.array([dc, -dr]) # (East, North)
                                norm_spread_cartesian = np.linalg.norm(spread_cartesian_vec)
                                if norm_spread_cartesian > 0:
                                    wind_align = np.dot(wind_vec / wind_norm, spread_cartesian_vec / norm_spread_cartesian)
                            
                            prob = base_prob * (1 + wind_strength * wind_align)
                            prob = max(prob, 0) # Ensure probability is not negative

                            # Slope effect calculation
                            elevation_current = grid['elevation'][r_curr, c_curr]
                            elevation_neighbor = grid['elevation'][r_next, c_next]
                            delta_elevation = elevation_neighbor - elevation_current
                            
                            # Determine distance based on cardinal or diagonal move
                            distance = cell_resolution
                            if abs(dr) == 1 and abs(dc) == 1: # Diagonal move
                                distance = math.sqrt(2) * cell_resolution
                            
                            slope_gradient = delta_elevation / distance if distance > 0 else 0
                            slope_factor = math.exp(slope_coefficient * slope_gradient)
                            
                            prob *= slope_factor
                            prob = max(prob, 0) # Ensure probability remains non-negative after slope factor

                            if np.random.rand() < prob:
                                new_ignitions.append((r_next, c_next))
                
                grid['state'][r_curr, c_curr] = CellState.BURNED # Mark as burned after considering all its neighbors

    unique_new_ignitions = []
    if new_ignitions: # Remove duplicates if multiple burning cells ignite the same new cell
        # Create a set of tuples to automatically handle duplicates
        seen_ignitions = set()
        for ig_r, ig_c in new_ignitions:
            if (ig_r, ig_c) not in seen_ignitions:
                unique_new_ignitions.append((ig_r, ig_c))
                seen_ignitions.add((ig_r, ig_c))

    for r_ig, c_ig in unique_new_ignitions:
        if grid['state'][r_ig, c_ig] == CellState.UNBURNED: # Ensure not already ignited in this step by another logic path
            grid['state'][r_ig, c_ig] = CellState.BURNING
            grid['ignition_time'][r_ig, c_ig] = time
    
    return len(unique_new_ignitions)

def run_simulation(grid, max_steps=100, moisture_threshold=0.3, wind=None, base_prob=1.0, wind_strength=0.5, random_seed=None, slope_coefficient=0.1, cell_resolution=30.0):
    history = []
    history.append(grid.copy()) # Store initial state
    
    current_seed = random_seed
    for t in range(1, max_steps + 1):
        if random_seed is not None:
            current_seed = random_seed + t # Vary seed per step for more diverse outcomes if desired, or keep fixed
        
        args_for_step = {
            'time': t,
            'moisture_threshold': moisture_threshold,
            'wind': wind,
            'base_prob': base_prob,
            'wind_strength': wind_strength,
            'random_seed': current_seed, # Pass potentially varying seed
            'slope_coefficient': slope_coefficient,
            'cell_resolution': cell_resolution
        }
        n_new = step(grid, **args_for_step)
        history.append(grid.copy())
        if n_new == 0 and np.all(grid['state'] != CellState.BURNING):
            break # No new ignitions and no cells are burning
    return history

def burned_area_over_time(history):
    burned_counts = []
    for grid_snapshot in history:
        burned_counts.append(np.count_nonzero(grid_snapshot['state'] == CellState.BURNED))
    return np.array(burned_counts)

# --- Plotting (optional, can be in notebook) ---
def plot_grid_state(grid, time_step=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    
    # Define colors for states
    # UNBURNED: green, BURNING: red, BURNED: black/gray
    # Create a colormap
    # 0 (UNBURNED) -> green, 1 (BURNING) -> red, 2 (BURNED) -> black
    cmap = plt.cm.colors.ListedColormap(['green', 'red', 'black'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(grid['state'], cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if time_step is not None:
        ax.set_title(f"Time: {time_step}")
    else:
        ax.set_title("Grid State")
    # Create a legend or colorbar
    cbar = plt.colorbar(ax.imshow(grid['state'], cmap=cmap, norm=norm), ax=ax, ticks=[0, 1, 2], orientation='vertical')
    cbar.ax.set_yticklabels(['Unburned', 'Burning', 'Burned'])

def plot_grid_property(grid, property_name, time_step=None, ax=None, cmap='viridis'):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    
    if property_name not in grid.dtype.names:
        print(f"Property '{property_name}' not in grid dtype.")
        return
    
    im = ax.imshow(grid[property_name], cmap=cmap)
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    title = f"{property_name.capitalize()}"
    if time_step is not None:
        title = f"Time: {time_step} - {title}"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, orientation='vertical')
