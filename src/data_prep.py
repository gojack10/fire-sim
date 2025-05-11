import numpy as np
import rasterio

def load_dem(filepath):
    """
    Load a DEM (Digital Elevation Model) from a GeoTIFF file.
    Args:
        filepath (str): Path to the GeoTIFF DEM file.
    Returns:
        np.ndarray: 2D array of elevation values.
    """
    with rasterio.open(filepath) as src:
        dem = src.read(1)
    return dem

def create_uniform_fuel_grid(shape, fuel_value=1.0):
    """
    Create a 2D array representing uniform fuel load across the grid.
    Args:
        shape (tuple): (rows, cols) of the grid
        fuel_value (float): Value for all cells
    Returns:
        np.ndarray: 2D array of fuel values
    """
    return np.full(shape, fuel_value, dtype=np.float32)

def get_weather_inputs():
    """
    Placeholder for weather input ingestion.
    Returns:
        dict: Static wind and humidity values (to be expanded later)
    """
    return {
        'wind_speed': 0.0,  # m/s
        'wind_direction': 0.0,  # degrees
        'humidity': 30.0  # percent
    }
