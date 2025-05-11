import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tempfile
import rasterio
from rasterio.transform import from_origin
from src import data_prep

def test_create_uniform_fuel_grid():
    shape = (5, 5)
    fuel = data_prep.create_uniform_fuel_grid(shape, fuel_value=2.5)
    assert fuel.shape == shape
    assert np.allclose(fuel, 2.5)

def test_get_weather_inputs():
    weather = data_prep.get_weather_inputs()
    assert 'wind_speed' in weather
    assert 'wind_direction' in weather
    assert 'humidity' in weather

def test_load_dem():
    # Create a temporary DEM GeoTIFF file
    arr = np.arange(9, dtype=np.float32).reshape((3, 3))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'dem.tif')
        transform = from_origin(0, 3, 1, 1)
        with rasterio.open(
            path, 'w', driver='GTiff',
            height=arr.shape[0], width=arr.shape[1],
            count=1, dtype=arr.dtype, crs='+proj=latlong', transform=transform
        ) as dst:
            dst.write(arr, 1)
        loaded = data_prep.load_dem(path)
        assert np.allclose(loaded, arr)
