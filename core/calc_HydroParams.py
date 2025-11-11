import numpy as np
from pysheds.grid import Grid
from pysheds.sview import Raster, ViewFinder
from affine import Affine
from pyproj import CRS

def calculateHydroParams(data, profile, slope, cellsize):
    # Reads DEM Profile
    ncols = profile["width"]
    nrows = profile["height"]
    transform = profile["transform"]
    nodata = profile["nodata"]
    shape = (nrows, ncols)
    crs = profile["crs"]
    crs = CRS.from_wkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]')
    if crs == 'None':
        crs = CRS.from_wkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]')
    print(crs)

    mask = np.where(data==nodata, np.bool(False), np.bool(True))

    # Sets the DEM Raster for Pysheds
    nodata = data.dtype.type(nodata)
    affine = Affine(transform[0], transform[1], transform[2], transform[3], transform[4], transform[5])
    viewfinder = ViewFinder(affine=affine, shape=shape, mask=mask, nodata=nodata, crs=crs)
    dem = Raster(data, viewfinder)

    # Instantiate Grid
    grd = Grid()
    grd.viewfinder = dem.viewfinder
    grid = grd

    # Conditioning the DEM
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    prepDem = grid.resolve_flats(flooded_dem)


    # ========================================================== #
    # 1.) Calculate Flow Direction and Accumulation
    fdir = grid.flowdir(prepDem, nodata_out=nodata)
    acc = grid.accumulation(fdir, nodata_out=nodata)
    TWI = calcTWI(acc, slope, cellsize, data)
  
    return fdir, acc, TWI

def calcTWI(flowAccumulation, slopeArray, cellsizeArray, data):

    slope_rad = np.radians(slopeArray)
    specific_catchment = flowAccumulation * cellsizeArray
    e = 1e-6 # avoid division by zero
    TWI = np.log((specific_catchment + e) / (np.tan(slope_rad) + e)) 
    TWI[data == -9999] = -9999

    return TWI
