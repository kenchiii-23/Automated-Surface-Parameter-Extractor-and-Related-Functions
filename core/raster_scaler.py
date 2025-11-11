import numpy as np
import pandas as pd
import tkinter as tk
from tkinter.filedialog import asksaveasfilename
import rasterio
import sampler as smp
from sklearn.preprocessing import StandardScaler

base = rf"C:\Users\KennethBaluyos (EPA)\Documents\04 - Manifesting RSTF\Projects\Project Reworked"
basePath = f"{base}/test"
elev = f"{base}/data/processed/cdo/_elevation.tif"
slope = f"{base}/data/processed/cdo/_slope.tif"
aspect = f"{base}/data/processed/cdo/_aspect.tif"
planCurvature  = f"{base}/data/processed/cdo/_planCurv.tif"
profCurvature  = f"{base}/data/processed/cdo/_profCurv.tif"
twi = f"{base}/data/processed/cdo/_TWI.tif"
relRel = f"{base}/data/processed/cdo/_relativeRelief.tif"
distRiv = f"{base}/data/processed/cdo/_distRiver.tif"

raster_paths=[elev,slope,aspect,planCurvature, profCurvature, twi, relRel, distRiv]
csvHeader = np.array(["LONG", "LAT", "Elevation", "Slope", "Aspect", "Planform", "Profile", "TWI", "Relative_Relief", "Distance_to_River"])

factors_full_unstacked = [] #array of unstacked feautures
meta = None

for i, path in enumerate(raster_paths):
    with rasterio.open(path) as factor_raster:
        factor_full_unscaled = factor_raster.read(1).astype(float)
        f_transform = factor_raster.transform
        f_profile = factor_raster.profile
        f_crs = factor_raster.crs
        f_ndVal = factor_raster.nodata

        if meta is None:
            meta = factor_raster.meta.copy()

        # Convert no data values to nan for easy working and masking later
        factor_ndVal = factor_raster.nodata

        if factor_ndVal is not None:
            factor_full_unscaled[factor_full_unscaled == factor_ndVal] = np.nan 
        
        factor_out = factor_full_unscaled

        
        mask = np.isnan(factor_full_unscaled)
        # Before we scale, we need to remove nans
        # This will return 1d array, of all valid unscaled. Position is unretained
        factor_valid_unscaled = factor_full_unscaled[~mask]
        factor_valid_unscaled= factor_valid_unscaled.reshape(-1,1)
        factor_full_scaled = factor_full_unscaled.copy()

        scaler = StandardScaler()
        factor_full_scaled[~mask] = scaler.fit_transform(factor_valid_unscaled).reshape(-1)
        factor_out = factor_full_scaled

        # =======================
        out_meta = meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": factor_full_unscaled.shape[0],
            "width": factor_full_unscaled.shape[1],
            "count": 1,
            "dtype": "float32",
            "transform": f_transform,
            "crs": f_crs
        })


        with rasterio.open(f"{base}/data/processed/cdo/{csvHeader[i+2]}_scaled.tif", "w", **out_meta) as dst:
            dst.write(factor_full_scaled.astype(np.float32), 1)

        
        factors_full_unstacked.append(factor_full_scaled)

cellsize = f_transform.a     
cellsize_y = -f_transform.e   
xllcorner = f_transform.c
height = f_profile['height']
yllcorner = f_transform.f - (cellsize_y * height)  # compute lower-left corner

inputPath = rf"C:\Users\KennethBaluyos (EPA)\Documents\04 - Manifesting RSTF\Projects\Project Reworked\data\raw\input.csv"

output = smp.stack(factors_full_unstacked, csvHeader, xllcorner, yllcorner, cellsize, filePath=inputPath, fileBase='SCALER', height=height)

        

