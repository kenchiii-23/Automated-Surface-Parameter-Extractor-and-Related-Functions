import numpy as np
import rasterio
import pickle
import tkinter as tk
from tkinter.filedialog import askopenfilename
from pyproj import CRS
import os
import sys


def getPath():
    tk.Tk().withdraw() 
    filename = askopenfilename(title="Open") 
    return filename


def save(elevation, slope, profCurv, planCurv, aspect, relativeRelief, profile, flowDir, flowAccu, TWI, distRiver, path, cityList):
    provinceName=cityList[0]
    provinceName = provinceName.replace(' ', '_')
    cityName=cityList[1]
    cityName =  cityName.replace(' ', '_')

    ###Saving Slope
    slope_profile = profile.copy()

    # Update the driver and data type
    slope_profile.update(
    driver="GTiff",          
    dtype=rasterio.float32, 
    count=1,                 
    nodata=-9999
    )

    savePath = f"{path}/OutputRaster/{provinceName}/{cityName}/"
    os.makedirs(savePath, exist_ok=True)
    
    with rasterio.open(f"{savePath}{cityName}_elevation.tif", "w", **slope_profile) as dst:
        dst.write(elevation.astype(np.float32), 1)

    with rasterio.open(f"{savePath}{cityName}_slope.tif", "w", **slope_profile) as dst:
        dst.write(slope.astype(np.float32), 1)

    with rasterio.open(f"{savePath}{cityName}_profCurv.tif", "w", **slope_profile) as dst:
        dst.write(profCurv.astype(np.float32), 1)

    with rasterio.open(f"{savePath}{cityName}_planCurv.tif", "w", **slope_profile) as dst:
        dst.write(planCurv.astype(np.float32), 1)

    with rasterio.open(f"{savePath}{cityName}_aspect.tif", "w", **slope_profile) as dst:
        dst.write(aspect.astype(np.float32), 1)

    with rasterio.open(f"{savePath}{cityName}_relativeRelief.tif", "w", **slope_profile) as dst:
        dst.write(relativeRelief.astype(np.float32), 1)

    with rasterio.open(f"{savePath}{cityName}_flowDirection.tif", "w", **slope_profile) as dst:
        dst.write(flowDir.astype(np.float64), 1)
    
    with rasterio.open(f"{savePath}{cityName}_flowAccumulation.tif", "w", **slope_profile) as dst:
        dst.write(flowAccu.astype(np.float64), 1)

    with rasterio.open(f"{savePath}{cityName}_TWI.tif", "w", **slope_profile) as dst:
        dst.write(TWI.astype(np.float64), 1)
    
    with rasterio.open(f"{savePath}{cityName}_distRiver.tif", "w", **slope_profile) as dst:
        dst.write(distRiver.astype(np.float64), 1)

def extractDEM(file):

    with rasterio.open(file) as src:
        elevation = src.read(1).astype(float)
        profile = src.profile
        transform = src.transform
        height = src.height

    if profile["crs"] == 'None':
        profile["crs"] = CRS.from_wkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]')
    return elevation, profile, transform, height


def loadDEM(base, demPath=None, loadFromConfig=False, city=None):
    #os.system('cls')
    provinceName=city[0]
    provinceName = provinceName.replace(' ', '_')
    cityName=city[1]
    cityName =  cityName.replace(' ', '_')
   
    if loadFromConfig:
        try: ## Reads config file...
            print("Loading DEM from previous configuration...")
            with open(f"{base}/config/{provinceName}/{cityName}/path_info.pkl", "rb") as t:
                path_info = pickle.load(t)

            data = np.load(path_info[6], allow_pickle=True)
            fn = os.path.basename(path_info[6])
            filename = os.path.splitext(str.replace(fn, "_temp", ""))[0] 

            with open(path_info[3], "rb") as t:
                profile = pickle.load(t)

            with open(path_info[4], "rb") as t:
                transform = pickle.load(t)

            with open(path_info[5], "rb") as t:
                height = pickle.load(t)

            return data, profile, transform, height, filename
        
        except FileNotFoundError:
            print("No previous DEM configuration found. Please select try loading from a DEM")
            raise FileNotFoundError

    
    elif loadFromConfig == False:
        print("No previous DEM configuration found. Please select a DEM file.")
        print(demPath)

        '''tk.Tk().withdraw()
        demPath = askopenfilename() 
        if not demPath.__contains__(".asc"):
            print(f"File at {demPath} is in incorrect file format...")
            sys.exit()'''

        
        fn = os.path.basename(demPath) # Filename with extension {city}.asc
        filename = os.path.splitext(fn)[0] #{city}

        print(f"Now loading DEM {filename}.asc from: {demPath}")

        try:
            #demPath is whole directory
            filePath = demPath
            base = os.getcwd() #return parent directory
            savePath = f"{base}/data/temp/{provinceName}/{cityName}" # custom save folder for the city

            data, profile, transform, height = extractDEM(filePath)
            dem_NPY_filename = savePath + f"/{filename}_temp.npy"
            dem_profile_filename = savePath + f"/{filename}_profile.pkl"
            dem_transform_filename = savePath + f"/{filename}_transform.pkl"
            dem_height_filename = savePath + f"/{filename}_height.pkl"
            
            path_info = np.array(["True", filePath, savePath, dem_profile_filename, dem_transform_filename, dem_height_filename, dem_NPY_filename])


            os.makedirs(f"{savePath}/", exist_ok=True)
            os.makedirs(f"{base}/config/{provinceName}/{filename}", exist_ok=True)


            np.save(dem_NPY_filename, data)

            with open(dem_profile_filename, "wb") as p:
                pickle.dump(profile, p)    

            with open(dem_transform_filename, "wb") as t:
                pickle.dump(transform, t)

            with open(dem_height_filename, "wb") as h:
                pickle.dump(height, h)

            with open(f"{base}/config/{provinceName}/{filename}/path_info.pkl", "wb") as l:
                pickle.dump(path_info, l)    

            return data, profile, transform, height, filename

        except Exception as e:
            print(f"Error loading DEM: {e}")
            return None, None, None, None, None

