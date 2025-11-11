from scipy import ndimage as ndi
import numpy as np
import os
import sys
## import matrix_operand as mm
import core.calc_SurfaceParams as csp
import core.calc_HydroParams as chp
import core.file_manager as fm
import core.sampler as smp
import core.area_extractor as ae
import matplotlib.pyplot as plt
import time


start = time.perf_counter()

def main():
    lonX, latY = input("Input Long and Lat: ").replace(" ", "").replace("°E", ", ").replace("°N", "").split(", ")
    row, col= getDEMCoordinates(lonX, latY)
    row = row -1
    col = col -1

    print (f"Thorne Slope: {slope[row,col]}")
    print (f"Thorne Prof Curv: {profCurv[row,col]}")
    print (f"Thorne Plan Curv: {planCurv[row,col]}")
    print(f"Aspect: {aspect[row, col]}")
    print(f"Relative Relief: {relativeRelief[row, col]}")
    print(f"Flow Direction: {flowDir[row, col]}")
    print(f"Flow Accumulation: {flowAccu[row, col]}")
    print(f"TWI: {TWI[row, col]}")
    print(f"distRiver: {distRiver[row, col]}")
    ##print(mm.showMatrix(row,col,filled))
    main()


def getDEMCoordinates(lon, lat):
    # Takes the total degrees elapsed from the x and y 
    # Also determines the raw row & colmun number, non 0 start
    absLong = (float(lon) - xllcorner) / cellsize
    absLat = height - ((float(lat) - yllcorner) / cellsize)

    # Gets the final number
    column = int(np.ceil(absLong))
    row = int(np.ceil(absLat))

    return row, column

def getParameters(filename):
    start = time.perf_counter()
    distX, distY = csp.calculateHaversine(np.shape(arrData), yllcorner, cellsize)
    slope, profCurv, planCurv, aspect, relativeRelief = csp.calculateSurface(arrData, filled, distX, distY)
    flowDir, flowAccu, TWI = chp.calculateHydroParams(arrData, profile, slope, distY)
    
    demPath = rf"{base}/data/raw/{filename}.asc"
    riverPath =rf"{base}/data/raw/dependencies/NAMRIA_RiverNetwork/phl_rivl_250k_NAMRIA.shp"
    distRiver = ae.calcRiverDist(demPath, riverPath, distX)

    end = time.perf_counter()    # Record the end time
    elapsed_time = end - start
    print(f"Parameters calculated in {elapsed_time:.2f} seconds. Saving for future use....")

    start = time.perf_counter()
    os.makedirs(f"{base}/data/temp/{filename}/params/", exist_ok=True)
    savePath = f"{base}/data/temp/{filename}/params/{filename}"
    np.save(f"{savePath}_distX.npy", distX)
    np.save(f"{savePath}_distY.npy", distY)
    np.save(f"{savePath}_slope.npy", slope)
    np.save(f"{savePath}_profCurv.npy", profCurv)
    np.save(f"{savePath}_planCurv.npy", planCurv)
    np.save(f"{savePath}_aspect.npy", aspect)
    np.save(f"{savePath}_relativeRelief.npy", relativeRelief)
    np.save(f"{savePath}_flowDir.npy", flowDir)
    np.save(f"{savePath}_flowAccu.npy", flowAccu)
    np.save(f"{savePath}_TWI.npy", TWI)
    np.save(f"{savePath}_distRiver.npy", distRiver)

    end = time.perf_counter()    # Record the end time
    elapsed_time = end - start
    print(f"Parameters saved in {elapsed_time:.2f} seconds. Base file: '{filename}.asc'")

    return distX, distY, slope, profCurv, planCurv, aspect, relativeRelief, flowDir, flowAccu, TWI, distRiver


def loadParameters(filename):
    loadPath = f"{base}/data/temp/{filename}/params/{filename}"
    distX = np.load(f"{loadPath}_distX.npy")
    distY = np.load(f"{loadPath}_distY.npy")
    slope = np.load(f"{loadPath}_slope.npy")
    profCurv = np.load(f"{loadPath}_profCurv.npy")
    planCurv = np.load(f"{loadPath}_planCurv.npy")
    aspect = np.load(f"{loadPath}_aspect.npy")
    relativeRelief = np.load(f"{loadPath}_relativeRelief.npy")
    flowDir = np.load(f"{loadPath}_flowDir.npy")
    flowAccu = np.load(f"{loadPath}_flowAccu.npy")
    TWI = np.load(f"{loadPath}_TWI.npy")
    distRiver = np.load(f"{loadPath}_distRiver.npy")

    return distX, distY, slope, profCurv, planCurv, aspect, relativeRelief, flowDir, flowAccu, TWI, distRiver

# ===== Phase 0: Open DEM File ===== #

base = os.getcwd()
data, profile, transform, height, filename = fm.loadDEM()

# ===== Phase 1: DEM Loading and Filling ===== #

#Get DEM Info
cellsize = transform.a     
cellsize_y = -transform.e   
xllcorner = transform.c
yllcorner = transform.f - (height * cellsize_y)  # compute lower-left corner
arrData = np.array(data)

print(f"\nFile: '{filename}.asc'")
print(f"Lower X: {xllcorner}")
print(f"Lower Y: {yllcorner}")
print(f"Cell Size: {cellsize}")
print(f"Height (rows): {height}")
print(f"Width (cols): {profile['width']}")
print(f"Projection CRS: {profile['crs']}\n")

## Fill the DEM for NoData Values
mask = (arrData == -9999)
dist, indices = ndi.distance_transform_edt(mask, return_indices=True)
filled = arrData.copy()
filled[mask] = arrData[tuple(i[mask] for i in indices)]
filled[mask & (dist > np.sqrt(2))] = -9999


end = time.perf_counter()    # Record the end time
elapsed_time = end - start 
print(f"DEM loaded and filled in {elapsed_time:.2f} seconds.")




# ===== Phase 2: Parameter Calculations ===== #
print("\n\n\n# ===== Phase 2: Parameter Calculations ===== #\nChecking files...")
paramsListText = np.array([f"{filename}_distX", f"{filename}_distY", f"{filename}_slope", f"{filename}_profCurv", f"{filename}_planCurv", f"{filename}_aspect", f"{filename}_relativeRelief", f"{filename}_flowDir", f"{filename}_flowAccu", f"{filename}_TWI"])
parametersExist = np.bool(True)

#Check if file exists
for params in paramsListText:
    if not os.path.isfile(f"{base}/data/temp/{filename}/params/{params}.npy"):
        print(f"{params}.npy not found")
        parametersExist = np.bool(False)


if parametersExist: # Found previous temp file
    dec = input("Previous parameters found... Would you like to load them? (y/n):")
    if dec == "y":
        start = time.perf_counter()
        distX, distY, slope, profCurv, planCurv, aspect, relativeRelief, flowDir, flowAccu, TWI, distRiver = loadParameters(filename)

        elapsed_time = time.perf_counter() - start
        print(f"Parameters loaded in {elapsed_time:.2f} seconds. Base file: '{filename}.asc'")

    elif dec == "n":
        print("Now Calculating...")
        distX, distY, slope, profCurv, planCurv, aspect, relativeRelief, flowDir, flowAccu, TWI, distRiver = getParameters(filename)
else:
    print("\nSome previous calculations not found. Now Calculating...")
    distX, distY, slope, profCurv, planCurv, aspect, relativeRelief, flowDir, flowAccu, TWI, distRiver = getParameters(filename)


# ===== Phase 3: Point Feauture Extraction ===== #
print("\n\n\n# ===== Phase 3: Point Feauture Extraction ===== #\nSelect input coordinates csv...")
# Get input csv containing points in decimal degree coordinates
inputTablePath = fm.getPath()
if not inputTablePath.__contains__(".csv"):
   print(f"File at {inputTablePath} is in incorrect file format...")
   sys.exit()
   

print(f"Running Extractor...")
start = time.perf_counter()
paramsList = np.array([arrData, np.radians(slope), profCurv, planCurv, aspect, relativeRelief, flowDir, flowAccu, TWI, distRiver])
csvHeader = np.array(["LONG", "LAT", "Elevation", "Slope", "Planform", "Profile", "Aspect", "Relative_Relief", "Flow_Direction", "Flow_Accumulation", "TWI", "Distance_to_River"])
output = smp.stack(paramsList, csvHeader, xllcorner, yllcorner, cellsize, inputTablePath, filename, profile['height'])

end = time.perf_counter()    # Record the end time
elapsed_time = end - start
print(f"Feautures Extracted and Saved in {elapsed_time:.2f} seconds... Base file: '{filename}.asc'")



# ===== Phase 4: Raster Saving ===== #
print("\n\n\n# ===== Phase 4: Raster Saving ===== #")
if input("Would you like to save rasters? (y/n): ") == "y":
    start = time.perf_counter()
    try:
        fm.save(arrData, slope, profCurv, planCurv, aspect, relativeRelief, profile, flowDir, flowAccu, TWI, distRiver, base, filename)
        end = time.perf_counter()    # Record the end time
        elapsed_time = end - start
        print(f"Outputs saved in {elapsed_time:.2f} seconds.  Base file: '{filename}.asc'")
    
    except Exception as e:
        print(f"Error Saving: {e}")
        print(f"Could not save...")
        

    


# ===== Phase 5: Custom Actions ===== #
print("\n\n\n# ===== Phase 5: Custom Actions ===== #")
main()
