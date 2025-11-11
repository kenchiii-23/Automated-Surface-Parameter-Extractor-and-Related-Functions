from scipy import ndimage as ndi
import numpy as np

def calculateSurface(arrData, filled, cellsizex, cellsizey):

    mask = (arrData == -9999) #Will be used to filter those that have center values of -9999

    #Zenberg and Thorne Kernels 
    dKernel = np.array([
        [0,0,0],
        [.5,-1,.5],
        [0,0,0]
    ], dtype=float)

    eKernel = np.array([
        [0,.5,0],
        [0,-1,0],
        [0,.5,0]
    ], dtype=float)  

    eKernel = np.array([
        [0,.5,0],
        [0,-1,0],
        [0,.5,0]
    ], dtype=float) 

    fKernel = np.array([
        [-1,0,1],
        [0,0,0],
        [1,0,-1]
    ], dtype=float)

    gKernel = np.array([
        [0,0,0],
        [1,0,-1],
        [0,0,0]
    ], dtype=float) 

    hKernel = np.array([
        [0,-1,0],
        [0,0,0],
        [0,1,0]
    ], dtype=float) 


    d = ndi.convolve(filled, dKernel, mode='nearest')/ (cellsizex**2)   
    e = ndi.convolve(filled, eKernel, mode='nearest')/ (cellsizey**2)
    f = ndi.convolve(filled, fKernel, mode='nearest')/ (4 * cellsizex**2)
    g = ndi.convolve(filled, gKernel, mode='nearest')/ (2 * cellsizex)
    h = ndi.convolve(filled, hKernel, mode='nearest')/ (2 * cellsizey)

    #Slope Calculation
    slopeRad = np.atan(np.sqrt(g**2 + h**2))
    slope = np.degrees(slopeRad)

    #Curvature
    denom = ((g**2) + (h**2))
    denom_safe = np.where(denom == 0, np.nan, denom) #if denom 0, replace by nan
    profCurv = (-200) * ((d*(g**2)) + (e*(h**2)) + (f*g*h)) / denom_safe #curvs now return nan if 0 denom
    planCurv = (200) * ((d*(h**2)) + (e*(g**2)) - (f*g*h)) / denom_safe
    profCurv = np.nan_to_num(profCurv, nan=0.0) #turns nan back to 0s
    planCurv= np.nan_to_num(planCurv, nan=0.0)

    #Aspect
    preAspect = np.atan2(-h,-g)
    aspect = (450-np.degrees(preAspect)) %360

    #Handling flat aspect cases
    aspectTolerance = 1e-8
    isFlat = denom <= aspectTolerance
    aspect = np.where(isFlat, -1, aspect)

    #Relative Relief
    maximum = ndi.maximum_filter(filled, size=3)
    minimum = ndi.minimum_filter(filled, size=3)
    relativeRelief = maximum - minimum


    #Removing values where elevation is no data
    slope[mask] = -9999
    profCurv[mask] = -9999
    planCurv[mask] = -9999
    aspect[mask] = -9999
    relativeRelief[mask] = -9999

    return slope, profCurv, planCurv, aspect, relativeRelief


def calculateHaversine(shape, yllcorner, cellsize):

    row, col = shape
    baseLat = yllcorner + (col*cellsize)

    prepArrLat = baseLat - (cellsize) * np.arange(row + 1)

    R = 6371000

    # ---------- Horizontal Distance Calculation ----------
    latX_rad = np.radians(prepArrLat)
    deltaLon_rad = np.radians(cellsize)

    ax = (np.cos(latX_rad)**2) * np.sin(deltaLon_rad / 2)**2
    cX = 2 * np.asin(np.sqrt(ax))


    # ---------- Vertical Distance Calculation ----------
    deltaLatY_rad = np.radians(cellsize)

    aY = np.sin(deltaLatY_rad / 2)**2
    cY = 2 * np.asin(np.sqrt(aY))

    colDist = R * cX
    horizontal_distances = np.broadcast_to(colDist[:-1, None], (row, col))
    rowDist = R * cY
    vertical_distances = np.full((row, col), rowDist)

    return horizontal_distances, vertical_distances

def initializeParameters(shape):
    empty = np.empty(shape=shape)
    return empty, empty, empty, empty, empty, empty, empty, empty, empty, empty