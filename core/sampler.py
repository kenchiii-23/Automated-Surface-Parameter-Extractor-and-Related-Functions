import numpy as np
import pandas as pd
import tkinter as tk
from tkinter.filedialog import asksaveasfilename


def stack(arrayList, column, xllcorner, yllcorner, cellsize, filePath, fileBase, height):
    tk.Tk().withdraw()
    stacked = np.stack(arrayList, axis=2)
    nodes, coords = getDEMCoordinates(filePath, xllcorner, yllcorner, cellsize, height)


    outShape = (nodes.shape[0], 2 + stacked[0,0].shape[0] )
    out = np.empty(outShape, dtype=np.float64)

    for m, (i,j) in enumerate(nodes):
        i, j = int(i), int(j)
        values = stacked[i, j] #returns [a,b,c,d] at row i and col j
        out[m] = np.insert(values, 0, coords[m])
    out = np.round(out, 6)


    df = pd.DataFrame(out, columns=column)
    print("Select ouput csv...")
    fileOut = asksaveasfilename(title="Save Output CSV:")
    fileOut = fileOut + f"_{fileBase}.csv"
    df.to_csv(fileOut, index=False, float_format="%.6f")

    return out



def getDEMCoordinates(filePath, xllcorner, yllcorner, cellsize, height):

    points = pd.read_csv(filePath)
    coords = points[["LONG", "LAT"]].to_numpy()
    nodes = np.empty_like(coords)
    for i, (long, lat) in enumerate(coords):

        absLong = (float(long) - xllcorner) / cellsize
        absLat = height - ((float(lat) - yllcorner) / cellsize)

        # Gets the final number
        column = int(np.ceil(absLong))
        row = int(np.ceil(absLat))
        column = column - 1
        row = row - 1
        nodes[i] = np.array([row, column])
        
    return nodes, coords

def extractPointFactor(arrayList, header, xs, ys, profile):
    stacked = np.stack(arrayList, axis=2)
    outShape = (xs.shape[0], 2 + stacked[0,0].shape[0] )
    out = np.empty(outShape, dtype=np.float64)
    rows, cols = xy_to_row_prof(xs, ys, profile)

    for m, (i,j) in enumerate(zip(rows, cols)):
        i, j = int(i), int(j)
        values = stacked[i, j] #returns [a,b,c,d] at row i and col j
        out[m] = np.insert(values, 0, (xs[m], ys[m]))
    out = np.round(out, 6)

    return out
    """
    dk = pd.DataFrame(out, columns=header)
    fileOut = "toARFF.csv"
    dk.to_csv(fileOut, index=False, float_format="%.6f")
    """


    

def xy_to_rowcol(xs, ys, transform):
    # xs array of x coords, dd
    # ys array of y coords
    # transform is affine containing conversion instruction from row col to xy
    rows_out = []
    cols_out = []
    for xi, yi in zip(xs, ys):
        col, row = ~transform * (xi, yi)  # we take the ~ of since we're reversing instructions (xy to row col)
        # round to nearest int
        r = int(np.floor(row + 0.5))
        c = int(np.floor(col + 0.5))
        rows_out.append(r)
        cols_out.append(c)
    return np.array(rows_out), np.array(cols_out)


def xy_to_row_prof(xs, ys, profile):
    transform = profile['transform']
    height = profile['height']
    cellsize = transform[0]
    xllcorner = transform[2]
    yllcorner = transform[5] - (cellsize * height)
    rows_out = []
    cols_out = []
    for long, lat in zip(xs, ys):

        absLong = (float(long) - xllcorner) / cellsize
        absLat = height - ((float(lat) - yllcorner) / cellsize)

        # Gets the final number
        c = int(np.ceil(absLong))
        r = int(np.ceil(absLat))
        c = c - 1
        r = r - 1

        rows_out.append(r)
        cols_out.append(c)


    return np.array(rows_out), np.array(cols_out)