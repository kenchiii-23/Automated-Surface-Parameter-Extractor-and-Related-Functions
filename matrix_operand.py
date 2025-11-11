import numpy as np
import os

def showMatrix(row, col, array):
    E = array[row,col]
    shpRow, shpCol = array.shape
    maxRow = shpRow-1
    maxCol = shpCol-1

    A = E if row == 0 or col == 0 else array[row-1, col-1]
    B = E if row == 0 else array[row-1, col]
    C = E if row == 0 or col == maxCol else array[row-1, col+1]

    D = E if col == 0 else array[row, col-1]
    F = E if col == maxCol else array[row, col+1]

    G = E if row == maxRow or col == 0 else array[row+1, col-1]
    H = E if row == maxRow else array [row+1, col]
    I = E if row == maxRow or col == maxCol else array[row+1,col+1]


    matrix = np.array([[A,B,C],
                       [D,E,F],
                       [G,H,I]])
#    matrix[matrix == -9999] = E
    return matrix



### 1: 1st Load, No TIFF Saving
### 1: 2nd Load, No TIFF Saving

def deleter(mode):
    if mode == 1:
        fileList = np.array(["dem_height.pkl", "dem_profile.pkl", "dem_transform.pkl", "dem.npy", "slope_output.tif"])
    
    elif mode == 2:
        fileList = np.array(["slope_output.tif"])
    
    for file in fileList:
        try:
            os.remove(file)
            print(f"File '{file}' deleted successfully.")
        except FileNotFoundError:
            print(f"Error: File '{file}' not found.")
        except PermissionError:
            print(f"Error: Permission denied to delete '{file}'.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")