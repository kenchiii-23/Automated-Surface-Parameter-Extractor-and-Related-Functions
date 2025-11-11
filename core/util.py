import os
import numpy as np
### 1: 1st Load, No TIFF Saving
### 1: 2nd Load, No TIFF Saving

def deleter(mode):
    path = r"C:\Users\KennethBaluyos (EPA)\Documents\04 - Manifesting RSTF\Projects\Project Reworked\data"
    if mode == 1:
        fileList = np.array([f"{path}dem_height.pkl", f"{path}dem_profile.pkl", f"{path}dem_transform.pkl", f"{path}dem.npy", f"{path}slope_output.tif"])
    
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

def main():
    os.system('cls')
    print("========= Utilities Available =========")
    print("1 ) Full Cleanup (Deletes all saved DEM and slope files)\n")
    dec = input("Select Utility #: ")

    if dec == '1':
        dec2 = input("Enter Delete Mode: ")
        if dec2 == '1':
            deleter(1)
            input("Press Enter to continue...")
            main()

        else:
            main()

    else:
        main()


main()