import numpy as np
import pickle as pkl
import rasterio
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tkinter.filedialog import askopenfilename
from pandas import read_csv as rd

def getDEMCoordinates(filePath, xllcorner, yllcorner, cellsize, height):

    points = rd(filePath)
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

base = r"C:\Users\KennethBaluyos (EPA)\Documents\04 - Manifesting RSTF\Projects\Project Reworked"
a_full_unscaled = np.load(f"{base}/data/temp/cdo/cdo_temp.npy")
b_full_unscaled = np.load(f"{base}/data/temp/cdo/params/cdo_TWI.npy")
c_full_unscaled = np.load(f"{base}/data/temp/cdo/params/cdo_slope.npy")
d_full_unscaled = np.load(f"{base}/data/temp/cdo/params/cdo_aspect.npy")
e_full_unscaled = np.load(f"{base}/data/temp/cdo/params/cdo_planCurv.npy")
f_full_unscaled = np.load(f"{base}/data/temp/cdo/params/cdo_profCurv.npy")
g_full_unscaled = np.load(f"{base}/data/temp/cdo/params/cdo_distRiver.npy")
h_full_unscaled = np.load(f"{base}/data/temp/cdo/params/cdo_relativeRelief.npy")

## Convert -9999 vaues to nan for easy operation later
a_full_unscaled[a_full_unscaled == -9999] = np.nan
b_full_unscaled[b_full_unscaled == -9999] = np.nan
c_full_unscaled[c_full_unscaled == -9999] = np.nan
d_full_unscaled[d_full_unscaled == -9999] = np.nan
e_full_unscaled[e_full_unscaled == -9999] = np.nan
f_full_unscaled[f_full_unscaled == -9999] = np.nan
g_full_unscaled[g_full_unscaled == -9999] = np.nan
h_full_unscaled[h_full_unscaled == -9999] = np.nan

#Create masks for the input arrays which will filter nan values
input_mask_a = np.isnan(a_full_unscaled)
input_mask_b = np.isnan(b_full_unscaled)
input_mask_c = np.isnan(c_full_unscaled)
input_mask_d = np.isnan(d_full_unscaled)
input_mask_e = np.isnan(e_full_unscaled)
input_mask_f = np.isnan(f_full_unscaled)
input_mask_g = np.isnan(g_full_unscaled)
input_mask_h = np.isnan(h_full_unscaled)

# Before we scale, we need to remove nans
# This will return 1d array, of all valid unscaled. Position is unretained
a_valid_unscaled = a_full_unscaled[~input_mask_a]
b_valid_unscaled = b_full_unscaled[~input_mask_b]
c_valid_unscaled = c_full_unscaled[~input_mask_c]
d_valid_unscaled = d_full_unscaled[~input_mask_d]
e_valid_unscaled = e_full_unscaled[~input_mask_e]
f_valid_unscaled = f_full_unscaled[~input_mask_f]
g_valid_unscaled = g_full_unscaled[~input_mask_g]
h_valid_unscaled = h_full_unscaled[~input_mask_h]

# Since scaler needs 2d array, it will be reshaped
a_valid_unscaled = a_valid_unscaled.reshape(-1,1)
b_valid_unscaled = b_valid_unscaled.reshape(-1,1)
c_valid_unscaled = c_valid_unscaled.reshape(-1,1)
d_valid_unscaled = d_valid_unscaled.reshape(-1,1)
e_valid_unscaled = e_valid_unscaled.reshape(-1,1)
f_valid_unscaled = f_valid_unscaled.reshape(-1,1)
g_valid_unscaled = g_valid_unscaled.reshape(-1,1)
h_valid_unscaled = h_valid_unscaled.reshape(-1,1)


# Create an array of nan same size with corresponding array
# will be used as place holder as the scaler calculates value for valids
a_full_scaled = a_full_unscaled.copy()
b_full_scaled = b_full_unscaled.copy()
c_full_scaled = c_full_unscaled.copy()
d_full_scaled = d_full_unscaled.copy()
e_full_scaled = e_full_unscaled.copy()
f_full_scaled = f_full_unscaled.copy()
g_full_scaled = g_full_unscaled.copy()
h_full_scaled = h_full_unscaled.copy()

## === Scale, separately since they differ in value range
scalerA = StandardScaler() #Initialize scaler
scalerB = StandardScaler()
scalerC = StandardScaler()
scalerD = StandardScaler()
scalerE = StandardScaler()
scalerF = StandardScaler()
scalerG = StandardScaler()
scalerH = StandardScaler()
a_full_scaled[~input_mask_a] = scalerA.fit_transform(a_valid_unscaled).reshape(-1)
b_full_scaled[~input_mask_b] = scalerB.fit_transform(b_valid_unscaled).reshape(-1)
c_full_scaled[~input_mask_c] = scalerC.fit_transform(c_valid_unscaled).reshape(-1)
d_full_scaled[~input_mask_d] = scalerD.fit_transform(d_valid_unscaled).reshape(-1)
e_full_scaled[~input_mask_e] = scalerE.fit_transform(e_valid_unscaled).reshape(-1)
f_full_scaled[~input_mask_f] = scalerF.fit_transform(f_valid_unscaled).reshape(-1)
g_full_scaled[~input_mask_g] = scalerG.fit_transform(g_valid_unscaled).reshape(-1)
h_full_scaled[~input_mask_h] = scalerG.fit_transform(h_valid_unscaled).reshape(-1)


## Create a cube of the arrays by stacking
full_scaled_stack3D = np.stack((a_full_scaled, b_full_scaled, c_full_scaled, d_full_scaled, e_full_scaled, f_full_scaled, g_full_scaled, h_full_scaled), axis=-1)
filename = f"{base}/data/raw/input.csv"
with open(f"{base}/data/temp/cdo/cdo_profile.pkl", "rb") as t:
    profile = pkl.load(t)

dem_cellsize = profile['transform'][0]
xllcorner = profile['transform'][2]
yllcorner = profile['transform'][5] - profile['height'] * dem_cellsize
nodes, coords = getDEMCoordinates(filename, xllcorner=xllcorner , yllcorner=yllcorner, cellsize=dem_cellsize, height=profile['height'])

nodes = nodes.astype(int)

## Create copy of the 3d hypercube, reshape into 2D
# Done such that it can be used to feed in the KDE after filtering
rows, cols, n_features = full_scaled_stack3D.shape
full_scaled_stack2D = full_scaled_stack3D.reshape(-1, n_features)

# Apply mask or filter on the full scaled stack 2d to remove nan values, leaving only valid pair
stack_scaled_mask = np.any(np.isnan(full_scaled_stack2D), axis=1)
valid_scaled_stack2D = full_scaled_stack2D[~stack_scaled_mask]


# Extract pairs for values at given nodes in the full stacked scaled 3d
points_scaled_2D = full_scaled_stack3D[nodes[:, 0], nodes[:, 1]]

# Calculates best bw
params = {'bandwidth': np.linspace(0.1, 2.0, 100)}
grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=5)
grid.fit(points_scaled_2D)
best_bw = grid.best_estimator_.bandwidth
print("Best bandwidth:", best_bw)


# Permorm Kernel Density Estimation at the points
kde = KernelDensity(kernel='gaussian', bandwidth=best_bw)
kde.fit(points_scaled_2D)

# Create a copy of the full scaled 2d stack, 
kde_map = a_full_unscaled.copy()
map_mask = np.isnan(a_full_unscaled)

log_density = kde.score_samples(valid_scaled_stack2D)
log_density -= log_density.max()
density = np.exp(log_density)
density = (density - density.min()) / (density.max() - density.min())

kde_map[~map_mask] = density



profile.update(
    driver="GTiff",          
    dtype=rasterio.float32,
    count=1,
    nodata=-9999
)

with rasterio.open(f"{base}/data/processed/kde_cdo.tif", "w", **profile) as dst:
    dst.write(kde_map.astype(np.float64), 1)


plt.imshow(kde_map, cmap='hot')  # 'hot', 'viridis', 'terrain', etc.
plt.colorbar(label='KDE Value')
plt.title("KDE Density Map")
plt.show()


print("saved")


