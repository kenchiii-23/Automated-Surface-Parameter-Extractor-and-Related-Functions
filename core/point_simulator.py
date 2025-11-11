import numpy as np
import rasterio
import core.sampler as smp
from rasterio import features
import core.machine_learner as ml
from rasterio.transform import Affine
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import Point, mapping
import pandas as pd
from sklearn.model_selection import GridSearchCV
import warnings
import os

   


def simulate_landslides(
    raster_paths, # Array of paths to raster (.tif)
    dem_path, # String of path to dem of are of interest
    lands_coords_xy,
    env_bandwidth=None,
    sampling_method="hds",
    buffer_m = 100,
    n_simulated=1000,
    n_dataset=10,
    threshold = 0.5,
    positiveOut_path="simulated_points.csv",
    negativeOut_path="simulated_points_neg.csv",
    arff_path = "none",
    output_kde_map_path=None,
    arff_header=[""],
    size_x=90,
    validationDS=None,
    scaleValues = False
):
    

    # ===== 1. Read the Raster and Stack them as 3D hypercube
    factors_full_unstacked = [] #array of unstacked feautures
    meta = None


    for path in raster_paths:
        with rasterio.open(path) as factor_raster:
            factor_full_unscaled = factor_raster.read(1).astype(float)

            if meta is None:
                meta = factor_raster.meta.copy()

            # Convert no data values to nan for easy working and masking later
            factor_ndVal = factor_raster.nodata

            if factor_ndVal is not None:
                factor_full_unscaled[factor_full_unscaled == factor_ndVal] = np.nan 
            
            factor_out = factor_full_unscaled

            if scaleValues == True:        
                mask = np.isnan(factor_full_unscaled)
                # Before we scale, we need to remove nans
                # This will return 1d array, of all valid unscaled. Position is unretained
                factor_valid_unscaled = factor_full_unscaled[~mask]
                factor_valid_unscaled= factor_valid_unscaled.reshape(-1,1)
                factor_full_scaled = factor_full_unscaled.copy()

                scaler = StandardScaler()
                factor_full_scaled[~mask] = scaler.fit_transform(factor_valid_unscaled).reshape(-1)
                factor_out = factor_full_scaled

            
            factors_full_unstacked.append(factor_out)
    
    # Stack the factors array --- (rows, cols, no_of_factors)
    factors_full_unscaled = np.stack(factors_full_unstacked, axis=-1)
    dem_rows, dem_cols, no_of_factors = factors_full_unscaled.shape
    total_pixels = dem_rows * dem_cols

    # Get transform and crs from dem_path (or meta)
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_transform = src.transform
        dem_profile = src.profile
        dem_crs = src.crs
        dem_ndVal = src.nodata
    
    # --- 2) Flatten and mask NaNs ---
    factors_full_unscaled2D = factors_full_unscaled.reshape(-1, no_of_factors)  # (M, F)
    valid_mask = ~np.any(np.isnan(factors_full_unscaled2D), axis=1)  # True = valid
    if valid_mask.sum() == 0:
        raise ValueError("No valid pixels found in raster stack.")

    factors_valid_unscaled2d = factors_full_unscaled2D[valid_mask]  # (M_valid, F)

    # --- 3) Standardize env features (fit on domain) ---
    scaler = StandardScaler()
    factors_valid_scaled2d = scaler.fit_transform(factors_valid_unscaled2d)


    # --- 4) Extract standardized env values for known landslide coords ---
    # convert XY to row,col indices
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

    # Convert lands_coords_xy -> rows, cols
    lands_coords_xy = np.asarray(lands_coords_xy)
    if lands_coords_xy.shape[1] != 2:
        raise ValueError("lands_coords_xy must be shape (n,2) of x,y coordinates")
    
    xs_known = lands_coords_xy[:,0] #slicer, turns the 1st col to 1d array
    ys_known = lands_coords_xy[:,1] # " " for 2nd col
    rows_known, cols_known = xy_to_rowcol(xs_known, ys_known, dem_transform)


    # Filter known points to valid pixel indices
    valid_idx_mask = (rows_known >= 0) & (rows_known < dem_rows) & (cols_known >= 0) & (cols_known < dem_cols)
    # rows_known < dem rows and __col____: checks if the index of the points is within the # of rows and/or col of dem
    # rows_known >= 0 and ____col____: checks if they are non-negative... ensures calculation was right from prev step

    if valid_idx_mask.sum() == 0:
        raise ValueError("None of the provided landslide coordinates fall inside raster extent.")
    
    if not valid_idx_mask.all(): # if something is invalid, filter .... 
        # filter works such that if one of the conditions is false, then remove that value across the lists
        rows_known = rows_known[valid_idx_mask] 
        cols_known = cols_known[valid_idx_mask]
        xs_known = xs_known[valid_idx_mask]
        ys_known = ys_known[valid_idx_mask]

    ##### !!!! rows,cols,xs,ys_known are filtered from here on #####


    flat_known_idx = rows_known * dem_cols + cols_known  
    # this returns an array of the indices of the xy known points in the flattened factor 3d cube


    # ensure they are valid pixels (not NaN in stack)
    valid_known_mask = valid_mask[flat_known_idx]
    # valid_mask is an array of bools that checks each row and col if theres nan
    # valid_mask [flat_known_idx] extracts the bool values at each index of the known points
    # this now creates a mask to be used to filter the points once again
    if not valid_known_mask.all():
        # drop invalid ones
        keep = np.where(valid_known_mask)[0]
        #returns the index of the points for which the validity check held true

        if len(keep) == 0:
            raise ValueError("All known landslide pixels contain NaNs in feature stack.")
        
        flat_known_idx = flat_known_idx[keep]
        rows_known = rows_known[keep]
        cols_known = cols_known[keep]
        xs_known = xs_known[keep]
        ys_known = ys_known[keep]

    factors_known_unscaled = factors_full_unscaled2D[flat_known_idx]             # raw features at known points
    factors_known_scaled = scaler.transform(factors_known_unscaled)     # standardized known features

    # --- 5) Fit environmental KDE on known standardized features ---
    if env_bandwidth is None:
        env_bandwidth = 0.5

    elif env_bandwidth == 'best':
        params = {'bandwidth': np.linspace(0.1, 2.0, 100)}
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=5)
        grid.fit(factors_known_scaled)
        env_bandwidth = grid.best_estimator_.bandwidth

    kde_env = KernelDensity(kernel='gaussian', bandwidth=env_bandwidth)
    kde_env.fit(factors_known_scaled)

    # Score entire domain (log space) only for valid pixels
    log_env = np.full(factors_full_unscaled2D.shape[0], np.nan, dtype=float)
    log_env_sample = kde_env.score_samples(factors_valid_scaled2d)
    log_env[valid_mask] = log_env_sample

    log_env_sample -= log_env_sample.max()
    density_env = np.exp(log_env_sample)
    density_env = (density_env - density_env.min()) / (density_env.max() - density_env.min())


    log_env_map = np.full(factors_full_unscaled2D.shape[0], dem_ndVal, dtype=float)
    log_env_map[valid_mask] = density_env


    log_env_map = np.reshape(log_env_map, (dem_rows, dem_cols))

    os.makedirs(f"{positiveOut_path}/{sampling_method}/{threshold}/", exist_ok=True)
    outputMapPathENV = f"{output_kde_map_path}/{sampling_method}/{threshold}/{sampling_method}_ENV_map.asc"
    if outputMapPathENV:
        out_metaENV = meta.copy()
        out_metaENV.update({
            "driver": "AAIGrid" if output_kde_map_path.lower().endswith(".asc") else "GTiff",
            "height": dem_rows,
            "width": dem_cols,
            "count": 1,
            "dtype": "float32",
            "transform": dem_transform,
            "crs": dem_crs
        })

    

    with rasterio.open(outputMapPathENV, "w", **out_metaENV) as dst:
        dst.write(log_env_map.astype(np.float32), 1)
    


    # --- 6) Compute spatial component ---
    # Build grid XY for all pixels
    rr, cc = np.indices((dem_rows, dem_cols))

    # use rasterio.transform.xy for arrays
    xs_grid = np.empty((dem_rows, dem_cols), dtype=float)
    ys_grid = np.empty((dem_rows, dem_cols), dtype=float)
    for i in range(dem_rows):
        # vectorized row to xy for each col
        col_coords = np.arange(dem_cols)
        xrow, yrow = rasterio.transform.xy(dem_transform, np.full(dem_cols, i), col_coords, offset='center')
        # returns the array of the x coords and y coords of each pixel of the current row in dem grid
        
        xs_grid[i, :] = xrow
        ys_grid[i, :] = yrow
        #fills the x and y coords of the current row of dem grid

    xs_flat = xs_grid.ravel()
    ys_flat = ys_grid.ravel()

    

    # For spatial method or choosing the points in terms of similarity in x and y (kde) or by buffer
   
    log_spat = np.full(factors_full_unscaled2D.shape[0], -1, dtype=float)
    
    # Create a hard mask: distance to nearest known point <= buffer_m → eligible
    # Compute distance transform from river-like mask of known points
    # Make a binary image where known points are 1
    known_mask = np.zeros((dem_rows, dem_cols), dtype=np.uint8)
    known_mask[rows_known, cols_known] = 1

    # distance in pixels
    dist_pixels = ndimage.distance_transform_edt(1 - known_mask)

    # convert to meters
    dist_m = dist_pixels * size_x
    # create boolean mask eligible pixels
    if (sampling_method == 'ehds') | (sampling_method == 'ebcs'):
        buffer_m = (buffer_m) * (1-threshold)

    buffer_mask = dist_m <= buffer_m
    # spatial "score": 0 for not in buffer, 1 for in buffer (in log-space => log(1)=0, others -> -inf)
    log_spat[buffer_mask.ravel()] = 0.0
    # keep others as NaN (or very small) so they are not sampled
    

    # Score entire domain (log space) only for valid pixels
    log_spat_sample = log_spat[valid_mask]

    # to create equi of log_env
    log_spat_pass = np.full(factors_full_unscaled2D.shape[0], np.nan, dtype=float)
    log_spat_pass[valid_mask] = log_spat_sample

    density_spat = np.exp(log_spat_sample)
    density_spat = (density_spat - density_spat.min()) / (density_spat.max() - density_spat.min())

    log_spat_map = np.full(factors_full_unscaled2D.shape[0], dem_ndVal, dtype=float)
    log_spat_map[valid_mask] = density_spat
    log_spat_map = np.reshape(log_spat_map, (dem_rows, dem_cols))


    os.makedirs(f"{positiveOut_path}/{sampling_method}/{threshold}/", exist_ok=True)
    outputMapPathSPAT = f"{output_kde_map_path}/{sampling_method}/{threshold}/{sampling_method}_SPAT_map.asc"
    if outputMapPathSPAT:
        out_metaSPAT = meta.copy()
        out_metaSPAT.update({
            "driver": "AAIGrid" if outputMapPathSPAT.lower().endswith(".asc") else "GTiff",
            "height": dem_rows,
            "width": dem_cols,
            "count": 1,
            "dtype": "float32",
            "transform": dem_transform,
            "crs": dem_crs
        })
    

    with rasterio.open(outputMapPathSPAT, "w", **out_metaSPAT) as dst:
        dst.write(log_spat_map.astype(np.float32), 1)
    

    # --- 7) Combine env + spatial in log-space ---
    if (sampling_method == 'hds') | (sampling_method == 'ehds'):
        valid_both = valid_mask & ~np.isnan(log_env) & ~np.isnan(log_spat_pass)
        if valid_both.sum() == 0:
            raise ValueError("No pixels have both environmental and spatial scores. Check inputs.")
        
        # shift each component (use only valid entries)
        le = log_env[valid_both]
        ls = log_spat_pass[valid_both]

        # numeric stabilization: subtract each max
        le = le - np.max(le)
        ls = ls - np.max(ls)

        joint = le + ls

    elif (sampling_method == 'bcs') | (sampling_method == 'ebcs'):
        valid_both = valid_mask & ~np.isnan(log_spat_pass)
        if valid_both.sum() == 0:
            raise ValueError("No pixels have both environmental and spatial scores. Check inputs.")

        # shift each component (use only valid entries)
        ls = log_spat_pass[valid_both]
        # numeric stabilization: subtract each max
        ls = ls - np.max(ls)
        joint = ls

    elif sampling_method == 'kdes':
        valid_both = valid_mask & ~np.isnan(log_env)
        if valid_both.sum() == 0:
            raise ValueError("No pixels have both environmental and spatial scores. Check inputs.")
        
        # shift each component (use only valid entries)
        le = log_env[valid_both]
        # numeric stabilization: subtract each max
        le = le - np.max(le)
        joint = le

    joint -= joint.max()
    density = np.exp(joint)
    density = (density - np.min(density)) / (np.max(density) - np.nanmin(density))
    density_map = np.full(factors_full_unscaled2D.shape[0], dem_ndVal, dtype=float)
    density_map[valid_both] = density
    density_map_norm = np.reshape(density_map, (dem_rows, dem_cols))
    

    # --- 8) If buffer method: ensure we only sample inside buffer -->
    # (already ensured via log_spat for buffer: outside buffer had NaN->density 0)
    probs = density_map_norm.ravel() 
    probs_n = density_map_norm.ravel()
    probs_n = 1 - probs_n # flips probs such that the unlikely (say .1) becomes likely (.9)
    # mask invalid pixels
    
    factorMask = np.ones(factors_full_unstacked[0].shape, dtype=bool)
    factorMask = factorMask.ravel()
    for factor in factors_full_unstacked:
        factorFlat = np.ravel(factor)
        factorMask &= np.isfinite(factorFlat) & (factorFlat != -9999)
    

    probs_mask = (probs >= threshold) 
    probs_mask_n = (probs_n >= threshold) & (probs_n <= 1)

    idx_pool = np.where(probs_mask)[0]
    idx_pool_n = np.where(probs_mask_n)[0]
    if idx_pool.size == 0:
        raise ValueError("No candidate pixels to sample positive from (probabilities all zero). Try increasing bandwidth or buffer.")

    if idx_pool_n.size == 0:
        raise ValueError("No candidate pixels to sample negative from (probabilities all zero). Try increasing bandwidth or buffer.")
    
    # Normalize probabilities over candidate pool
    p = probs[idx_pool].astype(float)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    if p.sum() == 0:
        raise ValueError("Sum of positive probabilities is zero. Check KDE or threshold.")
    p = p / p.sum()

    # Normalize probabilities over candidate pool (negative)
    p_n = probs_n[idx_pool_n].astype(float)
    p_n = np.nan_to_num(p_n, nan=0.0, posinf=0.0, neginf=0.0)
    if p_n.sum() == 0:
        raise ValueError("Sum of negative probabilities is zero. Check KDE or threshold.")
    p_n = p_n / p_n.sum()

    arff_array = []
    for i in range(n_dataset):
        # sample (without replacement)
        if idx_pool.size < n_simulated:
            replace = True
        else:
            replace = False

        n_draw = min(n_simulated, idx_pool.size)
        chosen_flat = np.random.choice(idx_pool, size=n_draw, replace=replace, p=p)
        #print(f"Pos Pool Size: {idx_pool.size}, Replace: {replace}, Chosen Size: {chosen_flat.size}, Draw Size: {n_draw}")

        # sample (without replacement)
        n_draw_n = min(n_simulated, idx_pool_n.size)
        if idx_pool_n.size < n_simulated:
            replace_n = True

        else:
            replace_n = False
        chosen_flat_n = np.random.choice(idx_pool_n, size=n_draw_n, replace=replace_n, p=p_n)
        #print(f"Neg Pool Size: {idx_pool_n.size}, Replace: {replace_n}, Chosen Size: {chosen_flat_n.size}")

        if chosen_flat.size > chosen_flat_n.size:
            print(f"Positive Sample size {chosen_flat.size} resampled to {chosen_flat_n.size}")
            new = np.random.choice(chosen_flat, size=chosen_flat_n.size, replace=False)
            chosen_flat = new
        elif chosen_flat.size < chosen_flat_n.size:
            print(f"Negative Sample size {chosen_flat_n.size} resampled to {chosen_flat.size}")
            new = np.random.choice(chosen_flat_n, size=chosen_flat.size, replace=False)
            chosen_flat_n = new

        chosen_rows = chosen_flat // dem_cols
        chosen_cols = chosen_flat % dem_cols
        chosen_xs = xs_flat[chosen_flat]
        chosen_ys = ys_flat[chosen_flat]
        chosen_probs = probs[chosen_flat]

        chosen_rows_n = chosen_flat_n // dem_cols
        chosen_cols_n = chosen_flat_n % dem_cols
        chosen_xs_n = xs_flat[chosen_flat_n]
        chosen_ys_n = ys_flat[chosen_flat_n]
        chosen_probs_n = probs[chosen_flat_n]


        df = pd.DataFrame({
            "x": chosen_xs,
            "y": chosen_ys,
            "row": chosen_rows,
            "col": chosen_cols,
            "prob": chosen_probs
        })

        df_n = pd.DataFrame({
            "x": chosen_xs_n,
            "y": chosen_ys_n,
            "row": chosen_rows_n,
            "col": chosen_cols_n,
            "prob": chosen_probs_n
        })
        
        os.makedirs(f"{positiveOut_path}/{sampling_method}/{threshold}/{i+1}/", exist_ok=True)
        os.makedirs(f"{negativeOut_path}/{sampling_method}/{threshold}/{i+1}/", exist_ok=True)
        os.makedirs(f"{arff_path}/{sampling_method}/{threshold}/{i+1}/", exist_ok=True)

        df.to_csv(positiveOut_path + f"/{sampling_method}/{threshold}/{i+1}/{threshold}_Positive_{i+1}_{np.round(env_bandwidth, 2)}.csv", index=False)
        df_n.to_csv(negativeOut_path + f"/{sampling_method}/{threshold}/{i+1}/{threshold}_Negative_{i+1}_{np.round(env_bandwidth,2)}.csv", index=False)

        extractedFactors = smp.extractPointFactor(factors_full_unstacked, arff_header, chosen_xs, chosen_ys, dem_profile)
        extractedFactors_n = smp.extractPointFactor(factors_full_unstacked, arff_header, chosen_xs_n, chosen_ys_n, dem_profile)
              
        dataset_outPath = f"{arff_path}/{sampling_method}/{threshold}/{i+1}/{threshold}_trainingSet_{i+1}.arff"
        with open(f"{arff_path}/{sampling_method}/{threshold}/{i+1}/{threshold}_trainingSet_{i+1}.arff", "w") as file:
            file.write(f"@relation SimulatedPoints_{i}\n\n")
            for k, factor in enumerate(arff_header):
                if k > 1:
                    file.write(f"@attribute {factor} numeric\n")

            file.write("@attribute occurence {0, 1}\n\n")
            file.write("@data\n")

            for row in extractedFactors[:, 2:]:
                # Convert each row to comma-separated string
                line = ",".join(map(str, row))
                file.write(line)
                file.write(", 1\n")

            for row in extractedFactors_n[:, 2:]:
                # Convert each row to comma-separated string
                line = ",".join(map(str, row))
                file.write(line)
                file.write(", 0\n")

        arff_array.append(dataset_outPath)
    

    """for modelType in np.array(["LR", "RF", "RSS"]):
        if modelType == "LR":
            cName = "weka.classifiers.functions.Logistic"
            optionList = ["-R", "1.0E-8", "-M","-1","-num-decimal-places","4"]
        elif modelType == "RF":
            cName = "weka.classifiers.trees.RandomForest"
            optionList = ["-P", "100", "-I", "100", "-num-slots", "1", "-K", "0", "-M", "1.0","-V", "0.001", "-S", "1"]
        elif modelType == "RSS":
            cName = "weka.classifiers.meta.RandomSubSpace"
            optionList = ["-P", "0.5", "-S", "1", "-num-slots", "1", "-I", "10", "-W", "weka.classifiers.trees.RandomForest","--","-P", "100", "-I", "100", "-num-slots", "1" ,"-K", "0","-M","1.0","-V", "0.001","-S","1"]
        
        ml.train_weka_model(
                arff_array=arff_array,
                model_output=f"{arff_path}/{sampling_method}/{threshold}/{threshold}_{modelType}model.model",
                cv_output=f"{arff_path}/{sampling_method}/{threshold}/{threshold}_{modelType}model_CV.txt",
                eval_output=f"{arff_path}/{sampling_method}/{threshold}/{threshold}_{modelType}model_EVALUATION.txt",
                eval_data = validationDS,
                classifier_name=cName,
                options=optionList
        )"""
    
    #weka.classifiers.functions.Logistic
    #"-R", "1.0E-8", "-M","-1","-num-decimal-places","4"
    
    #"weka.classifiers.meta.RandomSubSpace"
    #"-P", "0.5", "-S", "1", "-num-slots", "1", "-I", "10", "-W", "weka.classifiers.trees.RandomForest","--","-P", "100", "-I", "100", "-num-slots", "1" ,"-K", "0","-M","1.0","-V", "0.001","-S","1"

    #weka.classifiers.trees.RandomForest
    #"-P", "100", "-I", "100", "-num-slots", "1", "-K", "0", "-M", "1.0","-V", "0.001", "-S", "1"


        
        
    # --- 10) Optionally save KDE map as raster (AAIGrid or GeoTIFF depending on extension) ---
    outputMapPath = f"{output_kde_map_path}/{sampling_method}/{threshold}/{sampling_method}_map.asc"
    if outputMapPath:
        out_meta = meta.copy()
        out_meta.update({
            "driver": "AAIGrid" if output_kde_map_path.lower().endswith(".asc") else "GTiff",
            "height": dem_rows,
            "width": dem_cols,
            "count": 1,
            "dtype": "float32",
            "transform": dem_transform,
            "crs": dem_crs
        })


        with rasterio.open(outputMapPath, "w", **out_meta) as dst:
            dst.write(density_map_norm.astype(np.float32), 1)



    return arff_array
