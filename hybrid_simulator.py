import core.point_simulator as tri
from pandas import read_csv as rd
import numpy as np
import core.machine_learner as ml


path = r"C:\Users\KennethBaluyos (EPA)\Documents\04 - Manifesting RSTF\Projects\Project Reworked\test\0.5\1"
path_arff = f"{path}/0.5_trainingSet_1.arff"




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

points = rd(f"{base}/data/raw/input.csv")
coords = points[["LONG", "LAT"]].to_numpy()

hds_arff_array = []
bcs_arff_array = []
kdes_arff_array = []
ebcs_arff_array = []
ehds_arff_array = []
methods_arffArrayHolder = [hds_arff_array, bcs_arff_array, kdes_arff_array, ebcs_arff_array, ehds_arff_array]
methods_list = np.array(['hds','bcs','kdes', 'ebcs', 'ehds'])
#['hds','bcs','kdes', 'ebcs', 'ehds']

for method, array_holder in zip(methods_list, methods_arffArrayHolder):
    for i in range(10):
        arff_array = tri.simulate_landslides(
            raster_paths=[elev,slope,aspect,planCurvature, profCurvature, twi, relRel, distRiv],
            dem_path=elev,
            lands_coords_xy=coords,
            env_bandwidth='best',
            sampling_method=method,
            buffer_m=1000,
            n_simulated=750,    
            n_dataset=1,
            threshold=(i)/10,
            positiveOut_path = basePath,
            negativeOut_path = basePath,
            arff_path = basePath,
            output_kde_map_path=basePath,
            arff_header=np.array(["x", "y", "Elevation", "Slope", "Aspect", "Planform_Curvature", "Profile_Curvature","TWI", "Relative_Relief", "Distance_to_River"]),
            validationDS = f"{basePath}/validationDS_scale.arff",
            scaleValues=True
            )
        
        #arff_array now contains the path of all the arff data set for the current threshold
        #we'll save it to another method array holder to be used for training later
        array_holder.append(arff_array)
        
        print(f"Points Simulation #{i+1} Done using '{method}' method.")

for modelType in np.array(["LR", "RF", "RSS"]):
    if modelType == "LR":
        cName = "weka.classifiers.functions.Logistic"
        optionList = ["-R", "1.0E-8", "-M","-1","-num-decimal-places","4"]
    elif modelType == "RF":
        cName = "weka.classifiers.trees.RandomForest"
        optionList = ["-P", "100", "-I", "100", "-num-slots", "1", "-K", "0", "-M", "1.0","-V", "0.001", "-S", "1"]
    elif modelType == "RSS":
        cName = "weka.classifiers.meta.RandomSubSpace"
        optionList = ["-P", "0.5", "-S", "1", "-num-slots", "1", "-I", "10", "-W", "weka.classifiers.trees.RandomForest","--","-P", "100", "-I", "100", "-num-slots", "1" ,"-K", "0","-M","1.0","-V", "0.001","-S","1"]

    for method, pathArray in zip(methods_list, methods_arffArrayHolder):
        for i, arff_array in enumerate(pathArray):
            threshold = (i/10)
            ml.train_weka_model(
                    arff_array=arff_array,
                    model_output=f"{basePath}/{method}/{threshold}/{threshold}_{modelType}model.model",
                    cv_output=f"{basePath}/{method}/{threshold}/{threshold}_{modelType}model_CV.txt",
                    eval_output=f"{basePath}/{method}/{threshold}/{threshold}_{modelType}model_EVALUATION.txt",
                    eval_data = f"{basePath}/validationDS_scale.arff",
                    classifier_name=cName,
                    options=optionList
            )

            print(f"[DONE] {modelType} modelling for {method} sampling - threshold {threshold}")
