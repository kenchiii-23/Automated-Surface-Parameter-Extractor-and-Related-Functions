import core.point_simulator as tri
from pandas import read_csv as rd
import numpy as np
import core.machine_learner as ml
import os

basePath = os.getcwd()
arff_array = [rf"{basePath}/test/control_dataset.arff"]

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
    for i in range(9):
        ml.train_weka_model(
                arff_array=arff_array,
                model_output=f"{basePath}/test/control/{modelType}/{i}_{modelType}model.model",
                cv_output=f"{basePath}/test/control/{modelType}/{i}_{modelType}model_CV.txt",
                eval_output=f"{basePath}/test/control/{modelType}/{i}_{modelType}model_EVALUATION.txt",
                eval_data = f"{basePath}/test/validationDS_scale.arff",
                classifier_name=cName,
                options=optionList
        )