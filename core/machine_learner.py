import weka.core.jvm as jvm
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
import weka.core.classes as wclasses
from weka.filters import Filter
import numpy as np
import random
import os

def train_weka_model(arff_array, model_output, cv_output, eval_output, eval_data, classifier_name="weka.classifiers.trees.J48", options=None):
    """
    Trains a Weka classifier from an ARFF file and saves the trained model.

    Parameters
    ----------
    arff_path : str
        Path to the ARFF dataset for training.
    model_output : str
        Path to save the trained .model file.
    classifier_name : str, optional
        Full class name of the Weka classifier (default: J48 decision tree).
    options : list[str], optional
        List of command-line style options for the classifier.
        Example: ["-C", "0.25", "-M", "2"]

    Returns
    -------
    classifier : weka.classifiers.Classifier
        The trained Weka classifier object.
    """
    # Ensure JVM is running
    jvm.start(packages=True)

    try:
        loader = Loader(classname="weka.core.converters.ArffLoader")
        combined_data = None

        # Load and merge datasets
        for arff_file in arff_array:
            data = loader.load_file(arff_file)
            data.class_is_last()
            if combined_data is None:
                combined_data = data
            else:
                for inst in data:
                    combined_data.add_instance(inst)
       
        class_index = combined_data.class_index  # integer index of the class attribute

        # Count positive (1) and negative (0) instances
        """positive_count = 0
        negative_count = 0
        for i in range(combined_data.num_instances):
            inst = combined_data.get_instance(i)
            class_val = int(inst.get_value(class_index))  # use integer index
            if class_val == 1:
                positive_count += 1
            elif class_val == 0:
                negative_count += 1

        minority_count = positive_count
        majority_count = negative_count
        
        print(f"Positive: {minority_count}, Negative: {majority_count}")
        if minority_count < majority_count:
            sampling = int(((majority_count/minority_count)-1)*(100))
            smote = Filter(classname="weka.filters.supervised.instance.SMOTE", options=["-P", str(sampling)])  # 100% oversampling
            smote.inputformat(combined_data)
            balanced_data = smote.filter(combined_data)
        elif majority_count < minority_count:
            sampling = int(((minority_count/majority_count)-1)*(100))
            smote = Filter(classname="weka.filters.supervised.instance.SMOTE", options=["-P", str(sampling)])  # 100% oversampling
            smote.inputformat(combined_data)
            balanced_data = smote.filter(combined_data)
        else: 
        """
        balanced_data = combined_data
        
        cls = Classifier(classname=classifier_name, options=options)
        cls.build_classifier(balanced_data)

        n_instances = combined_data.num_instances

        if n_instances < 50:
            evaluation = Evaluation(combined_data)
            rnd = wclasses.Random(50)  # for reproducibility

            evaluation.crossvalidate_model(cls, combined_data, n_instances, rnd)

            with open(cv_output, "w") as f:
                f.write("=== Leave-One Out Cross-Validation on Training Data ===\n\n")
                f.write(evaluation.summary() + "\n")
                f.write(evaluation.class_details() + "\n")
                f.write(evaluation.matrix() + "\n")

        else: 
            # --- 10-Fold Cross-Validation ---
            evaluation_cv = Evaluation(balanced_data)
            rnd = wclasses.Random(50)
            evaluation_cv.crossvalidate_model(cls, balanced_data, 10, rnd)
            with open(cv_output, "w") as f:
                f.write("=== 10-Fold Cross-Validation on Training Data ===\n\n")
                f.write(evaluation_cv.summary() + "\n")
                f.write(evaluation_cv.class_details() + "\n")
                f.write(evaluation_cv.matrix() + "\n")
        

        # --- Evaluate on separate test dataset ---
        test_data = loader.load_file(eval_data)
        test_data.class_is_last()
        evaluation_test = Evaluation(balanced_data)
        evaluation_test.test_model(cls, test_data)
        with open(eval_output, "w") as f:
            f.write("=== Evaluation on Test Dataset ===\n\n")
            f.write(evaluation_test.summary() + "\n")
            f.write(evaluation_test.class_details() + "\n")
            f.write(evaluation_test.matrix() + "\n")

        # --- Save the trained model ---
        cls.serialize(model_output)

    except Exception as e:
        print("[ERROR] Training failed:", str(e))
        raise e
    
