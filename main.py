import os
import numpy as np
import utils

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset
# subject = "sub-01"

subjects = [f"sub-{1+i:02d}" for i in range(10)] 

# def get_decocing_curve_data(file):
#     fn = os.path.join(path, "decoding_curve", "offline", file, subject, f"{subject}_gdf.npz")
#     tmp = np.load(fn)

#     itr = tmp["itr"]
#     acc_trial = tmp["acc_trial"]
#     segments = tmp["segments"]
#     name = file

#     return itr, acc_trial, segments, name


# def get_learning_curve_data(file):
#     fn = os.path.join(path, "learning_curve", "offline", "short", file, subject, f"{subject}_gdf.npz")
#     tmp = np.load(fn)

#     accuracy = tmp["accuracy"]
#     train_trials = tmp["train_trials"]
#     name=file

#     return accuracy, train_trials, name


# for subject in subjects:
    
    # accuracy, train_trials, name = get_learning_curve_data("eegnet_4_2")
    # accuracy_1, _, name_1 = get_learning_curve_data("rcca")
    # accuracy_2, _, name_2 = get_learning_curve_data("eegnet_8_2")
    
    # utils.display_learning_curve([(accuracy, name), (accuracy_1, name_1), (accuracy_2, name_2)], train_trials)

results = utils.collect_multi_subject_data(path, subjects)

utils.plot_grouped_accuracy(results, is_full=True, uncertainty="std")

curve_results = utils.collect_multi_subject_decoding_curves(path, subjects)

utils.display_multi_subject_decoding_curve(
    curve_results,
    version="full",
    models=["eegnet_8_2", "eegnet_4_2", "rcca"],
    uncertainty="std"
)
  
learning_curve_results = utils.collect_multi_subject_learning_curves(path, subjects)

utils.display_multi_subject_learning_curve(
    learning_curve_results,
    version="full",
    models=["eegnet_8_2", "eegnet_4_2", "rcca"],
    uncertainty="std"
)
