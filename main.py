import os
import numpy as np
import utils

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset
# subject = "sub-01"

subjects = [f"sub-{1+i:02d}" for i in range(10)] 

# fn = os.path.join(path, "train", "offline", "eegnet_8_2", subject, f"{subject}_gdf.npz")
# tmp = np.load(fn)
# acc_epoch = tmp["acc_epoch"]
# acc_trial = tmp["acc_trial"]

# utils.display_accuracy(acc_epoch, acc_trial)

def get_decocing_curve_data(file):
    fn = os.path.join(path, "decoding_curve", "offline", file, subject, f"{subject}_gdf.npz")
    tmp = np.load(fn)

    itr = tmp["itr"]
    acc_trial = tmp["acc_trial"]
    segments = tmp["segments"]
    name = file

    return itr, acc_trial, segments, name


def get_learning_curve_data(file):
    fn = os.path.join(path, "learning_curve", "offline", file, subject, f"{subject}_gdf.npz")
    tmp = np.load(fn)

    accuracy = tmp["accuracy"]
    train_trials = tmp["train_trials"]
    name=file

    return accuracy, train_trials, name


def get_training_accuracy(file):
    fn = os.path.join(path, "train", "offline", file, subject, f"{subject}_gdf.npz")
    tmp = np.load(fn)

    accuracy_epoch = tmp["acc_epoch"]
    accuracy_trial = tmp["acc_trial"]

    return accuracy_epoch, accuracy_trial

for subject in subjects:
    itr, acc_trial, segments, name = get_decocing_curve_data("eegnet_full_8_2")
    itr_1, acc_trial_1, _, name_1 = get_decocing_curve_data("eegnet_full_4_2")
    itr_2, acc_trial_2, _, name_2 = get_decocing_curve_data("rcca_full")

    utils.display_decoding_curve([(itr, acc_trial, name), 
                                  (itr_1, acc_trial_1, name_1), 
                                  (itr_2, acc_trial_2, name_2)], 
                                  segments, subject)
    
    # accuracy, train_trials, name = get_learning_curve_data("eegnet_8_2")
    # accuracy_1, _, name_1 = get_learning_curve_data("rcca")
    
    # utils.display_learning_curve([(accuracy, name), (accuracy_1, name_1)], train_trials)

    # accuracy_epoch, accuracy_trial = get_training_accuracy("eegnet_short_8_2")

    # utils.display_accuracy(accuracy_epoch, accuracy_trial)


    