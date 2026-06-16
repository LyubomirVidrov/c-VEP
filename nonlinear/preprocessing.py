# from moabb.datasets import Thielen2021
# from moabb.paradigms import CVEP
import pyntbci
import scipy.io as sio
import numpy as np
import os

import sys
sys.path.insert(0, "/Users/lyubomirvidrov/Desktop/Thesis/MOABB")

from moabb.paradigms.cvep import CVEP
from moabb.datasets import Thielen2021

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset

# Configuration for the preprocessing
fs=240
FR=60

# Load dataset
dataset=Thielen2021()

# Load codes
gold_path = dataset.data_path(1)[-2]
codes = np.tile(sio.loadmat(gold_path)["codes"], (15, 1)).T

# Upsample codes from screen framerate to EEG sampling rate
V = sio.loadmat(gold_path)["codes"].T
V = np.repeat(V, fs // FR, axis=1).astype("uint8")

subjects = [f"sub-{1+i:02d}" for i in range(10)]  # all participants

# Configure the CVEP paradigm
paradigm = CVEP(
    fmin=1.0,
    fmax=45.0,
    tmin=0.0,
    tmax=0.5,
    resample=fs,
    events={"0.0": 0, "1.0": 1},
    n_classes=2,
)

# Loop subjects 
for i in range(10):
    # epoch data
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[i+1])

    # Preprocess data
    y = np.array([0 if label == '0.0' else 1 for label in y], dtype=np.int64)
    X = X[:, :, :int(0.5*fs)]
    X = X * 1e6
    
    # Calculate trial-wise labels
    y_trial = y.reshape(100, 1890)
    rho = pyntbci.utilities.correlation(y_trial, codes) 
    y_trial = np.argmax(rho, axis=1)

    # Create output folder
    if not os.path.exists(os.path.join(path, "preprocess", "offline", "eegnet", "240_500", subjects[i])): # change according to fs and tmax "fs_tmax"
        os.makedirs(os.path.join(path, "preprocess", "offline", "eegnet", "240_500", subjects[i]))
    
    # Save data
    np.savez(os.path.join(path, "preprocess", "offline", "eegnet", "240_500", subjects[i], f"{subjects[i]}_gdf.npz"), X=X, y=y, V=V, y_trial=y_trial)

    # Print summary
    print("Subject: ", subjects[i])
    print("\tX shape:", X.shape, "(epochs, channels, samples)")
    print("\ty shape:", y.shape, "(epochs)")
    print("\tV shape:", V.shape)
    print("\ty_trial shape:", y_trial.shape, "(trials)")
