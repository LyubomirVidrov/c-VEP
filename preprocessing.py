from moabb.datasets import Thielen2021
from moabb.paradigms import CVEP
import pyntbci
import scipy.io as sio
import numpy as np
import os

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset

fs=120
FR=60

dataset=Thielen2021()

gold_path = dataset.data_path(1)[-2]
codes = np.tile(sio.loadmat(gold_path)["codes"], (15, 1)).T

V = sio.loadmat(gold_path)["codes"].T
V = np.repeat(V, fs // FR, axis=1).astype("uint8")

subjects = [f"sub-{1+i:02d}" for i in range(10)]  # all participants

paradigm = CVEP(
    fmin=1.0,
    fmax=45.0,
    tmin=0.0,
    tmax=0.3,
    resample=120,
    events={"0.0": 0, "1.0": 1},
    n_classes=2,
)

for i in range(10):
    # tmp = np.load("/Users/lyubomirvidrov/data/thielen2021/derivatives/offline/" + subjects[i] + "/" + subjects[i] + "_gdf.npz")
    # V = tmp["V"][:, ::int(tmp["fs"] / 120)] # fs
    # y_trial = tmp["y"]

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[i+1])

    y = np.array([0 if label == '0.0' else 1 for label in y], dtype=np.int64)
    X = X * 1e6

    y_trial = y.reshape(100, 1890)
    rho = pyntbci.utilities.correlation(y_trial, codes)
    y_trial = np.argmax(rho, axis=1)

    # Create output folder
    if not os.path.exists(os.path.join(path, "preprocess", "offline", subjects[i])):
        os.makedirs(os.path.join(path, "preprocess", "offline", subjects[i]))
    
    # Save data
    np.savez(os.path.join(path, "preprocess", "offline", subjects[i], f"{subjects[i]}_gdf.npz"), X=X, y=y, V=V, y_trial=y_trial)

    # Print summary
    print("Subject: ", subjects[i])
    print("\tX shape:", X.shape, "(epochs, channels, samples)")
    print("\ty shape:", y.shape, "(epochs)")
    print("\tV shape:", V.shape)
    print("\ty_trial shape:", y_trial.shape, "(trials)")
