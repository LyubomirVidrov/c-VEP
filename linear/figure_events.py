import os
import numpy as np
import pyntbci
import matplotlib.pyplot as plt

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset

subjects = [f"sub-{1+i:02d}" for i in range(1)]  # all participants

for subject in subjects:
    # Load data
    fn = os.path.join(path, "preprocess", "offline", "rcca", "240", subject, f"{subject}_gdf.npz")
    tmp = np.load(fn)

    X = tmp["X"]
    y = tmp["y"]
    V = tmp["V"]
    fs = int(tmp["fs"])
    fr = 60

    # Extract data dimensions
    n_trials, n_channels, n_samples = X.shape
    n_classes = V.shape[0]

    # Create event matrix
    E, events = pyntbci.utilities.event_matrix(V, event="duration", onset_event=True)

    print("E:", E.shape, "(classes x events x samples)", E.dtype)
    print("Events:", ", ".join([str(event) for event in events]))

    # Visualize event time-series
    i_class = 0  # the class to visualize
    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    pyntbci.plotting.eventplot(V[i_class, ::int(fs/fr)], E[i_class, :, ::int(fs/fr)], fs=fr, ax=ax, events=events)
    ax.set_title(f"Event time-series (code {i_class})")
    plt.tight_layout()
    plt.show()