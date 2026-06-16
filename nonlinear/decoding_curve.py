import os
import numpy as np
import pyntbci
import torch

from pipeline import EEGNet_pipeline

# Set up GPU if available
mps = torch.backends.mps.is_available()
device = "mps" if mps else "cpu"
print("GPU is", "AVAILABLE" if mps else "NOT AVAILABLE")

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset

subjects = [f"sub-{1+i:02d}" for i in range(10)]  # all participants

# Set parameters
fs = 240
fr = 60
n_trials = 100 
n_channels = 8
epochtime = 0.5
trialtime = 31.5  # limit trials to a certain duration in seconds
intertrialtime = 1.0  # ITI in seconds for computing ITR

# Set epoch size
encoding_length = int(epochtime * fs)
encoding_stride = int(1 / fr * fs)

# Set number of epochs
n_epochs = int(round((fs*trialtime - encoding_length)/2 + 1))

# Chronological cross-validation
n_folds = 5
folds = np.repeat(np.arange(n_folds), n_trials / n_folds)

segmenttime = 0.7  # step size of the decoding curve in seconds
segments = np.concatenate([np.arange(0.1, 4.2, 0.1), np.arange(4.2, trialtime, segmenttime)])

# Account for the time constraint of 500 ms (0.5 s) for the decoding curve
index = np.where(np.isclose(segments, encoding_length/fs))[0][0]
segments = segments[index:]
n_segments = segments.size

# Loop subjects
for subject in subjects:
    # Load data
    fn = os.path.join(path, "preprocess", "offline", "eegnet", "240_500", subject, f"{subject}_gdf.npz")

    tmp = np.load(fn)
    X = tmp["X"]
    y = tmp["y"]
    V = tmp["V"]
    y_trial = tmp["y_trial"]

    # Reshape data to trial level
    X_ = X.reshape((n_trials, -1, X.shape[1], X.shape[2]))
    y_ = y.reshape((n_trials, -1))

    # Set up codebook for trial classification
    n = int(np.ceil(int(31.5 * fs) / V.shape[1]))
    _V = np.tile(V, (1, n)).astype("float32")[:, ::encoding_stride]
    _V = _V[:, :n_epochs] # align epochs

    # Loop folds
    accuracy_trial = np.zeros((n_folds, n_segments))
    for i_fold in range(n_folds):
        # Split data to train and valid set
        X_trn = X_[folds != i_fold, :n_epochs, :, :]
        X_tst = X_[folds == i_fold, :n_epochs, :, :]
        y_trn = y_[folds != i_fold, :n_epochs]
        y_tst = y_[folds == i_fold, :n_epochs]
        y_trn_trial = y_trial[folds != i_fold]
        y_tst_trial = y_trial[folds == i_fold]

        valid_size = 20  # one full block = 20 trials

        # Separate validation trails from the training set
        X_val = X_trn[-valid_size:]
        y_val = y_trn[-valid_size:]

        X_trn = X_trn[:-valid_size]
        y_trn = y_trn[:-valid_size]

        # Initialize pipeline
        pipe = EEGNet_pipeline(
                                n_times=encoding_length,
                                X_valid=X_val.reshape((-1, n_channels, encoding_length)), 
                                y_valid=y_val.flatten(),
                                F1=4,
                                device=device
                            )

        # Train classifier
        pipe.fit(X_trn.reshape((-1, n_channels, encoding_length)), y_trn.flatten())

        #Loop segments
        for i_segment in range(n_segments):
            # Apply pipeline (on epoch level)
            yh_tst = pipe.predict(X_tst[:, : int(round((fs*segments[i_segment] - encoding_length)/2 + 1)), :, :].reshape((-1, n_channels, encoding_length)))

            # Apply pipeline (on trial level)
            ph_tst = pipe.predict_proba(X_tst[:, : int(round((fs*segments[i_segment] - encoding_length)/2 + 1)), :, :].reshape((-1, n_channels, encoding_length)))[:, 1].reshape(y_tst[:, : int(round((fs*segments[i_segment] - encoding_length)/2 + 1))].shape)
            
            # Compute accuracy
            rho = pyntbci.utilities.correlation(ph_tst, _V[:, : int(round((fs*segments[i_segment] - encoding_length)/2 + 1))])
            yh_tst = np.argmax(rho, axis=1)
            accuracy_trial[i_fold, i_segment] = np.mean(yh_tst == y_tst_trial)

    # Compute ITR
    time = np.tile(segments[np.newaxis, :], (n_folds, 1))
    itr = pyntbci.utilities.itr(V.shape[0], accuracy_trial, time + intertrialtime) 

    # Create output folder
    if not os.path.exists(os.path.join(path, "decoding_curve", "offline", "full", "eegnet-4-2", subject)): # change according to # of temporal filters
        os.makedirs(os.path.join(path, "decoding_curve", "offline", "full", "eegnet-4-2", subject))
        
    # Save data
    np.savez(os.path.join(path, "decoding_curve", "offline", "full", "eegnet-4-2", subject, f"{subject}_gdf.npz"), itr=itr, acc_trial=accuracy_trial, segments=segments)
