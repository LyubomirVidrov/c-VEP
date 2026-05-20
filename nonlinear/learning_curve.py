import os
import numpy as np
import pyntbci
import torch

from pipeline import EEGNet_pipeline

# Set up GPU if it is there
mps = torch.backends.mps.is_available()
device = "mps" if mps else "cpu"
print("GPU is", "AVAILABLE" if mps else "NOT AVAILABLE")

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset

subjects = [f"sub-{1+i:02d}" for i in range(10)]  # all participants

fs = 240
fr = 60
n_trials = 100 
n_channels = 8
epochtime = 0.5
trialtime = 31.5  # limit trials to a certain duration in seconds

# Set epoch size
encoding_length = int(epochtime * fs)
encoding_stride = int(1 / fr * fs)

n_epochs = int(round((fs*trialtime - encoding_length)/2))

# Setup cross-validation
n_folds = 5
folds = np.repeat(np.arange(n_folds), n_trials / n_folds)

# train_trials = np.arange(1, 1 + np.sum(folds != 0))
train_trials = np.concatenate([
    np.arange(1, 11, 1),
    np.arange(12, 20, 2),
    np.arange(20, 81, 5)
])
n_train_trials = train_trials.size

for subject in subjects:
    # Load data
    fn = os.path.join(path, "preprocess", "offline", "eegnet", "noartifacts", "240_500", subject, f"{subject}_gdf.npz")

    tmp = np.load(fn)
    X = tmp["X"]
    y = tmp["y"]
    V = tmp["V"]
    y_trial = tmp["y_trial"]

    # pipe = EEGNet_pipeline(X)

    # Reshape data to trial level
    X_ = X.reshape((n_trials, -1, X.shape[1], X.shape[2]))
    y_ = y.reshape((n_trials, -1))

    # Set up codebook for trial classification
    n = int(np.ceil(int(31.5 * fs) / V.shape[1]))
    _V = np.tile(V, (1, n)).astype("float32")[:, ::encoding_stride]
    _V = _V[:, :n_epochs] # align epochs

    # Loop folds
    accuracy_trial = np.zeros((n_folds, n_train_trials))

    for i_fold in range(n_folds):
        # Split data to train and valid set
        X_trn = X_[folds != i_fold, :n_epochs, :, :]
        X_tst = X_[folds == i_fold, :n_epochs, :, :]
        y_trn = y_[folds != i_fold, :n_epochs]
        y_tst = y_[folds == i_fold, :n_epochs]
        y_trn_trial = y_trial[folds != i_fold]
        y_tst_trial = y_trial[folds == i_fold]

        valid_size = 20  # one full block = 20 trials

        X_val = X_trn[-valid_size:]
        y_val = y_trn[-valid_size:]

        X_trn = X_trn[:-valid_size]
        y_trn = y_trn[:-valid_size]

        # # Initialize pipeline
        # pipe = EEGNet_pipeline(
        #                         n_times=encoding_length,
        #                         X_valid=X_val.reshape((-1, n_channels, encoding_length)), 
        #                         y_valid=y_val.flatten(),
        #                         F1=8,
        #                         device=device
        #                     )

        # Loop train trials
        for i_trial in range(n_train_trials):

            # Initialize pipeline
            pipe = EEGNet_pipeline(
                                    n_times=encoding_length,
                                    X_valid=X_val.reshape((-1, n_channels, encoding_length)), 
                                    y_valid=y_val.flatten(),
                                    F1=8,
                                    device=device
                                )

            # Train pipeline (on epoch level)
            pipe.fit(X_trn[:train_trials[i_trial], :, :, :].reshape((-1, n_channels, encoding_length)), y_trn[:train_trials[i_trial], :].flatten())
            
            # Apply pipeline (on epoch level)
            yh_tst = pipe.predict(X_tst.reshape((-1, n_channels, encoding_length)))
        
            # Apply pipeline (on trial level)
            ph_tst = pipe.predict_proba(X_tst.reshape((-1, n_channels, encoding_length)))[:, 1].reshape(y_tst.shape)
            rho = pyntbci.utilities.correlation(ph_tst, _V[:, :n_epochs])
            yh_tst = np.argmax(rho, axis=1)
            accuracy_trial[i_fold, i_trial] = np.mean(yh_tst == y_tst_trial)

        print(f"Now running: Fold {i_fold}...")

        # Create output folder
    if not os.path.exists(os.path.join(path, "learning_curve", "offline", "full", "eegnet_8_2", subject)):
        os.makedirs(os.path.join(path, "learning_curve", "offline", "full", "eegnet_8_2", subject))
        
    # Save data
    np.savez(os.path.join(path, "learning_curve", "offline", "full", "eegnet_8_2", subject, f"{subject}_gdf.npz"), accuracy=accuracy_trial, train_trials=train_trials)