import os
import numpy as np
import pyntbci

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset

subjects = [f"sub-{1+i:02d}" for i in range(10)]  # all participants

# Loop subjects
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

    # Create structure matrix
    encoding_length = int(0.5 * fs)  # 500 ms responses
    M = pyntbci.utilities.encoding_matrix(E, encoding_length)   

    trialtime = 31.5  # limit trials to a certain duration in seconds
    intertrialtime = 1.0  # ITI in seconds for computing ITR
    n_samples = int(trialtime * fs)

    # Chronological cross-validation
    n_folds = 5
    folds = np.repeat(np.arange(n_folds), int(n_trials / n_folds))

    segmenttime = 0.7  # step size of the decoding curve in seconds
    segments = np.concatenate([np.arange(0.1, 4.2, 0.1), np.arange(4.2, trialtime, segmenttime)])
    n_segments = segments.size

    # Loop folds
    accuracy = np.zeros((n_folds, n_segments))
    for i_fold in range(n_folds):

        # Split data to train and test set
        X_trn, y_trn = X[folds != i_fold, :, :n_samples], y[folds != i_fold]
        X_tst, y_tst = X[folds == i_fold, :, :n_samples], y[folds == i_fold]

        excess = 20  # one full block = 20 trials (used for validation)

        # Remove validation trails from the training set 
        X_trn = X_trn[:-excess]
        y_trn = y_trn[:-excess]

        # Setup classifier
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=0.5, onset_event=True)

        # Train classifier
        rcca.fit(X_trn, y_trn)

        # Loop segments
        for i_segment in range(n_segments):
            # Apply classifier
            yh_tst = rcca.predict(X_tst[:, :, :int(fs * segments[i_segment])])

            # Compute accuracy
            accuracy[i_fold, i_segment] = np.mean(yh_tst == y_tst)

    # Compute ITR
    time = np.tile(segments[np.newaxis, :], (n_folds, 1))
    itr = pyntbci.utilities.itr(n_classes, accuracy, time + intertrialtime)

    # Create output folder
    if not os.path.exists(os.path.join(path, "decoding_curve", "offline", "full", "rcca", subject)):
        os.makedirs(os.path.join(path, "decoding_curve", "offline", "full", "rcca", subject))
        
    # Save data
    np.savez(os.path.join(path, "decoding_curve", "offline", "full", "rcca", subject, f"{subject}_gdf.npz"), itr=itr, acc_trial=accuracy, segments=segments)
