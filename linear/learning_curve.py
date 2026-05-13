import os
import numpy as np
import pyntbci

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset

subjects = [f"sub-{1+i:02d}" for i in range(3)]  # all participants

for subject in subjects:
    # Load data
    fn = os.path.join(path, "derivatives", "offline", subject, f"{subject}_gdf.npz")
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
    encoding_length = int(0.3 * fs)  # 300 ms responses
    M = pyntbci.utilities.encoding_matrix(E, encoding_length)   

    trialtime = 4.2  # limit trials to a certain duration in seconds
    n_samples = int(trialtime * fs)

    # Chronological cross-validation
    n_folds = 5
    folds = np.repeat(np.arange(n_folds), int(n_trials / n_folds))

    # Set learning curve axis
    # train_trials = np.arange(1, 1 + np.sum(folds != 0))
    train_trials = np.concatenate([
        np.arange(1, 11, 1),
        np.arange(12, 20, 2),
        np.arange(20, 81, 5)
    ])
    n_train_trials = train_trials.size

    # Loop folds
    accuracy = np.zeros((n_folds, n_train_trials))
    for i_fold in range(n_folds):

        # Split data to train and test set
        X_trn, y_trn = X[folds != i_fold, :, :n_samples], y[folds != i_fold]
        X_tst, y_tst = X[folds == i_fold, :, :n_samples], y[folds == i_fold]

        # Loop train trials
        for i_trial in range(n_train_trials):
            # Train classifier
            rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=0.3, onset_event=True)
            rcca.fit(X_trn[:train_trials[i_trial], :, :], y_trn[:train_trials[i_trial]])

            # Apply classifier
            yh_tst = rcca.predict(X_tst)

            # Compute accuracy
            accuracy[i_fold, i_trial] = np.mean(yh_tst == y_tst)
    
        # Create output folder
    if not os.path.exists(os.path.join(path, "learning_curve", "offline", "rcca", subject)):
        os.makedirs(os.path.join(path, "learning_curve", "offline", "rcca", subject))
        
    # Save data
    np.savez(os.path.join(path, "learning_curve", "offline", "rcca", subject, f"{subject}_gdf.npz"), accuracy=accuracy, train_trials=train_trials)