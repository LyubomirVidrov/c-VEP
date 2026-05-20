import torch

from moabb.utils import setup_seed
from braindecode import EEGClassifier
from braindecode.models import EEGNet
from skorch.dataset import Dataset
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.helper import predefined_split

from sklearn.pipeline import make_pipeline

def EEGNet_pipeline(n_times, X_valid=None, y_valid=None, n_channels=8, F1=8, D=2, seed=42, device="cpu"):

    setup_seed(seed)
    
    LEARNING_RATE =  0.0625 * 0.01 # parameter taken from Braindecode 0.0625 * 0.01
    BATCH_SIZE = 64  # parameter taken from BrainDecode
    EPOCH = 10
    PATIENCE = 3

    model = EEGNet(
        n_chans = n_channels,
        n_outputs = 2,
        n_times = n_times, # epoch length in samples
        F1 = F1, 
        D=D,
        F2 = F1 * D,
        kernel_length=n_times // 2,
        drop_prob = 0.25
    )

    if X_valid is None:
        train_split = None
    else:
        train_split = predefined_split(Dataset(X_valid, y_valid))

    clf = EEGClassifier(
        module=model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=EPOCH, 
        train_split=train_split, # stratified=True
        device=device,
        callbacks=[
            EarlyStopping(monitor="valid_loss", patience=PATIENCE),
            EpochScoring(
                scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
            ),
            EpochScoring(
                scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
            ),
        ] if X_valid is not None else [
            EpochScoring("accuracy", on_train=True, name="train_acc"),
        ],
        verbose=1,  # Not printing the results for each epoch
    )

    # Create the pipelines
    pipes = {}
    pipes["EEGNet"] = make_pipeline(clf)

    return pipes['EEGNet']