import matplotlib.pyplot as plt
import numpy as np

def display_accuracy(accuracy_epoch, accuracy_trial, n_folds=5):
    # Plot epoch accuracy (over folds)
    plt.figure(figsize=(15, 3))
    plt.bar(np.arange(n_folds), accuracy_epoch)
    plt.hlines(np.mean(accuracy_epoch), -.5, n_folds - 0.5, color="k")
    # plt.hlines(1 / 2, -.5, n_folds - 0.5, color="k", linestyle='dashed')
    plt.xlabel("(test) fold")
    plt.ylabel("accuracy")
    plt.title(f"EEGNet: accuracy (epoch): avg={np.mean(accuracy_epoch):.2f} std={np.std(accuracy_epoch):.2f}")

    plt.show()

    # Plot trial accuracy (over folds)
    plt.figure(figsize=(15, 3))
    plt.bar(np.arange(n_folds), accuracy_trial)
    plt.hlines(np.mean(accuracy_trial), -.5, n_folds - 0.5, color="k")
    # plt.hlines(1 / 20, -.5, n_folds - 0.5, color="k", linestyle='dashed')
    plt.xlabel("(test) fold")
    plt.ylabel("accuracy")
    plt.title(f"EEGNet: accuracy (trial): avg={np.mean(accuracy_trial):.2f} std={np.std(accuracy_trial):.2f}") 

    plt.show()


def display_decoding_curve(data, segments, subject):
    # Plot results
    fig, ax = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    
    for instance in data:
        avg = instance[1].mean(axis=0)
        std = instance[1].std(axis=0)
        ax[0].plot(segments, avg, linestyle='-', marker='o', label=instance[2])
        ax[0].fill_between(segments, avg + std, avg - std, alpha=0.2, label="_" + instance[2])
        avg = instance[0].mean(axis=0)
        std = instance[0].std(axis=0)
        ax[1].plot(segments, avg, linestyle='-', marker='o', label=instance[2])
        ax[1].fill_between(segments, avg + std, avg - std, alpha=0.2, label="_" + instance[2])

    ax[0].axhline(1 / 20, color="k", linestyle="--", alpha=0.5, label="chance")
    ax[1].set_xlabel("decoding time [s]")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("ITR [bits/min]")
    ax[0].legend()
    ax[0].set_title(f"Decoding curve - {subject}")
    fig.tight_layout()
    plt.show()


def display_learning_curve(data, train_trials, trialtime=4.2):
    # Plot results
    plt.figure(figsize=(15, 3))
    for instance in data:
        avg = instance[0].mean(axis=0)
        std = instance[0].std(axis=0)
        plt.plot(train_trials * trialtime, avg, linestyle='-', marker='o', label=instance[1])
        plt.fill_between(train_trials * trialtime, avg + std, avg - std, alpha=0.2, label="_" + instance[1])
    plt.axhline(1 / 20, color="k", linestyle="--", alpha=0.5, label="chance")
    plt.xlabel("learning time [s]")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Learning curve")
    plt.tight_layout()
    plt.show()