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


def plot_grouped_accuracy(summary, is_full=True, metric="acc_trial"):
    versions = ["120_300", "120_500", "240_300", "240_500"]

    if is_full:
        models = ["eegnet_full_8_2", "rcca"]
    else:
        models = ["eegnet_short_8_2", "rcca_short"]

    colors = ["#dbe9f6", "#9ecae1", "#4292c6", "#084594"]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, version in enumerate(versions):

        means = []
        stds_lower = []
        stds_upper = []

        for model in models:

            if model == "rcca" or model == "rcca_short":
                metric_name = "accuracy"
            else:
                metric_name = metric

            means.append(summary[version][model][metric_name]["mean"])
            stds_lower.append(summary[version][model][metric_name]["std_lower"])
            stds_upper.append(summary[version][model][metric_name]["std_upper"])

        means = np.array(means)
        lower_errors = np.array(stds_lower)
        upper_errors = np.array(stds_upper)

        ax.bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=[lower_errors, upper_errors],
            capsize=5,
            label=version,
            color=colors[i]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["eegnet", "rcca"])
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    if is_full:
        ax.set_title(f"Mean accuracy across subjects (trial length = 31.5s)")
    else:
        ax.set_title(f"Mean accuracy across subjects (trial length = 4.2s)")
    ax.legend(title="Version")

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def display_decoding_curve_from_dict(curve_results, version, models):
    fig, ax = plt.subplots(2, 1, figsize=(15, 5), sharex=True)

    for model in models:
        segments = curve_results[version][model]["segments"]

        acc_mean = curve_results[version][model]["accuracy"]["mean"]
        # acc_std = curve_results[version][model]["accuracy"]["std"]
        acc_std_lower = curve_results[version][model]["accuracy"]["std_lower"]
        acc_std_upper = curve_results[version][model]["accuracy"]["std_upper"]

        itr_mean = curve_results[version][model]["itr"]["mean"]
        # itr_std = curve_results[version][model]["itr"]["std"]
        itr_std_lower = curve_results[version][model]["itr"]["std_lower"]
        itr_std_upper = curve_results[version][model]["itr"]["std_upper"]

        ax[0].plot(segments, acc_mean, linestyle="-", marker="o", label=model)
        ax[0].fill_between(
            segments,
            # acc_mean - acc_std,
            # acc_mean + acc_std,
            acc_mean - acc_std_lower,
            acc_mean + acc_std_upper,
            alpha=0.2,
            label="_" + model
        )

        ax[1].plot(segments, itr_mean, linestyle="-", marker="o", label=model)
        ax[1].fill_between(
            segments,
            # itr_mean - itr_std,
            # itr_mean + itr_std,
            itr_mean - itr_std_lower,
            itr_mean + itr_std_upper,
            alpha=0.2,
            label="_" + model
        )

    ax[0].axhline(1 / 20, color="k", linestyle="--", alpha=0.5, label="chance")
    ax[1].set_xlabel("decoding time [s]")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("ITR [bits/min]")
    ax[0].legend()
    if version == "short":
        ax[0].set_title(f"Decoding curve across subjects (trial length = 4.2s)")
    else:
        ax[0].set_title(f"Decoding curve across subjects (trial length = 31.5s)")   

    fig.tight_layout()
    plt.show()

def collect_multi_subject_learning_curves(curve_results, version, models):
    fig, ax = plt.subplots(figsize=(15, 3))

    trialtime = 4.2 if version == "short" else 31.5
    
    for model in models:
        train_trials = curve_results[version][model]["train_trials"]

        acc_mean = curve_results[version][model]["accuracy"]["mean"]
        acc_std_lower = curve_results[version][model]["accuracy"]["std_lower"]
        acc_std_upper = curve_results[version][model]["accuracy"]["std_upper"]

        ax.plot(train_trials * trialtime, acc_mean, linestyle="-", marker="o", label=model)
        ax.fill_between(
            train_trials * trialtime,
            acc_mean - acc_std_lower,
            acc_mean + acc_std_upper,
            alpha=0.2,
            label="_" + model
        )

    ax.axhline(1 / 20, color="k", linestyle="--", alpha=0.5, label="chance")
    ax.set_xlabel("learning time [s]")
    ax.set_ylabel("accuracy")
    ax.legend()
    if version == "short":
        ax.set_title(f"Learning curve across subjects (trial length = 4.2s)")
    else:
        ax.set_title(f"Learning curve across subjects (trial length = 31.5s)")
    ax.legend(title="Version")

    plt.tight_layout()
    plt.show()
