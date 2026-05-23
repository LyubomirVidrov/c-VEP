import matplotlib.pyplot as plt
import numpy as np
import os


def _get_subject_learning_curve(path, version, model, subject):
    fn = os.path.join(
        path, "learning_curve", "offline",
        version, model, subject, f"{subject}_gdf.npz"
    )

    tmp = np.load(fn)

    return {
        "accuracy": tmp["accuracy"],   # shape: folds x train_trials
        "train_trials": tmp["train_trials"]
    }


def _get_subject_decoding_curve(path, version, model, subject):
    fn = os.path.join(
        path, "decoding_curve", "offline",
        version, model, subject, f"{subject}_gdf.npz"
    )

    tmp = np.load(fn)

    return {
        "segments": tmp["segments"],
        "accuracy": tmp["acc_trial"],   # shape: folds x segments
        "itr": tmp["itr"]              # shape: folds x segments
    }


def _get_subject_accuracy(path, version, model, subject):
    fn = os.path.join(
        path, "train", "offline", "noartifacts",
        version, model, subject, f"{subject}_gdf.npz"
    )

    tmp = np.load(fn)

    if model == "rcca" or model == "rcca_short":
        accuracy = np.mean(tmp["accuracy"])  # mean over 5 folds

        return {"accuracy": accuracy,}
    else:
        acc_epoch = np.mean(tmp["acc_epoch"])  # mean over 5 folds
        acc_trial = np.mean(tmp["acc_trial"])  # mean over 5 folds

        return {"acc_epoch": acc_epoch, "acc_trial": acc_trial,}
    

def _collect_subject_learning_curves(path, subjects, versionds, models):
    results = {}

    for version in versionds:
        results[version] = {}

        for model in models:
            results[version][model] = {}

            for subject in subjects:
                results[version][model][subject] = _get_subject_learning_curve(
                    path, version, model, subject
                )

    return results
    

def _collect_subject_decoding_curves(path, subjects, versions, models):
    results = {}

    for version in versions:
        results[version] = {}

        for model in models:
            results[version][model] = {}

            for subject in subjects:
                results[version][model][subject] = _get_subject_decoding_curve(
                    path, version, model, subject
                )

    return results


def _collect_subject_accuracies(path, subjects, versions, models):
    results = {}

    for version in versions:
        results[version] = {}

        for model in models:
            results[version][model] = {}

            for subject in subjects:
                results[version][model][subject] = _get_subject_accuracy(
                    path, version, model, subject
                )

    return results


def collect_multi_subject_learning_curves(path, subjects):
    versions = ["short", "full"]
    models = ["eegnet_4_2", "eegnet_8_2", "rcca"]

    subj_results = _collect_subject_learning_curves(path, subjects, versions, models)
    multi_subj_results = {}

    for version, version_data in subj_results.items():
        multi_subj_results[version] = {}

        for model, subject_data in version_data.items():
            multi_subj_results[version][model] = {}

            first_subject = list(subject_data.keys())[0]
            train_trials = subject_data[first_subject]["train_trials"]

            for metric in ["accuracy"]:

                vals = np.array([
                    subj_data[metric].mean(axis=0)   # mean over 5 folds first
                    for subj_data in subject_data.values()
                ])

                mean_val = np.nanmean(vals, axis=0)

                # keep only points below / above the across-subject mean
                lower_vals = np.where(vals <= mean_val, vals, np.nan)
                upper_vals = np.where(vals >= mean_val, vals, np.nan)

                # asymmetric SD around the global mean
                std_lower = np.sqrt(np.nanmean((lower_vals - mean_val) ** 2, axis=0))
                std_upper = np.sqrt(np.nanmean((upper_vals - mean_val) ** 2, axis=0))

                # number of subjects contributing to each side
                n_lower = np.sum(~np.isnan(lower_vals), axis=0)
                n_upper = np.sum(~np.isnan(upper_vals), axis=0)

                # asymmetric SE
                se_lower = std_lower / np.sqrt(n_lower)
                se_upper = std_upper / np.sqrt(n_upper)

                multi_subj_results[version][model][metric] = {
                    "mean": mean_val,
                    "std_lower": std_lower,
                    "std_upper": std_upper,
                    "se_lower": se_lower,
                    "se_upper": se_upper,
                    "values": vals
                }

            multi_subj_results[version][model]["train_trials"] = train_trials

    return multi_subj_results


def collect_multi_subject_decoding_curves(path, subjects):
    versions = ["short", "full"]
    models = ["eegnet_4_2", "eegnet_8_2", "rcca"]

    subj_results = _collect_subject_decoding_curves(path, subjects, versions, models)
    multi_subj_results = {}

    for version, version_data in subj_results.items():
        multi_subj_results[version] = {}

        for model, subject_data in version_data.items():
            multi_subj_results[version][model] = {}

            first_subject = list(subject_data.keys())[0]
            segments = subject_data[first_subject]["segments"]

            for metric in ["accuracy", "itr"]:

                vals = np.array([
                    subj_data[metric].mean(axis=0)   # mean over 5 folds first
                    for subj_data in subject_data.values()
                ])

                mean_val = np.nanmean(vals, axis=0)

                # keep only points below / above the across-subject mean
                lower_vals = np.where(vals <= mean_val, vals, np.nan)
                upper_vals = np.where(vals >= mean_val, vals, np.nan)

                # asymmetric SD around the global mean
                std_lower = np.sqrt(np.nanmean((lower_vals - mean_val) ** 2, axis=0))
                std_upper = np.sqrt(np.nanmean((upper_vals - mean_val) ** 2, axis=0))

                # number of subjects contributing to each side
                n_lower = np.sum(~np.isnan(lower_vals), axis=0)
                n_upper = np.sum(~np.isnan(upper_vals), axis=0)

                # asymmetric SE
                se_lower = std_lower / np.sqrt(n_lower)
                se_upper = std_upper / np.sqrt(n_upper)

                multi_subj_results[version][model][metric] = {
                    "mean": mean_val,
                    "std_lower": std_lower,
                    "std_upper": std_upper,
                    "se_lower": se_lower,
                    "se_upper": se_upper,
                    "values": vals
                }

            multi_subj_results[version][model]["segments"] = segments

    return multi_subj_results


def collect_multi_subject_data(path, subjects):
    versions = ["120_300", "120_500", "240_300", "240_500"]
    models = ["eegnet_full_8_2", "eegnet_short_8_2", "rcca", "rcca_short"]

    subj_results = _collect_subject_accuracies(path, subjects, versions, models)
    multi_subj_results = {}

    for version, version_data in subj_results.items():
        multi_subj_results[version] = {}

        for model, subject_data in version_data.items():
            multi_subj_results[version][model] = {}

            # metrics = list(next(iter(subject_data.values())).keys())
            first_subject = list(subject_data.keys())[0]
            metrics = subject_data[first_subject].keys()

            for metric in metrics:
                vals = np.array([
                    acc_data[metric]
                    for acc_data in subject_data.values()
                ])

                mean_val = np.nanmean(vals, axis=0)

                # keep only points below / above the across-subject mean
                lower_vals = np.where(vals <= mean_val, vals, np.nan)
                upper_vals = np.where(vals >= mean_val, vals, np.nan)

                # asymmetric SD around the global mean
                std_lower = np.sqrt(np.nanmean((lower_vals - mean_val) ** 2, axis=0))
                std_upper = np.sqrt(np.nanmean((upper_vals - mean_val) ** 2, axis=0))

                # number of subjects contributing to each side
                n_lower = np.sum(~np.isnan(lower_vals), axis=0)
                n_upper = np.sum(~np.isnan(upper_vals), axis=0)

                # asymmetric SE
                se_lower = std_lower / np.sqrt(n_lower)
                se_upper = std_upper / np.sqrt(n_upper)

                multi_subj_results[version][model][metric] = {
                    "mean": mean_val,
                    "std_lower": std_lower,
                    "std_upper": std_upper,
                    "se_lower": se_lower,
                    "se_upper": se_upper,
                    "values": vals
                }

    return multi_subj_results


def plot_grouped_accuracy(summary, is_full=True, metric="acc_trial", uncertainty="std"):
    versions = ["120_300", "120_500", "240_300", "240_500"]

    if is_full:
        models = ["eegnet_full_8_2", "rcca"]
    else:
        models = ["eegnet_short_8_2", "rcca_short"]

    colors_eegnet = ["#dbe9f6", "#9ecae1", "#4292c6", "#084594"]
    colors_rcca = ["#245501", "#538d22", "#73a942", "#aad576"]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, version in enumerate(versions):

        means = []
        lower = []
        upper = []
        colors = []

        for model in models:

            if model == "rcca" or model == "rcca_short":
                metric_name = "accuracy"
                colors.append(colors_rcca[i])
            else:
                metric_name = metric
                colors.append(colors_eegnet[i])

            means.append(summary[version][model][metric_name]["mean"])
            if uncertainty == "std":
                lower.append(summary[version][model][metric_name]["std_lower"])
                upper.append(summary[version][model][metric_name]["std_upper"])
            elif uncertainty == "se":
                lower.append(summary[version][model][metric_name]["se_lower"])
                upper.append(summary[version][model][metric_name]["se_upper"])

        means = np.array(means)
        lower_errors = np.array(lower)
        upper_errors = np.array(upper)

        ax.bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=[lower_errors, upper_errors],
            capsize=5,
            label=version,
            # color=colors
        )

    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(["eegnet", "rcca"])
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    if is_full:
        ax.set_title(f"Mean accuracy across subjects (trial length = 31.5s)")
    else:
        ax.set_title(f"Mean accuracy across subjects (trial length = 4.2s)")
    ax.legend(title="Version", loc="lower left")

    plt.tight_layout()
    plt.show()


def display_multi_subject_decoding_curve(curve_results, version, models, uncertainty="std"):
    fig, ax = plt.subplots(2, 1, figsize=(15, 5), sharex=True)

    for model in models:
        segments = curve_results[version][model]["segments"]

        acc_mean = curve_results[version][model]["accuracy"]["mean"]

        itr_mean = curve_results[version][model]["itr"]["mean"]

        if uncertainty == "std":
            acc_lower = curve_results[version][model]["accuracy"]["std_lower"]
            acc_upper = curve_results[version][model]["accuracy"]["std_upper"]

            itr_lower = curve_results[version][model]["itr"]["std_lower"]
            itr_upper = curve_results[version][model]["itr"]["std_upper"]
        elif uncertainty == "se":
            acc_lower = curve_results[version][model]["accuracy"]["se_lower"]
            acc_upper = curve_results[version][model]["accuracy"]["se_upper"]

            itr_lower = curve_results[version][model]["itr"]["se_lower"]
            itr_upper = curve_results[version][model]["itr"]["se_upper"]

        ax[0].plot(segments, acc_mean, linestyle="-", marker="o", label=model)
        ax[0].fill_between(
            segments,
            acc_mean - acc_lower,
            acc_mean + acc_upper,
            alpha=0.2,
            label="_" + model
        )

        ax[1].plot(segments, itr_mean, linestyle="-", marker="o", label=model)
        ax[1].fill_between(
            segments,
            itr_mean - itr_lower,
            itr_mean + itr_upper,
            alpha=0.2,
            label="_" + model
        )
    if version == "full":
        ax[0].axvspan(0, 4.2, color="grey", alpha=0.15, zorder=0)
        ax[1].axvspan(0, 4.2, color="grey", alpha=0.15, zorder=0)
    ax[0].axhline(1 / 20, color="k", linestyle="--", alpha=0.5, label="chance")
    ax[0].set_ylim(-0.02, 1.03)
    ax[1].set_ylim(-2, 103)
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


def display_multi_subject_learning_curve(curve_results, version, models, uncertainty="std"):
    fig, ax = plt.subplots(figsize=(15, 3))

    trialtime = 4.2 if version == "short" else 31.5
    
    for model in models:
        train_trials = curve_results[version][model]["train_trials"]

        acc_mean = curve_results[version][model]["accuracy"]["mean"]
        if uncertainty == "std":
            acc_lower = curve_results[version][model]["accuracy"]["std_lower"]
            acc_upper = curve_results[version][model]["accuracy"]["std_upper"]
        elif uncertainty == "se":
            acc_lower = curve_results[version][model]["accuracy"]["se_lower"]
            acc_upper = curve_results[version][model]["accuracy"]["se_upper"]

        ax.plot(train_trials * trialtime, acc_mean, linestyle="-", marker="o", label=model)
        ax.fill_between(
            train_trials * trialtime,
            acc_mean - acc_lower,
            acc_mean + acc_upper,
            alpha=0.2,
            label="_" + model
        )
    if version == "full":
        upper_bound = 4.2 * train_trials
        ax.axvspan(0, upper_bound[-1], color="grey", alpha=0.15, zorder=0)
    ax.axhline(1 / 20, color="k", linestyle="--", alpha=0.5, label="chance")
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("learning time [s]")
    ax.set_ylabel("accuracy")
    ax.legend()
    if version == "short":
        ax.set_title(f"Learning curve across subjects (trial length = 4.2s)")
    else:
        ax.set_title(f"Learning curve across subjects (trial length = 31.5s)")
    ax.legend(title="Version", loc="lower right")

    plt.tight_layout()
    plt.show()


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
