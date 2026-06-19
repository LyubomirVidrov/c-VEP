import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
import os


def _get_subject_learning_curve(path, version, model, subject):
    """Retrieve subject-level learning-curve metrics.

    Parameters
    ----------
    path : str
        Path to the directory containing the subject-level learning-curve results.
    version : str
        trial version to load.
    model : str
        Model name.
    subject : str
        Subject label (e.g., "sub-01").

    Returns
    -------
    dict
        Dictionary containing classification accuracy and the corresponding
        number of training trials for the specified subject.
    """

    fn = os.path.join(
        path, "learning_curve", "offline",
        version, model, subject, f"{subject}_gdf.npz"
    )

    # load subject learning curve results
    tmp = np.load(fn)

    return {
        "accuracy": tmp["accuracy"],
        "train_trials": tmp["train_trials"]
    }


def _get_subject_decoding_curve(path, version, model, subject):
    """Retrieve subject-level decoding curve metrics.
    
    Parameters
    ----------
    path: str
        Path to the directory containing the subject-level training results.
    version: str 
        trial version to load.
    model: str 
        Model name.
    subject: str
        Subject label (e.g., "sub-01").

    Returns
    -------
    dict
        Dictionary containing decoding-curve segments, trial accuracy,
        and ITR values for the specified subject.
    """

    fn = os.path.join(
        path, "decoding_curve", "offline",
        version, model, subject, f"{subject}_gdf.npz"
    )

    # load subject decoding curve results 
    tmp = np.load(fn)

    return {
        "segments": tmp["segments"],
        "accuracy": tmp["acc_trial"],   
        "itr": tmp["itr"]
    }


def _get_subject_accuracy(path, param, version, model, subject):
    """Retrieve subject-level performance metrics.
    
    Parameters
    ----------
    path: str
        Path to the directory containing the subject-level training results.
    param: str
        Parameter configuration to load .
    version: str 
        trial version to load.
    model: str 
        Model name.
    subject: str
        Subject label (e.g., "sub-01").

    Returns
    -------
    dict
        Dictionary containing mean performance metrics across all 5 folds for
        the specified subject. The returned metrics depend on the model.
    """

    fn = os.path.join(
        path, "train", "offline",
        param, version, model, subject, f"{subject}_gdf.npz"
    )

    # load subject-specific results
    tmp = np.load(fn)

    # compute mean performance metric
    if model == "rcca":
        accuracy = np.mean(tmp["accuracy"])  # mean over 5 folds

        return {"accuracy": accuracy,}
    else:
        acc_epoch = np.mean(tmp["acc_epoch"] )  # mean over 5 folds
        acc_trial = np.mean(tmp["acc_trial"])  # mean over 5 folds

        return {"acc_epoch": acc_epoch, "acc_trial": acc_trial,}
    

def _collect_subject_learning_curves(path, subjects, versions, models):
    """Store subject-level learning curve metrics across configurations.

    Parameters
    ----------
    path : str
        Path to the directory containing the subject-level decoding-curve results.
    subjects : list[str]
        List of subjects whose results will be collected.
    versions : list[str]
        List of trial versions.
    models : list[str]
        List of model names.

    Returns
    -------
    dict
        Nested dictionary containing subject-level learning curve metrics. 
    """

    results = {}

    # Loop training versions
    for version in versions:
        results[version] = {}

        # Loop models 
        for model in models:
            results[version][model] = {}

            # Collect subject-level results
            for subject in subjects:
                results[version][model][subject] = _get_subject_learning_curve(
                    path, version, model, subject
                )

    return results
    

def _collect_subject_decoding_curves(path, subjects, versions, models):
    """Store subject-level decoding curve metrics across configurations.

    Parameters
    ----------
    path : str
        Path to the directory containing the subject-level decoding-curve results.
    subjects : list[str]
        List of subjects whose results will be collected.
    versions : list[str]
        List of trial versions.
    models : list[str]
        List of model names.

    Returns
    -------
    dict
        Nested dictionary containing subject-level decoding curve metrics. 
    """

    results = {}

    # Loop training versions
    for version in versions:
        results[version] = {}

        # Loop models 
        for model in models:
            results[version][model] = {}

            # Collect subject-level results
            for subject in subjects:
                results[version][model][subject] = _get_subject_decoding_curve(
                    path, version, model, subject
                )

    return results


def _collect_subject_accuracies(path, subjects, params, versions, models):
    """Store multi-subject training results. 
    
    Parameters
    ----------
    path: str
        Path to the directory containing the subject-level training results.
    subjects: list
        List of subjects whose results will be included in the evaluation.
    params: list
        List of parameter configurations.
    versions: list 
        List of trial versions.
    models: list 
        List of model names.

    Returns
    -------
    dict
        Nested dictionary containing subject-level performance metrics. 
    """

    results = {}

    # Loop parameter configurations
    for param in params:
        results[param] = {}

        # Loop training versions
        for version in versions:
            results[param][version] = {}

            # Loop models 
            for model in models:
                results[param][version][model] = {}

                # Collect subject-level results
                for subject in subjects:
                    results[param][version][model][subject] = _get_subject_accuracy(
                        path, param, version, model, subject
                    )

    return results


def collect_multi_subject_learning_curves(path, subjects):
    """Process multi-subject learning curve results and compute aggregate performance metrics.
    
    Parameters
    ----------
    path: str
        Path to the directory containing the subject-level training results.
    subjects: list
        List of subjects whose results will be included in the evaluation.

    Returns
    -------
    dict
        Nested dictionary containing aggregated learning curve metrics, including mean classification accuracy, 
        assymetric standard deviation, assymetric standard error, and per-subject accuracy values. 
    """

    # files
    versions = ["short", "full"]
    models = ["rcca", "eegnet-8-2", "eegnet-4-2"]

    # Get data
    subj_results = _collect_subject_learning_curves(path, subjects, versions, models)
    multi_subj_results = {}

    # Loop training versions
    for version, version_data in subj_results.items():
        multi_subj_results[version] = {}

        # Loop models 
        for model, subject_data in version_data.items():
            multi_subj_results[version][model] = {}

            first_subject = list(subject_data.keys())[0]
            train_trials = subject_data[first_subject]["train_trials"]

            # Loop metric 
            for metric in ["accuracy"]:

                vals = np.array([
                    subj_data[metric].mean(axis=0)   # mean over 5 folds first
                    for subj_data in subject_data.values()
                ])

                # Compute across-subject mean
                mean_val = np.nanmean(vals, axis=0)

                # number of subjects contributing to each side
                lower_vals = np.where(vals <= mean_val, vals, np.nan)
                upper_vals = np.where(vals >= mean_val, vals, np.nan)

                # compute asymmetric SD
                std_lower = np.sqrt(np.nanmean((lower_vals - mean_val) ** 2, axis=0))
                std_upper = np.sqrt(np.nanmean((upper_vals - mean_val) ** 2, axis=0))

                # number of subjects contributing to each side
                n_lower = np.sum(~np.isnan(lower_vals), axis=0)
                n_upper = np.sum(~np.isnan(upper_vals), axis=0)

                # compute asymmetric SE
                se_lower = std_lower / np.sqrt(n_lower)
                se_upper = std_upper / np.sqrt(n_upper)

                # Store aggregated results
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
    """Process multi-subject decoding curve results and compute aggregate performance metrics.
    
    Parameters
    ----------
    path: str
        Path to the directory containing the subject-level training results.
    subjects: list
        List of subjects whose results will be included in the evaluation.

    Returns
    -------
    dict
        Nested dictionary containing aggregated decoding-curve metrics. For each 
        metric (accuracy and ITR), the dictionary includes the across-subject mean curve, 
        asymmetric standard deviation, asymmetric standard error, and per-subject values. 
        Segment information is also included.
    """

    # files
    versions = ["full"]
    models = ["rcca", "eegnet-8-2", "eegnet-4-2"]

    # Get data 
    subj_results = _collect_subject_decoding_curves(path, subjects, versions, models)
    multi_subj_results = {}
    
    # Loop training versions
    for version, version_data in subj_results.items():
        multi_subj_results[version] = {}

        # Loop models
        for model, subject_data in version_data.items():
            multi_subj_results[version][model] = {}

            first_subject = list(subject_data.keys())[0]
            segments = subject_data[first_subject]["segments"]

            # Loop metrics 
            for metric in ["accuracy", "itr"]:

                vals = np.array([
                    subj_data[metric].mean(axis=0)   # mean over 5 folds first
                    for subj_data in subject_data.values()
                ])

                # Compute across-subject mean
                mean_val = np.nanmean(vals, axis=0)

                # split values below and above the mean
                lower_vals = np.where(vals <= mean_val, vals, np.nan)
                upper_vals = np.where(vals >= mean_val, vals, np.nan)

                # compute asymmetric SD
                std_lower = np.sqrt(np.nanmean((lower_vals - mean_val) ** 2, axis=0))
                std_upper = np.sqrt(np.nanmean((upper_vals - mean_val) ** 2, axis=0))

                # number of subjects contributing to each side
                n_lower = np.sum(~np.isnan(lower_vals), axis=0)
                n_upper = np.sum(~np.isnan(upper_vals), axis=0)

                # compute asymmetric SE
                se_lower = std_lower / np.sqrt(n_lower)
                se_upper = std_upper / np.sqrt(n_upper)

                # Store aggregated results
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
    """Process multi-subject training results and compute aggregate performance metrics.
    
    Parameters
    ----------
    path: str
        Path to the directory containing the subject-level training results.
    subjects: list
        List of subjects whose results will be included in the evaluation.

    Returns
    -------
    dict
        Nested dictionary containing aggregated results, including mean classification accuracy, 
        assymetric standard deviation, assymetric standard error, and per-subject accuracy values. 
    """

    # files
    params = ["120_300", "120_500", "240_300", "240_500"]
    versions =["short", "full"]
    models = ["rcca", "eegnet-8-2"]

    # Get data
    subj_results = _collect_subject_accuracies(path, subjects, params, versions, models)

    multi_subj_results = {}

    # Loop data configurations
    for param, param_data in subj_results.items():
        multi_subj_results[param] = {}

        # Loop training versions
        for version, version_data in param_data.items():
            multi_subj_results[param][version] = {}

            # Loop models
            for model, subject_data in version_data.items():
                multi_subj_results[param][version][model] = {}

                # Get available metrics from the first subject
                first_subject = list(subject_data.keys())[0]
                metrics = subject_data[first_subject].keys()

                # Loop metrics
                for metric in metrics:
                    vals = np.array([
                        acc_data[metric]
                        for acc_data in subject_data.values()
                    ])

                    # Compute across-subject mean
                    mean_val = np.nanmean(vals, axis=0)

                    # split values below and above the mean
                    lower_vals = np.where(vals <= mean_val, vals, np.nan)
                    upper_vals = np.where(vals >= mean_val, vals, np.nan)

                    # Compute asymmetric SD
                    std_lower = np.sqrt(np.nanmean((lower_vals - mean_val) ** 2, axis=0))
                    std_upper = np.sqrt(np.nanmean((upper_vals - mean_val) ** 2, axis=0))

                    # number of subjects contributing to each side
                    n_lower = np.sum(~np.isnan(lower_vals), axis=0)
                    n_upper = np.sum(~np.isnan(upper_vals), axis=0)

                    # Compute asymmetric SE
                    se_lower = std_lower / np.sqrt(n_lower)
                    se_upper = std_upper / np.sqrt(n_upper)
                    
                    # Store aggregated results
                    multi_subj_results[param][version][model][metric] = {
                        "mean": mean_val,
                        "std_lower": std_lower,
                        "std_upper": std_upper,
                        "se_lower": se_lower,
                        "se_upper": se_upper,
                        "values": vals
                    }

    return multi_subj_results


def display_multi_subject_accuracy(results, types, version, models, uncertainty="std"):
    """Display a bar plot showing the average multi-subject classification accuracy achieved 
    by each model for all parameter configurations. 

    Parameters
    ----------
    results : dict
        Nested dictionary containing aggregated multi-subject performance metrics.
    types : list[str]
        List of parameter configurations.
    version : str
        trial version (i.e., "short" or "full")
    models : list[str]
        List of model names.
    uncertainty : str (default: "std")
        Type of uncertainty to display as error bars. Use "std" for asymmetric 
        standard deviation or "se" for asymmetric standard error.
    """

    x = np.arange(len(models))
    width = 0.2

    # Initialize figure
    _, ax = plt.subplots(figsize=(10, 6))

    for i, type in enumerate(types):

        means = []
        lower = []
        upper = []

        for model in models:

            if model == "rcca":
                metric_name = "accuracy"
            else:
                metric_name = "acc_trial"
            
            means.append(results[type][version][model][metric_name]["mean"])

            if uncertainty == "std":
                lower.append(results[type][version][model][metric_name]["std_lower"])
                upper.append(results[type][version][model][metric_name]["std_upper"]) 
            elif uncertainty == "se":
                lower.append(results[type][version][model][metric_name]["se_lower"])
                upper.append(results[type][version][model][metric_name]["se_upper"])

        means = np.array(means)
        lower_errors = np.array(lower)
        upper_errors = np.array(upper)

        # Plot 
        ax.bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=[lower_errors, upper_errors],
            capsize=5,
            label=type
        )

    # Format axes
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(["eegnet", "rcca"], fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.set_xlabel("Model", fontsize=20)

    if version == "full":
        ax.set_title(f"Mean accuracy (trial length = 31.5s)", fontsize=24)
    else:
        ax.set_title(f"Mean accuracy (trial length = 4.2s)", fontsize=24)

    ax.legend(loc="lower left", fontsize=16)

    plt.tight_layout()
    plt.show()


def display_multi_subject_decoding_curve(curve_results, models, uncertainty="std"):
    """Display decoding curves showing the average multi-subject classification accuracy achieved 
    by each model. The upper plot represents a zoomed-in view of the shaded region in the bottom plot.

    Parameters
    ----------
    curve_results : dict
        Nested dictionary containing aggregated multi-subject performance metrics.
    models : list[str]
        List of model names.
    uncertainty : str (default: "std")
        Type of uncertainty to display as error bars. Use "std" for asymmetric 
        standard deviation or "se" for asymmetric standard error.
    """

    # Initialize figure
    fig, ax = plt.subplots(2, 1, figsize=(15, 5), sharex=False)

    for model in models:
        segments = curve_results["full"][model]["segments"]
        acc_mean = curve_results["full"][model]["accuracy"]["mean"]

        if uncertainty == "std":
            acc_lower = curve_results["full"][model]["accuracy"]["std_lower"]
            acc_upper = curve_results["full"][model]["accuracy"]["std_upper"]

        elif uncertainty == "se":
            acc_lower = curve_results["full"][model]["accuracy"]["se_lower"]
            acc_upper = curve_results["full"][model]["accuracy"]["se_upper"]

        # Plot (short trial)
        if model == "rcca":
            ax[0].plot(segments[:42], acc_mean[:42], linestyle="-", marker="o", label=model)
            ax[0].fill_between(
                segments[:42],
                acc_mean[:42] - acc_lower[:42],
                acc_mean[:42] + acc_upper[:42],
                alpha=0.2,
                label="_" + model
            )
        else:
            ax[0].plot(segments[:38], acc_mean[:38], linestyle="-", marker="o", label=model)
            ax[0].fill_between(
                segments[:38],
                acc_mean[:38] - acc_lower[:38],
                acc_mean[:38] + acc_upper[:38],
                alpha=0.2,
                label="_" + model
            )

        # Plot (full trial)
        ax[1].plot(segments, acc_mean, linestyle="-", marker="o", label=model)
        ax[1].fill_between(
            segments,
            acc_mean - acc_lower,
            acc_mean + acc_upper,
            alpha=0.2,
            label="_" + model
        )

    # Shaded region 
    ax[1].axvspan(0, 4.2, color="grey", alpha=0.15, zorder=0)

    # Chance level 
    ax[0].axhline(1 / 20, color="k", linestyle="--", alpha=0.5, label="chance")

    # Format axes
    ax[0].set_ylim(-0.02, 1.03)
    ax[1].axhline(1 / 20, color="k", linestyle="--", alpha=0.5, label="chance")
    ax[1].set_ylim(-0.02, 1.03)
    ax[1].set_xlabel("decoding time [s]", fontsize=16)
    ax[0].set_ylabel("accuracy", fontsize=16)
    ax[1].set_ylabel("accuracy", fontsize=16)

    ax[0].set_title(f"Decoding curve", fontsize=20) 
    
    ax[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=20)

    fig.tight_layout()
    plt.show()


def display_multi_subject_itr(curve_results, models, uncertainty="std"):
    """Display ITR curves. The upper plot represents a zoomed-in view of the shaded region in the bottom plot.

    Parameters
    ----------
    curve_results : dict
        Nested dictionary containing aggregated multi-subject performance metrics.
    models : list[str]
        List of model names.
    uncertainty : str (default: "std")
        Type of uncertainty to display as error bars. Use "std" for asymmetric 
        standard deviation or "se" for asymmetric standard error.
    """

    # Initialize figure
    fig, ax = plt.subplots(2, 1, figsize=(15, 5), sharex=False)

    for model in models:
        segments = curve_results["full"][model]["segments"]

        itr_mean = curve_results["full"][model]["itr"]["mean"]

        if uncertainty == "std":
            itr_lower = curve_results["full"][model]["itr"]["std_lower"]
            itr_upper = curve_results["full"][model]["itr"]["std_upper"]
        elif uncertainty == "se":
            itr_lower = curve_results["full"][model]["itr"]["se_lower"]
            itr_upper = curve_results["full"][model]["itr"]["se_upper"]

        # plot (short trial) 
        if model == "rcca":
            ax[0].plot(segments[:42], itr_mean[:42], linestyle="-", marker="o", label=model)
            ax[0].fill_between(
                segments[:42],
                itr_mean[:42] - itr_lower[:42],
                itr_mean[:42] + itr_upper[:42],
                alpha=0.2,
                label="_" + model
            )
        else:
            ax[0].plot(segments[:38], itr_mean[:38], linestyle="-", marker="o", label=model)
            ax[0].fill_between(
                segments[:38],
                itr_mean[:38] - itr_lower[:38],
                itr_mean[:38] + itr_upper[:38],
                alpha=0.2,
                label="_" + model
            )

        # Plot (full trial) 
        ax[1].plot(segments, itr_mean, linestyle="-", marker="o", label=model)
        ax[1].fill_between(
            segments,
            itr_mean - itr_lower,
            itr_mean + itr_upper,
            alpha=0.2,
            label="_" + model
        )

    # Shaded region 
    ax[1].axvspan(0, 4.2, color="grey", alpha=0.15, zorder=0)

    # Format axes
    ax[0].set_ylim(-2, 103)
    ax[1].set_ylim(-2, 103)
    ax[0].set_ylabel("ITR [bits/min]", fontsize=16)
    ax[1].set_xlabel("decoding time [s]", fontsize=16)
    ax[1].set_ylabel("ITR [bits/min]", fontsize=16)

    ax[0].set_title(f"ITR curve", fontsize=20)

    ax[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=20)

    fig.tight_layout()
    plt.show()


def display_multi_subject_learning_curve(curve_results, version, models, uncertainty="std"):
    """Display a learning curve showing the average classification accuracy across subjects for each model.

    Parameters
    ----------
    curve_results : dict
        Nested dictionary containing aggregated multi-subject performance metrics.
    version : str
        trial version (i.e., "short" or "full").
    models : list[str]
        List of model names.
    uncertainty : str (default: "std")
        Type of uncertainty to display as error bars. Use "std" for asymmetric 
        standard deviation or "se" for asymmetric standard error.
    """

    # Initialize figure
    _, ax = plt.subplots(figsize=(15, 3))

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

        # Plot 
        ax.plot(train_trials * trialtime, acc_mean, linestyle="-", marker="o", label=model)
        ax.fill_between(
            train_trials * trialtime,
            acc_mean - acc_lower,
            acc_mean + acc_upper,
            alpha=0.2,
            label="_" + model
        )

    # Chance level
    ax.axhline(1 / 20, color="k", linestyle="--", alpha=0.5, label="chance")

    # Format axes
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("learning time [s]", fontsize=16)
    ax.set_ylabel("accuracy", fontsize=16)
    
    if version == "short":
        ax.set_title(f"Learning curve (trial length = 4.2s)", fontsize=20)
    else:
        ax.set_title(f"Learning curve (trial length = 31.5s)", fontsize=20)

    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=20)

    plt.tight_layout()
    plt.show()


def calc_statistical_significance(curve_results, version, models, index):
    """Calculate statistical significance between two models at a given time point.

    Parameters
    ----------
    curve_results : dict
        Nested dictionary containing aggregated multi-subject performance metrics.
    version : str
        trial version (i.e., "short" or "full").
    models : list[str]
        List of model names.
    index : int
        Index of the time point to compare. 
    Returns
    -------
    p_value : float
        p-value from the Wilcoxon signed-rank test.
    mean_diff : float
        Mean paired difference between the two models.
    std_diff : float
        Standard deviation of the paired differences.
    hedges_g : float
        Hedges' g effect size between the two models.
    """

    if "itr" in curve_results[version][models[0]]:
        if "rcca" in models:
            set1 = curve_results[version][models[0]]["accuracy"]["values"][:, index - 4]
            set2 = curve_results[version][models[1]]["accuracy"]["values"][:, index]
        else:
            set1 = curve_results[version][models[0]]["accuracy"]["values"][:, index - 4]
            set2 = curve_results[version][models[1]]["accuracy"]["values"][:, index - 4]
    else:
        set1 = curve_results[version][models[0]]["accuracy"]["values"][:, index]
        set2 = curve_results[version][models[1]]["accuracy"]["values"][:, index]

    # mean paired difference
    mean_diff = np.mean(set1 - set2)

    # standard deviation
    std_diff = np.std(set1 - set2)
    
    # Perform Wilcoxon signed-rank test
    _, p_value = wilcoxon(set1, set2)

    # Hedges' g
    n1 = len(set1)
    n2 = len(set2)
    s1 = np.std(set1)
    s2 = np.std(set2)

    pooled_sd = np.sqrt(
    ((n1 - 1) * s1**2 + (n2 - 1) * s2**2)
    / (n1 + n2 - 2)
    )

    d = (np.mean(set1) - np.mean(set2)) / pooled_sd

    df = n1 + n2 - 2
    J = 1 - (3 / (4 * df - 1))

    hedges_g = d * J

    return p_value, mean_diff, std_diff, hedges_g
