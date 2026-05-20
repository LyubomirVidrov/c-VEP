import os
import numpy as np
import utils

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset
# subject = "sub-01"

subjects = [f"sub-{1+i:02d}" for i in range(10)] 

# versions = ["120_300", "120_500", "240_300", "240_500"]
# models = ["eegnet_full_8_2", "eegnet_short_8_2", "rcca", "rcca_short"]

# def get_decocing_curve_data(file):
#     fn = os.path.join(path, "decoding_curve", "offline", file, subject, f"{subject}_gdf.npz")
#     tmp = np.load(fn)

#     itr = tmp["itr"]
#     acc_trial = tmp["acc_trial"]
#     segments = tmp["segments"]
#     name = file

#     return itr, acc_trial, segments, name


def get_learning_curve_data(file):
    fn = os.path.join(path, "learning_curve", "offline", "short", file, subject, f"{subject}_gdf.npz")
    tmp = np.load(fn)

    accuracy = tmp["accuracy"]
    train_trials = tmp["train_trials"]
    name=file

    return accuracy, train_trials, name

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
    versions = ["short"]
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

                mean_val = vals.mean(axis=0)        # mean across subjects

                # lower_vals = np.where(vals <= mean_val, vals, np.nan)
                # upper_vals = np.where(vals >= mean_val, vals, np.nan)

                lower_dev = np.where(vals <= mean_val, mean_val - vals, np.nan)
                upper_dev = np.where(vals >= mean_val, vals - mean_val, np.nan)

                std_lower = np.nanmean(lower_dev, axis=0)
                std_upper = np.nanmean(upper_dev, axis=0)

                multi_subj_results[version][model][metric] = {
                    "mean": mean_val,
                    "std": vals.std(axis=0),
                    "std_lower": std_lower,
                    "std_upper": std_upper,
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

                mean_val = vals.mean(axis=0)        # mean across subjects

                lower_vals = np.where(vals <= mean_val, vals, np.nan)
                upper_vals = np.where(vals >= mean_val, vals, np.nan)

                multi_subj_results[version][model][metric] = {
                    "mean": mean_val,
                    "std": vals.std(axis=0),
                    "std_lower": np.nanstd(lower_vals, axis=0),
                    "std_upper": np.nanstd(upper_vals, axis=0),
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

                mean_val = vals.mean()

                lower_vals = np.where(vals <= mean_val, vals, np.nan)
                upper_vals = np.where(vals >= mean_val, vals, np.nan)

                multi_subj_results[version][model][metric] = {
                    "mean": mean_val,
                    "std_lower": np.std(lower_vals),
                    "std_upper": np.std(upper_vals),
                    "values": vals
                }

    return multi_subj_results


# for subject in subjects:
    
    # accuracy, train_trials, name = get_learning_curve_data("eegnet_4_2")
    # accuracy_1, _, name_1 = get_learning_curve_data("rcca")
    # accuracy_2, _, name_2 = get_learning_curve_data("eegnet_8_2")
    
    # utils.display_learning_curve([(accuracy, name), (accuracy_1, name_1), (accuracy_2, name_2)], train_trials)

results = collect_multi_subject_data(path, subjects)

utils.plot_grouped_accuracy(results, is_full=False)

# versions = ["240_500"]
# models = ["eegnet_short_4_2", "eegnet_short_8_2", "rcca_short"]

# print(results["120_300"]["eegnet_full_8_2"]["acc_epoch"])
# print(results["120_300"]["eegnet_full_8_2"]["acc_trial"])
# print(results["120_300"]["rcca"]["accuracy"])

# print(results["240_500"]["eegnet_short_8_2"]["acc_epoch"])
# print(results["240_500"]["eegnet_short_8_2"]["acc_trial"])
# print(results["240_500"]["rcca_short"]["accuracy"])

curve_results = collect_multi_subject_decoding_curves(path, subjects)

# print(curve_results["240_500"]["eegnet_short_8_2"]["itr"])
# print(curve_results["240_500"]["rcca_short"]["accuracy"])
# print(curve_results["240_500"]["rcca_short"]["segments"])

utils.display_decoding_curve_from_dict(
    curve_results,
    version="full",
    models=["eegnet_8_2", "eegnet_4_2", "rcca"]
)

learning_curve_results = collect_multi_subject_learning_curves(path, subjects)

# print(learning_curve_results["short"]["eegnet_4_2"]["accuracy"])
# print(learning_curve_results["short"]["rcca"]["train_trials"])

utils.collect_multi_subject_learning_curves(
    learning_curve_results,
    version="short",
    models=["eegnet_8_2", "eegnet_4_2", "rcca"]
)