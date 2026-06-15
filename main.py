import os
import numpy as np
import utils

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset

subjects = [f"sub-{1+i:02d}" for i in range(10)] 

# Parameters search 
results = utils.collect_multi_subject_data(path, subjects)

utils.plot_grouped_accuracy(results, is_full=False, uncertainty="se")

# Decoding curve
curve_results = utils.collect_multi_subject_decoding_curves(path, subjects)

utils.display_multi_subject_decoding_curve(
    curve_results,
    # version="full",
    models=["eegnet-8-2", "eegnet-4-2", "rcca"],
    uncertainty="se"
)

utils.display_multi_subject_itr(
    curve_results,
    # version="full",
    models=["eegnet-8-2", "eegnet-4-2", "rcca"],
    uncertainty="se"
)

# Learning curve
learning_curve_results = utils.collect_multi_subject_learning_curves(path, subjects)

utils.display_multi_subject_learning_curve(
    learning_curve_results,
    version="short",
    models=["eegnet-8-2", "eegnet-4-2", "rcca"],
    uncertainty="se"
)

# Statistical analysis
# p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(learning_curve_results, version="full", models = ["eegnet_8_2", "rcca"], index=0)
# print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")
# p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(learning_curve_results, version="full", models = ["eegnet_8_2", "rcca"], index=14)
# print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")
# p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(learning_curve_results, version="full", models = ["eegnet_8_2", "rcca"], index=26)
# print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")

# p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(curve_results, version="full", models = ["eegnet_8_2", "eegnet_4_2"], index=42)
# print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")
# p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(curve_results, version="full", models = ["eegnet_8_2", "eegnet_4_2"], index=51)
# print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")
# p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(curve_results, version="full", models = ["eegnet_8_2", "eegnet_4_2"], index=79)
# print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")