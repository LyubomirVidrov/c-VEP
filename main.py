import os
import utils

home = os.path.expanduser("~")  # the path to the home folder
path = os.path.join(home, "data", "thielen2021")  # the path to the dataset

# List of subjects to analyze
subjects = [f"sub-{1+i:02d}" for i in range(10)] 

# Parameters search 
results = utils.collect_multi_subject_data(path, subjects)

# Display results 
utils.display_multi_subject_accuracy(
    results,
    types=["120_300", "120_500", "240_300", "240_500"],
    version="full",
    models=["eegnet-8-2", "rcca"],
    uncertainty="se"
)

# Get decoding curve data
curve_results = utils.collect_multi_subject_decoding_curves(path, subjects)

# Display decoding curve 
utils.display_multi_subject_decoding_curve(
    curve_results,
    models=["eegnet-8-2", "eegnet-4-2", "rcca"],
    uncertainty="se"
)

# Display ITR curve
utils.display_multi_subject_itr(
    curve_results,
    models=["eegnet-8-2", "eegnet-4-2", "rcca"],
    uncertainty="se"
)

# Get learning curve data 
learning_curve_results = utils.collect_multi_subject_learning_curves(path, subjects)

# Display learning curve
utils.display_multi_subject_learning_curve(
    learning_curve_results,
    version="full",
    models=["eegnet-8-2", "eegnet-4-2", "rcca"],
    uncertainty="se"
)

# Statistical analysis
print("Learning curve stat:")
p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(learning_curve_results, version="full", models = ["eegnet-8-2", "rcca"], index=0) # index != number of trials 
print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")
p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(learning_curve_results, version="full", models = ["eegnet-8-2", "rcca"], index=14)
print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")
p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(learning_curve_results, version="full", models = ["eegnet-8-2", "rcca"], index=26)
print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")

print("Decoding curve stat:")
p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(curve_results, version="full", models = ["eegnet-8-2", "eegnet-4-2"], index=42)
print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")
p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(curve_results, version="full", models = ["eegnet-8-2", "eegnet-4-2"], index=51)
print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")
p_value, mean_diff, std_diff, hedges_g = utils.calc_statistical_significance(curve_results, version="full", models = ["eegnet-8-2", "eegnet-4-2"], index=79)
print(f"p-value: {p_value:.3f}, mean difference: {(mean_diff*100):.1f}%, std difference: {std_diff:.3f}, Hedges' g: {hedges_g:.3f}")
