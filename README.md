# BACON

## Description
This is a repository of code for the Bayesian Confidence Estimation (BACON) algorithm for estimating confidences in predictions made by deep neural networks

# Workflow

Step 1: Set initial hyperparameters
edit hyperparametetrs.json

Step 2: Create random seeds
	python create_seeds.py

Step 3: Train networks and compute angles and sofmax (Note: either drive run_script_train.py
with a PBS script, or edit file to loop through random seeds
	python run_script_train_job_script.py
	python run_script_compute_dev_test_angle_and_softmax_job_script.py

Step 4:
Optimize value of "delta" for estimating BACON
	python run_script_self_optimize.py

Step 5:
edit hyperparameters.json -- edit "delta" value

Step 6: Estimate temperature (1/beta) for computing temperature scaled softmax
	python scale_temperatures.py

Step 7:
edit hyperparameters.json -- edit "beta" value

Step 8: Compute temperature scaled softmax values
	python run_get_scaled_softmax.py

Step 9: Sample test set to create an imbalanced class test set
	python run_create_imbalanced_test_set.py

Step 10: Compute bacon estimates and calibration error metrics
	python run_script_compute_bacon.py
	python run_get_metrics_bacon.py
