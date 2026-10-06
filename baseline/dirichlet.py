## dirichlet.py

# This module computes probabilities using Dirichlet calibrated Softmax

# imports
import random
import numpy as np
import pandas as pd
import os

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dirichlet Class Declaration
class DirichletCalibrationODIR(nn.Module):
	def __init__(self, num_classes, mu=1.0, nu = 1.0):
		super(DirichletCalibrationODIR, self).__init__()
		# Initialize an affine layer
		self.linear = nn.Linear(num_classes, num_classes)
		
		# Initialize weights to identity matrix and biases to zero
		with torch.no_grad():
			self.linear.weight.copy_(torch.eye(num_classes))
			self.linear.bias.zero_()
			
		self.mu = mu # Regularization for off-diagonal weights
		self.nu = nu # Regularization for bias
		
	def forward(self, probs):
		eps = 1e-12 # avoid log(0) errors
		log_probs = torch.log(torch.clamp(probs, min=eps))
		logits = self.linear(log_probs)
		return F.softmax(logits, dim=-1)
		
	def odir_regularization(self):
		W = self.linear.weight
		b = self.linear.bias
		num_classes = W.size(0)
		
		off_diagonal_mask = 1.0 - torch.eye(num_classes, device=W.device)
		penalty_w = torch.sum((W*off_diagonal_mask)**2)
		
		penalty_b = torch.sum(b**2)
		
		return self.mu*penalty_w+self.nu*penalty_b
		
def train_calibrator(model, val_probs, val_labels, epochs = 100, lr=0.01):
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	for epoch in range(epochs):
		optimizer.zero_grad()
		calibrated_probs = model(val_probs.float())
		
		nll_loss = F.nll_loss(torch.log(calibrated_probs + 1e-12), val_labels)
		reg_loss = model.odir_regularization()
		
		total_loss = nll_loss + reg_loss
		total_loss.backward()
		optimizer.step()
		


# Dirichlet computation routine
if __name__ == "__main__":

	hyperparameter_file = 'hyperparameters.json'
	save_folder = './output/'
	seeds_file = save_folder + 'seeds.txt'
	file_list = [save_folder]
    
	#Ensure files have a home
	for file in file_list:
		if os.path.exists(file) == False:
			os.mkdir(file)

	with open(hyperparameter_file) as jsonFile:
		jsonObject = json.load(jsonFile)
		jsonFile.close()
		
	seed_holdout = int(jsonObject['seed_holdout'])
	class_count = int(jsonObject['class_count'])	

	# Get list of seeds
	list_of_seeds = []
	with open("./output/seeds.txt","r") as f:
		for line in f:
			list_of_seeds.append(int(line.strip()))
            
	print("Seed = ", list_of_seeds)
		
	# Set device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("device = ", device)

	# Train Dirichlet parameters
	
	# Specify softmax file from holdout seed
	seed_val = list_of_seeds[seed_holdout]
	uncalibrated_probability_file = save_folder + 'seed' + str(seed_val) + '/' + 'softmax.csv'
	
	# Open pandas dataframe
	beginning_column_number = int(2)
	print("beginning column number = ", beginning_column_number, '\n')
	end_column_number = int(beginning_column_number + class_count)
	print("end column number = ", end_column_number, '\n')
	
	# obtain range of probabilities
	df_uncalibrated_probabilities = pd.read_csv(uncalibrated_probability_file)
	#print(df_uncalibrated_probabilities.head())
	df_uncalibrated_probabilities_probs = df_uncalibrated_probabilities.iloc[:, beginning_column_number:end_column_number].values
	
	print("uncal probs shape = ", df_uncalibrated_probabilities_probs.shape, '\n')
	
	val_uncalibrated_probabilities_tensor = torch.tensor(df_uncalibrated_probabilities_probs)
	val_labels_tensor = torch.from_numpy(df_uncalibrated_probabilities['Label'].values.astype(int))
	
	# Call train routine
	my_calibrator = DirichletCalibrationODIR(num_classes=class_count, mu=0.01, nu=0.01)
	train_calibrator(model=my_calibrator, val_probs=val_uncalibrated_probabilities_tensor, val_labels=val_labels_tensor)
		
	# Loop through seeds and compute Dirichlet probabilities
	for seed in list_of_seeds:

		if seed == list_of_seeds[seed_holdout]:
        
			continue
			
		print(seed,'\n')

		# Create Directory
		savedir = save_folder + 'seed' + str(seed) +'/'
		if savedir is not None:
			try:
				os.mkdir(savedir)
			except FileExistsError: n=1		
			
		dirichlet_probs_file = savedir + 'dirichlet_probs_file.csv'
		
		# Load in uncalibrated probabilities
		test_uncal_probability_file = savedir + 'softmax.csv'
	
		# Open pandas dataframe
		beginning_column_number = int(2)
		end_column_number = int(beginning_column_number + class_count)
	
		# obtain range of probabilities
		df_test_uncalibrated_probabilities = pd.read_csv(test_uncal_probability_file)
		test_uncalibrated_probabilities_tensor = torch.tensor(df_test_uncalibrated_probabilities.iloc[:,beginning_column_number:end_column_number].values)
		test_labels_tensor = torch.from_numpy(df_test_uncalibrated_probabilities['Label'].values.astype(int))
		test_predicted_tensor = torch.from_numpy(df_test_uncalibrated_probabilities['Predicted'].values.astype(int))		
	
		# Calibrate probabilities
		test_calibrated_probabilities = my_calibrator(test_uncalibrated_probabilities_tensor.float())
		
		# Construct output file
		full_array = torch.column_stack((test_labels_tensor, test_predicted_tensor, test_calibrated_probabilities)).detach().cpu().numpy()
		dirichlet_frame = pd.DataFrame(full_array,columns=['Label','Predicted']+['Dir'+str(i) for i in range(0,class_count)])
		
		# Save output file
		save_file = open(dirichlet_probs_file, 'w')
		print("Saving File", dirichlet_probs_file, '\n')
		
		dirichlet_frame.to_csv(save_file,index=False)
		save_file.close()
			
