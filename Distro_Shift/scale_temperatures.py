## scale_temperatures.py

import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import os
import random
from get_scaled_softmax import get_Tscaled_softmax
from confidence_errors import get_confidence_errors
import matplotlib.pyplot as plt

from models import ResNet2
from models import Effnet
from models import VGG16

import get_softmax
from get_data_dataloader import dataloaders

if __name__ == "__main__":
    
	hyperparameter_file = 'hyperparameters.json'
	datasets_file = './datasets'
	save_folder = './output/'
	file_list = [datasets_file,save_folder]
    
	#Ensure files have a home
	for file in file_list:
		if os.path.exists(file) == False:
			os.mkdir(file)

            
	# Import JSON parameters            
	with open(hyperparameter_file) as jsonFile:
		jsonObject = json.load(jsonFile)
		jsonFile.close()
        
	seed_holdout = jsonObject['seed_holdout']
	model_type = jsonObject['model_type']
	class_count = int(jsonObject['class_count'])
	pretrained = bool(jsonObject['pretrained'])
	freeze_net = bool(jsonObject['freeze_net'])
    
	# Set device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("device = ", device)

	# set seed to holdout set seed
	list_of_seeds = []
	with open("./output/seeds.txt","r") as f:
		for line in f:
			list_of_seeds.append(int(line.strip()))

	seedhold = list_of_seeds[seed_holdout]            
	print("Seed = ", seedhold)
    
	opt_list = []
	opt_list.append(seedhold)
    
	seed = opt_list[0]
    
	print('opt_list = ', opt_list, seed)
    
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False
	torch.use_deterministic_algorithms(True)
    
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    
	# Set save directory
	savedir = save_folder + 'seed' + str(seed) +'/'
	if savedir is not None:
		try:
			os.mkdir(savedir)
		except FileExistsError: n=1

	# Import dataloaders
	dev_loader_post,test_loader = dataloaders(device,seed)

	# import model
	# model = ResNet2(model_type,class_count,pretrained,freeze_net).to(device)
	# model = Effnet(class_count).to(device)
	model = VGG16(class_count).to(device)
	model_file = savedir + "best_model.pth"
	model.load_state_dict(torch.load(model_file))    
    
	# Initialize beta value
	beta = np.arange(0.4,0.8,0.01)
	print('beta = ', beta)
    
	NLL_list = []    
    
	#softmax_output_file = savedir + 'Unscaled_softmax.csv'
	scaled_softmax_output_file = savedir + 'Tscaled_softmax.csv'    
    
	for beta_inst in beta:
		T_softmax_vals = get_Tscaled_softmax(model,dev_loader_post,hyperparameter_file, beta_inst, device='cuda', file_name=None)
        
		SM_stem = "SM"
    
		#SM_frame = pd.DataFrame(columns=["Accuracy","Confidence"])
		T_SM_list = []
    
		for j,item in enumerate(T_softmax_vals.iloc()):
			if item['Label'] == item['Predicted']:
				accur = 1
			else:
				accur = 0
			conf = item[SM_stem + str(int(item['Label']))]
			#SM_frame.loc[j] = [accur,conf]
			T_SM_list.append(np.log(conf))
            
		T_SM_arr = np.array(T_SM_list)
		NLL_beta = - np.sum(T_SM_arr)
		NLL_list.append(NLL_beta)
        
		print('beta = ', beta_inst, '\n')
		print('NLL (beta) = ', NLL_beta, '\n')
    
	NLL_arr = np.array(NLL_list)
	#print('NLL Array = ', NLL_arr)


	p = np.polyfit(beta,NLL_arr,2)

	best_beta = -p[1]/(2*p[0])
	print('best beta = ',best_beta)
	


