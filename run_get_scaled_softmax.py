#run_get_scaled_softmax.py

import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import os
import random
from get_softmax import get_softmax
from get_scaled_softmax import get_Tscaled_softmax
from confidence_errors import get_confidence_errors

from models import ResNet2
from models import Effnet

import get_softmax
from get_data_dataloaders import dataloaders

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
    class_count = int(jsonObject['class_count'])
    beta = float(jsonObject['beta'])
    model_type = jsonObject['model_type']
    pretrained = bool(['pretrained'])
    freeze_net = bool(['freeze_net'])

    print("seed_holdout:", seed_holdout)
    print("class_count:", class_count)
    print("beta:",beta)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    # Get list of seeds
    list_of_seeds = []
    with open("./output/seeds.txt","r") as f:
        for line in f:
            list_of_seeds.append(int(line.strip()))
            
    print("Seed = ", list_of_seeds)

    for j in list_of_seeds:

        if j == list_of_seeds[seed_holdout]:
        
            continue
        
        random.seed(j)
        np.random.seed(j)
        torch.manual_seed(j)
        torch.cuda.manual_seed(j)
        torch.cuda.manual_seed_all(j)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        # This is needed for CUDA to run deterministically
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        
        print(j,'\n')

        # Create Directory
        savedir = save_folder + 'seed' + str(j) +'/'
        if savedir is not None:
            try:
                os.mkdir(savedir)
            except FileExistsError: n=1

        #Get Dataloaders
        dev_loader_post,test_loader = dataloaders(device)

        # import model
        #model = ResNet2(model_type,class_count,pretrained,freeze_net).to(device)
        model = Effnet(class_count).to(device)
        model_file = savedir + "best_model.pth"
        model.load_state_dict(torch.load(model_file))

        T_SM_file = savedir + 'balanced_scaled_softmax.csv'

        scaled_softmax = get_Tscaled_softmax(model,test_loader,hyperparameter_file, beta, device='cuda', file_name=T_SM_file)
