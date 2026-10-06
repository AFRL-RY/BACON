## run_script_compute_dev_test_angles_and_softmax.py

from get_data_dataloader import dataloaders
from get_angles import get_angles
from get_softmax import get_softmax
from models import ResNet2
from models import Effnet
from models import VGG16

import json
import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import random
import numpy as np
import os

if __name__ == "__main__":

    hyperparameter_file = 'hyperparameters.json'
    datasets_folder = './datasets'
    save_folder = './output/'
    seeds_file = save_folder + 'seeds.txt'
    file_list = [datasets_folder,save_folder]

    #Ensure files have a home
    for file in file_list:
        if os.path.exists(file) == False:
            os.mkdir(file)

    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    class_count = int(jsonObject['class_count'])
    model_type = jsonObject['model_type']
    file_model = jsonObject['file_model']
    dev_batch_size_post = int(jsonObject['dev_batch_size_post'])
    test_batch_size = int(jsonObject['test_batch_size'])
    pretrained = bool(jsonObject['pretrained'])
    freeze_net = bool(jsonObject['freeze_net'])

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('* * * * * * * * * * * * * * * *')
    print('Device Loaded')
    print('device = ', device)
    print('* * * * * * * * * * * * * * * *')


    with open(seeds_file) as f:
        list_of_seeds = [int(x) for x in f.read().split()]

    # Loop through seeds
    for seed in list_of_seeds:
    
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        # This is needed for CUDA to run deterministically
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        print(seed,'\n')

        # Create Directory
        savedir = save_folder + 'seed' + str(seed) +'/'
        if savedir is not None:
            try:
                os.mkdir(savedir)
            except FileExistsError: n=1

        #Get Dataloaders
        dev_loader_post,test_loader = dataloaders(device,seed)

        #Get the model
        # choose ResNet or EfficientNet-B0 by commenting out / uncommenting the following two lines as desired
        if model_type == 'resnet18':
        	model = ResNet2(model_type,class_count,pretrained,freeze_net).to(device)
        elif model_type == 'Effnet':
        	model = Effnet(class_count).to(device)
        elif model_type == 'VGG-16':
        	model = VGG16(class_count).to(device)
        else:
        	print("Model type not properly specified")

        model_path = savedir + "best_model.pth"
        model.load_state_dict(torch.load(model_path))
        print(model_type)
        model_file = savedir+file_model

        print("Starting dev_file angles", "\n")
        dev_file_name = savedir+'dev_angles.csv'
        dev_angles = get_angles(model, dev_loader_post, dev_batch_size_post,hyperparameter_file,print_accuracy=True, device=device,
                    angle_file_name=dev_file_name)

        print("Starting test_file angles", "\n")
        test_file_name = savedir+'test_angles.csv'
        test_angles = get_angles(model,test_loader,test_batch_size,hyperparameter_file,print_accuracy=True,device=device,angle_file_name=test_file_name)
        
        
        print("Starting dev softmax", "\n")
        dev_softmax = get_softmax(model, dev_loader_post,hyperparameter_file,device,savedir+'dev_softmax.csv')

        print("Starting softmax", "\n")
        softmax = get_softmax(model,test_loader,hyperparameter_file,device,savedir+'softmax.csv')

