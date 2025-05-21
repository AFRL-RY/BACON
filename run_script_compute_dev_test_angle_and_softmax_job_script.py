# run_script_compute_dev_test_angle_and_softmax_job_script.py

from get_data_dataloader import dataloaders
from get_angles import get_angles
from get_softmax import get_softmax
from get_scaled_softmax import get_Tscaled_softmax
from models import ResNet2
from models import Effnet

import json
import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import sys
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

    seed_idx=int(sys.argv[1])
    seed = list_of_seeds[seed_idx]
    print("seed = " , seed, "\n")

    # Loop through seeds

    j = seed

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

    #Get the model
    model = ResNet2(model_type,class_count,pretrained,freeze_net).to(device)
    #model = Effnet(class_count).to(device)
    model_path = savedir + "best_model.pth"
    model.load_state_dict(torch.load(model_path))
    print(model_type)
    model_file = savedir+file_model

    dev_file_name = savedir+'dev_angles.csv'
    dev_angles = get_angles(model, dev_loader_post, dev_batch_size_post,hyperparameter_file,print_accuracy=True, device=device,
                angle_file_name=dev_file_name)

    test_file_name = savedir+'test_angles.csv'
    test_angles = get_angles(model,test_loader,test_batch_size,hyperparameter_file,print_accuracy=True,device=device,angle_file_name=test_file_name)

    softmax = get_softmax(model,test_loader,hyperparameter_file,device,savedir)

