# run_script_train.py

# This module trains multiple neural networks, using random seeds to deterministically seed the random number generators used to do train/val splits.
# This module is driven by a PBS script that provides a seed to initialize random functions

import sys

from train_network import train_network
from train_dataloader import dataloaders
from models import ResNet2
from models import Effnet

import json
import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import shutil

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

    #Load hyperparameters from json file
    print("Script parameters loading")

    classes = jsonObject['classes']
    class_count = int(jsonObject['class_count'])
    model_type = jsonObject['model_type']
    freeze_net = bool(jsonObject['freeze_net'])
    pretrained = jsonObject['pretrained']
    file_model = jsonObject['file_model']
    target_acc = float(jsonObject['target_acc'])
    dev_batch_size_post = int(jsonObject['dev_batch_size_post'])
    test_batch_size = int(jsonObject['test_batch_size'])

    print("classes:",classes)
    print("class_count:",class_count)
    print("file_model:",file_model)
    print("target_acc:",target_acc)
    print("dev_batch_size_post:",dev_batch_size_post)
    print("test_batch_size:", test_batch_size)

    print("Script parameters successfully loaded")

    if len(classes) != class_count:
        raise ValueError("Length of Classes at",len(classes),"does not equal the class_count variable at",class_count,"!!!")

    #Copy hyperparameter_file to output folder
    hyperparameter_destination = './output/' + hyperparameter_file
    shutil.copyfile(hyperparameter_file, hyperparameter_destination)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('* * * * * * * * * * * * * * * *')
    print('Device Loaded')
    print('device = ', device)
    print('* * * * * * * * * * * * * * * *')

    with open(seeds_file) as f:
        list_of_seeds = [int(x) for x in f.read().split()]
        
    seed_idx = int(sys.argv[1])
    seed=list_of_seeds[seed_idx]
    print("seed = ", seed, '\n')

    # Loop through seeds
    #for j in list_of_seeds:
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
    # train_loader is used for actual training, train_loader_2 uses a different batch size and is used with dev_loader_train to evaluate training progress
    train_loader,train_loader_2,dev_loader_train = dataloaders(device)
    
    #Get the model
    
    # choose ResNet or EfficientNet-B0 by commenting out / uncommenting the following two lines as desired
    #model = ResNet2(model_type,class_count,pretrained,freeze_net).to(device)
    model = Effnet(class_count).to(device)
    
    print(model_type)
    model_file = savedir+file_model
    
    # Train the network
    model = train_network(model, train_loader, dev_loader_train, hyperparameter_file, target_acc, device=device, save_best=True, root=savedir,save_file=model_file )
    print('model trained')
