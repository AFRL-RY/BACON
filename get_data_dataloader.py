# get_data_dataloader.py

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import json
import torch
import pandas as pd
from torch import utils
from models import ResNet2


def dataloaders(device):
    hyperparameter_file = 'hyperparameters.json'
    datasets_folder = './datasets'
    save_folder = './output_100/'
    file_list = [datasets_folder,save_folder]
    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    print("Dataloaders parameters loading")


    train_dev_split = float(jsonObject['train_dev_split'])
    train_batch_size = int(jsonObject['train_batch_size'])
    dev_batch_size_train = int(jsonObject['dev_batch_size_train'])
    dev_batch_size_post = int(jsonObject['dev_batch_size_post'])
    test_batch_size = int(jsonObject['test_batch_size'])


    print("train_dev_split:",train_dev_split)
    print("train_batch_size:",train_batch_size)
    print("dev_batch_size_train:",dev_batch_size_train)
    print("dev_batch_size_post:",dev_batch_size_post)
    print("test_batch_size:",test_batch_size)

    #print("Dataloaders parameters successfully loaded")


    transform = transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(size=224),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

    traindev_dataset = torchvision.datasets.CIFAR10(root=datasets_folder, train=True,
                                        download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root=datasets_folder, train=False,
                                        download=True, transform=transform)

    len_traindev_datapoints = len(traindev_dataset)

    train_size = int(len_traindev_datapoints*train_dev_split)
    dev_size = len_traindev_datapoints-train_size


    train_dataset, dev_dataset = torch.utils.data.random_split(traindev_dataset, [train_size, dev_size])


    dev_loader_post = torch.utils.data.DataLoader(dev_dataset, batch_size=dev_batch_size_post,
                                          shuffle=False)   

    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=test_batch_size,
                                            shuffle=False)

    return dev_loader_post,test_loader
