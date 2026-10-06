## train_dataloader.py

from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torchvision.transforms as transforms
import torchvision
import json
import torch
import pandas as pd
import numpy as np
from torch import utils


def dataloaders(device, seed):
	hyperparameter_file = 'hyperparameters.json'
	datasets_folder = './datasets'
	save_folder = './output/'
	file_list = [datasets_folder,save_folder]
	with open(hyperparameter_file) as jsonFile:
		jsonObject = json.load(jsonFile)
		jsonFile.close()

	print("Dataloaders parameters loading")


	train_dev_split = float(jsonObject['train_dev_split'])
	train_batch_size = int(jsonObject['train_batch_size'])
	dev_batch_size_train = int(jsonObject['dev_batch_size_train'])

	print("train_dev_split:",train_dev_split)
	print("train_batch_size:",train_batch_size)
	print("dev_batch_size_train:",dev_batch_size_train)

	transform = transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(size=224),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

	traindev_dataset = torchvision.datasets.CIFAR10(root=datasets_folder, train=True, download=True, transform=transform)

	test_dataset = torchvision.datasets.CIFAR10(root=datasets_folder, train=False, download=True, transform=transform)

	len_traindev_datapoints = len(traindev_dataset)

	train_size = int(len_traindev_datapoints*train_dev_split)
	dev_size = len_traindev_datapoints-train_size
    
    
	####  Here do the stratified split
	targets = traindev_dataset.targets
	class_indices = {}
	for idx, target in enumerate(targets):
		if target not in class_indices:
			class_indices[target] = []
		class_indices[target].append(idx)

	train_ratio = train_dev_split
	dev_ratio  = 1 - train_ratio
	
	train_indices = []
	dev_indices = []
	
	np.random.seed(seed)
	for class_label, indices in class_indices.items():
		np.random.shuffle(indices)
		split_point = int(len(indices)*train_ratio)
		train_indices.extend(indices[:split_point])
		dev_indices.extend(indices[split_point:])
		
	train_sampler = SubsetRandomSampler(train_indices)
	dev_sampler = SubsetRandomSampler(dev_indices)
    
	dev_sampler_file = save_folder + 'seed' + str(seed) + '/dev_sampler_idx.pt'
	torch.save(dev_indices, dev_sampler_file)
    
	####


	train_loader = DataLoader(traindev_dataset, batch_size=train_batch_size,
		sampler=train_sampler)

	train_loader_2 = torch.utils.data.DataLoader(traindev_dataset, batch_size=1,
		sampler=train_sampler)

	dev_loader_train = DataLoader(traindev_dataset, batch_size=dev_batch_size_train, 
		sampler=dev_sampler)



	return train_loader,train_loader_2,dev_loader_train
