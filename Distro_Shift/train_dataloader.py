## train_dataloader.py

from torch.utils.data import Dataset, Subset, SubsetRandomSampler, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torchvision
import json
import torch
import pandas as pd
import numpy as np
from torch import utils

class TransformedSubset(Subset):
	def __init__(self, dataset, indices, transform=None):
		super().__init__(dataset, indices)
		self.transform = transform

	def __getitem__(self, idx):
		
		img, label = self.dataset[self.indices[idx]]
		
		if self.transform:
			img = self.transform(img)
			
		return img, label


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
	dev_size = int(jsonObject['dev_size'])
	dev_batch_size_train = int(jsonObject['dev_batch_size_train'])

	print("train_dev_split:",train_dev_split)
	print("train_batch_size:",train_batch_size)
	print("dev_batch_size_train:",dev_batch_size_train)
	
	train_transforms = transforms.Compose([
		
		transforms.RandomCrop(size=(32, 32), padding=4, padding_mode='reflect'),
		
		
		transforms.RandomAffine(
			degrees=(-15, 15),
			shear=(-10, 10),
			resample=False 
		),

		transforms.ColorJitter(brightness=0.2, contrast=0.2),
		
		transforms.Resize((224, 224)),
		
		transforms.ToTensor(),
		
		transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
		
		transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
	])


	test_transforms = transforms.Compose([
	
	    transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
	])

	train_dataset = torchvision.datasets.SVHN(root=datasets_folder, split = 'train', download=True, transform=None)
	extra_dataset = torchvision.datasets.SVHN(root=datasets_folder, split='extra', download=True, transform=None)
	
	traindev_dataset = ConcatDataset([train_dataset, extra_dataset])
	

	targets = np.concatenate([train_dataset.labels, extra_dataset.labels])

	class_indices = {}
	for idx, target in enumerate(targets):
		if target not in class_indices:
			class_indices[target] = []
		class_indices[target].append(idx)

	train_indices = []
	dev_indices = []
	
	np.random.seed(seed)
	for class_label, indices in class_indices.items():
		np.random.shuffle(indices)
		split_point = dev_size
		dev_indices.extend(indices[:split_point])
		train_indices.extend(indices[split_point:])
		print('class_label = ', class_label, 'dev_indices_len = ', len(dev_indices), 'train_indices_len = ', len(train_indices), '\n')
		
	train_sampler = SubsetRandomSampler(train_indices)
	dev_sampler = SubsetRandomSampler(dev_indices)
	
	dev_sampler_file = save_folder + 'seed' + str(seed) + '/dev_sampler_idx.pt'
	torch.save(dev_indices, dev_sampler_file)

	train_subset = TransformedSubset(traindev_dataset, train_indices, transform=train_transforms)
	dev_subset = TransformedSubset(traindev_dataset, dev_indices, transform=test_transforms)
	
	test_dataset = torchvision.datasets.SVHN(root=datasets_folder, split = 'test', download=True, transform=test_transforms)

	len_traindev_datapoints = len(traindev_dataset)

	train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True)
	train_loader_2 = DataLoader(train_subset, batch_size=1, shuffle=True)
	dev_loader_train = DataLoader(dev_subset, batch_size=dev_batch_size_train, shuffle=False)

	return train_loader,train_loader_2,dev_loader_train
