## get_data_dataloader.py

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
	dev_batch_size_post = int(jsonObject['dev_batch_size_post'])
	test_batch_size = int(jsonObject['test_batch_size'])

	# Define the validation/test pipeline (No augmentations, only normalization)
	test_transforms = transforms.Compose([
	
	    transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
	])

	train_dataset = torchvision.datasets.SVHN(root=datasets_folder, split = 'train', download=True, transform=None)
	extra_dataset = torchvision.datasets.SVHN(root=datasets_folder, split='extra', download=True, transform=None)
	
	traindev_dataset = ConcatDataset([train_dataset, extra_dataset])
	

	dev_sampler_file = save_folder + 'seed' + str(seed) + '/dev_sampler_idx.pt'
	dev_sampler = torch.load(dev_sampler_file)	

	dev_subset = TransformedSubset(traindev_dataset, dev_sampler, transform=test_transforms)
	
	test_dataset = torchvision.datasets.SVHN(root=datasets_folder, split = 'test', download=True, transform=test_transforms)

	dev_loader_post = DataLoader(dev_subset, batch_size=dev_batch_size_post, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=test_batch_size,shuffle=False)

	return dev_loader_post,test_loader



