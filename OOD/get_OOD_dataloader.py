## get_OOD_dataloader.py

import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import json
import pandas as pd
from torch import utils


def dataloaders(device,seed):
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
    dev_batch_size_post = int(jsonObject['dev_batch_size_post'])
    test_batch_size = int(jsonObject['test_batch_size'])

    print("train_dev_split:",train_dev_split)
    print("train_batch_size:",train_batch_size)
    print("dev_batch_size_train:",dev_batch_size_train)
    print("dev_batch_size_post:",dev_batch_size_post)
    print("test_batch_size:",test_batch_size)

    #print("Dataloaders parameters successfully loaded")

    # Define CIFAR-10 equivalent labels present in CIFAR-100 to be excluded
    semantically_overlapping_classes = [
        # Vehicle duplicates/near-matches
        'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 'streetcar', 'tractor', 'tank', 'rocket',
        # Animal/Pet duplicates or severe near-matches
        'bear', 'leopard', 'lion', 'tiger', 'wolf', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
        # Small animals/birds/insects (conflicts with bird, cat, dog, frog)
        'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
        'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
        'crab', 'lobster', 'snail', 'spider', 'worm', 'beaver', 'dolphin', 'otter', 'seal', 'whale'
    ]

    transform = transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(size=224),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

    ood_dataset = torchvision.datasets.CIFAR100(root=datasets_folder, train=False, download=True, transform = transform)
    
    # 3. Filter indices based on class names
    cifar100_classes = ood_dataset.classes
    orthogonal_indices = [
        idx for idx, target in enumerate(ood_dataset.targets)
        if cifar100_classes[target] not in semantically_overlapping_classes
    ]

    # 4. Create the final clean OOD Dataset
    orthogonal_ood_cifar100 = Subset(ood_dataset, orthogonal_indices)

    ood_loader = torch.utils.data.DataLoader(orthogonal_ood_cifar100,batch_size=test_batch_size, shuffle=False)

    return ood_loader
