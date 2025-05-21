# get_angles.py

import os
import json
import torch
from torch import linalg as LA
import numpy as np
import pandas as pd
import torch.nn.functional as F

def get_angles(model, dataloader, batchsize, hyperparameter_file,print_accuracy=False, device='cpu', angle_file_name = None):
    
    angle_file = open(angle_file_name, 'w')
    
    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    class_count = int(jsonObject['class_count'])
    classes = jsonObject['classes']

    array_w = 2+class_count
    image_angles = np.empty(array_w)
    full_array = np.empty([len(dataloader)*batchsize,array_w])

    model.eval()    #Put Model in Evaluate
    #Set up activation dictionary and get activation function for forward hooks
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Register forward hook to be able to extract activations 
    # fc2 and fc3 are the last two fully connected layers of the DNN
    model.fc.Fc2.register_forward_hook(get_activation('Fc2'))  # used to compute phi
    output_layer = model.fc.Fc3.state_dict() # Get the last fully connected layer (Output Layer)

    wt = output_layer['weight']
    bias = output_layer['bias']
    wt_norm = LA.vector_norm(wt,dim=1) # Compute magnitude of output layer weights

    phi = np.zeros([class_count],dtype=float) # initialize vector for angle values

    # model.eval() set to disable batch normalilzation and dropout
    model.eval()
    #no_grad is selected so that NN weights will not change
    with torch.no_grad():
        # initialize/zeroize accumulators
        n_correct = 0
        n_samples = len(dataloader)
        n_class_correct = [0 for i in range(class_count)]
        n_class_samples = [0 for i in range(class_count)]
        
        for index,(images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device).view(-1,1)
            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            predicted = predicted.view(-1,1)
            if print_accuracy:
                n_class_samples[predicted] += 1
                if predicted == labels:
                    n_correct += 1
                    n_class_correct[labels] += 1
                
            act = F.relu(activation['Fc2']) # Get decision layer activations
            act_norm = LA.vector_norm(act) # Get activation layer magnitude

            outs = torch.tensordot(wt,act,dims=([1],[1])).T  
            phi = (torch.arccos(torch.divide(torch.divide(outs,wt_norm).T,act_norm).T)*180/np.pi)
            ind_start = index*batchsize
            full_array[ind_start:ind_start+len(images)] = torch.cat((labels,predicted,phi),dim=1).detach().cpu()
            
    if print_accuracy:
        print("--------Accuracy of Network---------")
        print("Network:",angle_file_name)

        #Compute accuracy of network
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
        

        #Compute accuracy for each class
        acc_list = []
        for i in range(class_count):
            if n_class_samples[i] != 0:
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            else:
                acc = 0
            acc_list.append(acc)
            print(f'Accuracy of {classes[i]}: {acc} %')

    print("")
    angle_frame = pd.DataFrame(full_array,columns=['Label','Predicted']+['Phi'+str(i) for i in range(0,class_count)])

    # Close angle output file
    if angle_file is not None:
        angle_frame.to_csv(angle_file)
        angle_file.close()
    else:
        raise ValueError("angle_file is not defined!")

    return angle_frame
