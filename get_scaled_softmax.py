# get_scaled_softmax.py

import torch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json


def get_Tscaled_softmax(model,dataloader,hyperparameter_file, T_recip, device='cpu', file_name=None):
    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    class_count = int(jsonObject['class_count'])
    classes = jsonObject['classes']

    batchsize = dataloader.batch_size #Figure out which batchsize!!!
    array_w = 2+class_count
    full_array = np.empty([len(dataloader)*batchsize,array_w])
    
    model.eval()
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.fc.Fc3.register_forward_hook(get_activation('Fc3'))
    
    with torch.no_grad():        
        for index,(images, labels) in enumerate(dataloader):
            
            images = images.to(device)
            labels = labels.to(device).view(-1,1)
            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            predicted = predicted.view(-1,1)
            
            act_output = activation['Fc3'] # use this to compute softmax
            
            
            num = torch.exp(T_recip*act_output).T
            denom = torch.sum(torch.exp(T_recip*act_output),dim=1)
            sftmx = torch.divide(num,denom).T

            ind_start = index*batchsize 
            full_array[ind_start:ind_start+len(images)] = torch.cat((labels,predicted,sftmx),dim=1).detach().cpu()
                      
    softmax_frame = pd.DataFrame(full_array,columns=['Label','Predicted']+['SM'+str(i) for i in range(0,class_count)])

    # Close angle output file
    if file_name is not None:
        save_file = open(file_name, 'w')
    else:
        save_file = open('balanced_scaled_softmax.csv', 'w')
    softmax_frame.to_csv(save_file)
    save_file.close()

    return softmax_frame
