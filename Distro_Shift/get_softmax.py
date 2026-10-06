## get_softmax.py

import torch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json


def get_softmax(model,dataloader,hyperparameter_file, device='cpu', file_name=None):
    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    class_count = int(jsonObject['class_count'])
    classes = jsonObject['classes']

    batchsize = dataloader.batch_size 
    array_w = 2+class_count
    full_array = np.empty([len(dataloader)*batchsize,array_w])
    
    model.eval()
    
    with torch.no_grad():        
        for index,(images, labels) in enumerate(dataloader):
            
            images = images.to(device)
            labels = labels.to(device).view(-1,1)
            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            predicted = predicted.view(-1,1)
            
            sftmx = F.softmax(outputs,dim=-1)

            ind_start = index*batchsize 
            full_array[ind_start:ind_start+len(images)] = torch.cat((labels,predicted,sftmx),dim=1).detach().cpu()

            
                                  
    softmax_frame = pd.DataFrame(full_array,columns=['Label','Predicted']+['SM'+str(i) for i in range(0,class_count)])

    # Close angle output file
    if file_name is not None:
        save_file = open(file_name, 'w')
    else:
        save_file = open(file_name+'softmax.csv', 'w')
    softmax_frame.to_csv(save_file,index=False)
    save_file.close()

    return softmax_frame
