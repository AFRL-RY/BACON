## models.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

#from typing import OrderedDict
import torchvision.models as models



class Effnet(nn.Module):
    def __init__(self,output_size, trained=True):
        super(Effnet, self).__init__()
        self.ptm = models.efficientnet_b0(pretrained=trained)
        self.fc = nn.Sequential(OrderedDict([
            ('drop',nn.Dropout(0.2,inplace=True)),
            ('Fc2',nn.Linear(1000,512)),
            ('silu',nn.SiLU(inplace=True)),
            ('Fc3',nn.Linear(512,output_size))]))    
        self.fc.apply(self.init_weights)
        

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()           
    
    def forward(self, x):
        x = self.ptm(x)
        x = self.fc(x)
        return x

class VGG16(nn.Module):
    def __init__(self, output_size, trained=True):
        super(VGG16, self).__init__()
        self.ptm = models.vgg16_bn(pretrained=trained)
        
        # 1. First freeze ALL parameters globally
        for param in self.ptm.parameters():
            param.requires_grad = False       
        
        # 2. Unfreeze the final convolutional block (features block 4)
        # This allows VGG to adapt its high-level filters to upscaled SVHN shapes
        for param in self.ptm.features[40:].parameters():
            param.requires_grad = True
        
        # 3. Define the new classifier block (automatically requires_grad=True)
        self.ptm.classifier = nn.Sequential(OrderedDict([
            ('drop', nn.Dropout(0.2, inplace=False)),
            ('Fc2', nn.Linear(25088, 512)), 
            ('relu', nn.ReLU(inplace=True)),
            ('Fc3', nn.Linear(512, output_size))
        ]))


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, x):
        x = self.ptm.features(x)
        x = self.ptm.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.ptm.classifier(x)
        return x


#
class ResNet(nn.Module):
    def __init__(self,model_type,num_classes,trained):
        super(ResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=trained,num_classes=100)
        self.fc = nn.Sequential(OrderedDict([
            ('relu',nn.ReLU()),
            ('fc2',nn.Linear(100,50)),
            ('drop',nn.Dropout()),
            ('relu2',nn.ReLU()),
            ('fc3',nn.Linear(50,num_classes))]))


    def forward(self,x):
        x = self.resnet(x)
        x = self.fc(x)
        return x


def buildResNetModel(model_type,numClasses,train,freeze_net):
    if model_type == "resnet18":
        model = models.resnet18(pretrained=train)
    elif model_type == "resnet34":
        model = models.resnet34(pretrained=train)
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=train)
    elif model_type == "resnet101":
        model = models.resnet101(pretrained=train)
    elif model_type == "resent152":
        model = models.resnet152(pretrained=train)
    else:
        raise ValueError(model_type,"does not match any registered model in func buildResNetModel!!!")
    for param in model.parameters():
        param.requires_grad = freeze_net
    model.conv1 = nn.Conv2d(3,64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False) 
    numFcInputs = model.fc.in_features
    return model


class ResNet2(nn.Module):
    def __init__(self,model_type,num_classes,train,freeze_net):
        super(ResNet2,self).__init__()
        self.net = buildResNetModel(model_type,num_classes,train,freeze_net)
        self.fc = nn.Sequential(OrderedDict([
            ('relu',nn.ReLU()),
            ('Fc2',nn.Linear(1000,512)),
            ('drop',nn.Dropout()),
            ('relu2',nn.ReLU()),
            ('Fc3',nn.Linear(512,num_classes))]))

    def forward(self,x):
        x = self.net(x)
        x = self.fc(x)
        return x 
