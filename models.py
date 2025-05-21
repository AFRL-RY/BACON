#models.py


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
    def __init__(self,num_classes):
        class_count = num_classes
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,32,3, padding=1)
        self.conv3 = nn.Conv2d(32,64,3, padding=1)
        self.conv4 = nn.Conv2d(64,64,3, padding = 1)
        self.conv5 = nn.Conv2d(64,128,3, padding = 1)
        self.conv6 = nn.Conv2d(128,128,3, padding = 1)
        self.conv7 = nn.Conv2d(128,128,3, padding = 1)
        self.conv8 = nn.Conv2d(128,256,3, padding = 1)
        self.conv9 = nn.Conv2d(256,256,3, padding = 1)
        self.conv10 = nn.Conv2d(256,256,3, padding = 1)
        self.conv11 = nn.Conv2d(256,256,3, padding = 1)
        self.conv12 = nn.Conv2d(256,256,3, padding = 1)
        self.conv13 = nn.Conv2d(256,256,3, padding = 1)
        self.fc1 = nn.Linear(2 * 2 * 256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, class_count)
        self.drop = nn.Dropout()
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()            

    def forward(self, x):
        # ->n, 3, 32, 32
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(F.relu(self.conv7(x)))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool(F.relu(self.conv10(x)))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x



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
