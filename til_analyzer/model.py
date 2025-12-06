import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class DifferentResNet186(nn.Module): #256
    def __init__(self, num_intervals, intermediate_features = 256, second_intermediate_features=256, dropout_prob = 0.5):
        super(DifferentResNet186, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.resnet18.fc = Identity()
        
        self.fc1 = nn.Linear(512, intermediate_features)
        self.sigmoid1 = nn.Sigmoid() 
        self.batch_norm1 = nn.BatchNorm1d(intermediate_features)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.fc2 = nn.Linear(intermediate_features, second_intermediate_features)
        self.sigmoid2 = nn.Sigmoid() 
        self.batch_norm2 = nn.BatchNorm1d(second_intermediate_features)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.survival = nn.Linear(second_intermediate_features, num_intervals)
        self.sigmoid3 = nn.Sigmoid() 

        
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        # Apply dropout before the FC layer
        
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = self.survival(x)
        x = self.sigmoid3(x)

        return x

class GoodUmapandGraph(nn.Module):
    def __init__(self, num_intervals, intermediate_features = 512, dropout_prob = 0.5):
        super(GoodUmapandGraph, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)

        self.resnet18.fc = Identity()
        
        self.fc = nn.Linear(512, intermediate_features)
        self.sigmoid1 = nn.Sigmoid() 
        self.batch_norm = nn.BatchNorm1d(intermediate_features)
        self.dropout = nn.Dropout(p=dropout_prob)
        

        self.survival = nn.Linear(intermediate_features, num_intervals)
        self.sigmoid2 = nn.Sigmoid() 

        
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        # Apply dropout before the FC layer
        
        x = self.fc(x)
        x = self.sigmoid1(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.survival(x)
        x = self.sigmoid2(x)

        return x
    

