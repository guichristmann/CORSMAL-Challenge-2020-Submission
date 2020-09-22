import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

class SimpleConvNet(nn.Module):
    ''' Conv model used for Task 1 - Filling type Classification '''

    def __init__(self):
        super(SimpleConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=(120, 40),
                               stride=4)
        
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=8,
                               kernel_size=(40, 1),
                               stride=2)
        
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)
        
        self.linear1 = nn.Linear(8*154, 4)
        
    def forward(self, x):
        o1 = F.relu(self.conv1(x))
        o1 = self.dropout1(o1)

        o2 = F.relu(self.conv2(o1))
        o2 = self.dropout2(o2)
        
        out = self.linear1(o2.view(-1, 8*154))
        
        return out

class BiggishConvNet(nn.Module):
    ''' Conv model used for Task 3 - Container Capacity estimation '''
    def __init__(self):
        super(BiggishConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)
        
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)
        
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)
        
        self.bn4 = nn.BatchNorm2d(128)
        
        self.linear1 = nn.Linear(7*7*128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.linear2 = nn.Linear(64, 6)
        self.linear3 = nn.Linear(8, 1) # 6 + 2 roi_info features
        
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, img, roi_info):
        o1 = F.relu(self.conv1(img))
        o1 = self.pool(self.bn1(o1))
        #print(o1.shape)
        
        o2 = F.relu(self.conv2(o1))
        o2 = self.pool(self.bn2(o2))
        #print(o2.shape)
        
        o3 = F.relu(self.conv3(o2))
        o3 = self.pool(self.bn3(o3))
        #print(o3.shape)
        
        o4 = F.relu(self.conv4(o3))
        o4 = self.pool(self.bn4(o4))
        #print(o4.shape)

        # Keep batch dim and flatten
        conv_out = o4.view(-1, 7*7*128)
        
        l1 = F.relu(self.linear1(conv_out))
        l1 = self.bn5(l1)
        
        l2 = F.relu(self.linear2(l1))
        
        # Concat roi_info along with the processed conv features
        concat = torch.cat((l2, roi_info), 1)
        #print(l1[0])

        # Pay attention to the activation here
        out = self.linear3(concat)
        
        return out
