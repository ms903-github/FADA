import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):         #encoder:g
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.fc1 = nn.Linear(16, 120)     #再現実装だと２５６ｘ１２０だが１６ｘ１２０じゃないと動かない
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 64)
        #self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   #16 -> 12, 1->6
        x = F.max_pool2d(x, 2)      #12 -> 6
        x = F.relu(self.conv2(x))   #6 -> 2,  6->16
        x = F.max_pool2d(x, 2)      #2 -> 1
        x = x.view(x.size(0), -1)      
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))

        return(x)
class Encoder_2(nn.Module):
    def __init__(self):
        super(Encoder_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) 
        self.conv2 = nn.Conv2d(32, 64, 3) 
        self.batchnorm1 = nn.BatchNorm2d(32)
        #self.batchnorm2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2) 
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(6 * 6 * 64, 256)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(256, 64)
        #self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))                # 16x16x1 -> 14x14x32
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.conv2(x)))     #14x14x32 -> 12x12x64 -> 6x6x64
        #x = self.dropout1(x)
        #print(x.shape)
        x = x.view(-1, 6 * 6 * 64)             #6x6x64 -> 2304
        #print(x.shape)
        x = F.relu(self.fc1(x))                  #2304 -> 256
        #x = self.batchnorm2(x)
        #x = self.dropout2(x)
        x = self.fc2(x)                          #256 -> 64
        #x = self.fc3(x)
        return x





class classifier(nn.Module):     #classifier:h
    def __init__(self):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc1(x)

class DCD(nn.Module):
    def __init__(self):
        super(DCD, self).__init__()
        self.fc1 = nn.Linear(128, 64)        #64が２つで128
        #self.fc2 = nn.Linear(64, 64)         #この層がない方がinitial trainの精度が上がる
        self.fc3 = nn.Linear(64, 4)
        #self.dropout = nn.Dropout()
        self.batchnorm = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.batchnorm(x)
        return self.fc3(x)
