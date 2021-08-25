# kuzu.py


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.input = nn.Linear(28*28, 10) # linear function of the pixels in the image 

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = F.log_softmax(self.input(out), dim = 1)
        return out 

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.fc_layer1 = nn.Linear(28*28, 540) # input to hidden (num of hidden nodes chosen)
        self.fc_output_layer = nn.Linear(540, 10) # hidden to output       

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = torch.tanh(self.fc_layer1(out))
        out = torch.log_softmax(self.fc_output_layer(out), dim = 1)
        return out

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 14, kernel_size = 5, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 14, out_channels = 28, kernel_size = 5)
        self.fc_layer = nn.Linear(700, 470) #conv2 to fc layer
        self.fc_output_layer = nn.Linear(470, 10) #fc layer to fc output 

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_layer(out))
        out = F.log_softmax(self.fc_output_layer(out), dim = 1)
        
        return out
