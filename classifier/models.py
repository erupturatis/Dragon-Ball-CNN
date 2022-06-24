import torch.nn as nn
import torch.nn.functional as F
import torch

class Models(object):

    def __init__(self) -> None:
        pass

    @staticmethod
    def model1():
        global NeuralNetwork
        class NeuralNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 3 64 64
                self.input = nn.Conv2d(3,5,kernel_size=5,padding=2,stride=1)
                # 6 32 32
                self.conv1 = nn.Conv2d(5,10,kernel_size=5,padding=2,stride=1)
                # 10 16 16
                self.conv2 = nn.Conv2d(10,15,kernel_size=5,padding=2,stride=1)
                #15 8 8
                self.conv3 = nn.Conv2d(15,20,kernel_size=3,padding=1,stride=1)
                #30 4 4
                self.expected_size = 20*(4**2)
                self.fc1 = nn.Linear(self.expected_size,50)
                self.out = nn.Linear(50,4)

            def forward(self, x):
                #print(x.shape)
                x = self.input(x)
                x = F.max_pool2d(x,2)
                x = F.relu(x)
                #print(x.shape)
                x = self.conv1(x)
                x = F.max_pool2d(x,2)
                x = F.relu(x)
                #print(x.shape)
                x = self.conv2(x)
                x = F.max_pool2d(x,2)
                x = F.relu(x)
                #print(x.shape)
                x = self.conv3(x)
                x = F.max_pool2d(x,2)
                x = F.relu(x)
                #print(x.shape)
                x = x.reshape((x.shape[0],self.expected_size))
                #print(x.shape)
                #print("bef relu")
                x = self.fc1(x)
                x = F.relu(x)
                #print("last relu")
                x = self.out(x)
                return x
       
        net = NeuralNetwork()
        lossfun = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(),lr=.001)

        return net,lossfun,optimizer
                
                
