
import torch 
import torch.nn as nn  
import matplotlib.pyplot as plt 
import numpy as np
from 数据处理 import *
from 聚类处理 import knn
from 归一化 import *
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(6, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 30)
        self.fc4 = nn.Linear(30, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)        
        out = 2*self.sigmoid(out)
        return out

model = NeuralNet()       
model.load_state_dict(torch.load('net_params.pkl')) 



inputs = torch.tensor(data[:,0:6]).float().view(-1, 6)
out = model(inputs)
out = np.array(out.detach().numpy())

x=out.reshape(-1)
y=data[:,6:].reshape(-1)



# 绘制回归线
plt.plot(x, y, 'o', label='Actual')
plt.show()