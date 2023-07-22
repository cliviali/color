
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
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        #out = self.sigmoid(out)*2-1
        return out

model = NeuralNet()       
model.load_state_dict(torch.load('net_params.pkl')) 



inputs = torch.tensor(test).float().view(-1, 6)
out = model(inputs)
out = np.array(out.detach().numpy())
out = output_scaler.inverse_transform(out)+ recipe

x=out.reshape(-1)
print(x)
y=np.tile(out1,(5,1)) .reshape(-1)
print(y)


# 绘制回归线
plt.plot(x, y, 'o', label='Actual')
plt.show()