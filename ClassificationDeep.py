from os import DirEntry
import torch
from time import asctime, localtime
from torch import cuda
from torch.utils.data import TensorDataset,DataLoader
import torchvision.transforms as transforms
import torchvision.models as torchModels
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from time import time
import pandas as pd
import numpy as np
from MyData import MyData    
from sklearn.model_selection import train_test_split
from sam import SAMSGD

# 模型，resnet50
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.resnet=torchModels.resnet50(pretrained=True)
        fc_features=self.resnet.fc.in_features
        self.resnet.fc=nn.Linear(fc_features,20)

        
    def forward(self,x):
        return self.resnet(x)
        
# 设置相关的参数
class paramClass:
    def __init__(self) -> None:
        self.batch_size = 32
        self.epoches = 30
        self.saveModel = False
        self.ModelFileName = "ModelLM.model"
        self.device = 'cuda' if(torch.cuda.is_available()) else 'cpu'
# 对数据做预处理
def transformData(X):
    trans = transforms.Resize(224)
    flip = transforms.RandomHorizontalFlip(p=0.3)
    Y = trans(X)
    Y = flip(Y)
    Y = Y/255
    return Y
# 计算准确率和loss
def calculateAccuracyAndLoss(net,device,loader):
    correct=0
    totol=0
    runloss=[]
    with torch.no_grad():
        for data in loader:
            images,labels=data
            outputs=net(transformData(images.to(device)))
            labels=labels.to(device)
            labels=labels.view(-1)
            loss=criterion(outputs,labels)
            runloss.append(loss.item())
            _,predicted=torch.max(outputs.data,1)
            totol=totol+labels.size(0)
            labels=labels.view(-1)
            correct=correct+(predicted==labels).sum().item()
    acc = correct/totol
    meanloss = np.mean(runloss)
    return acc,meanloss
# 提取数据集
mydata = MyData()
param = paramClass()
all_Pictures,all_Labels=mydata.getData()
train_Pictures,val_Pictures,train_Labels,val_Labels = train_test_split(all_Pictures,all_Labels,test_size=0.2)

mytrainset=TensorDataset(torch.tensor(train_Pictures,dtype=torch.float32),torch.tensor(train_Labels,dtype=torch.long))
myvalset = TensorDataset(torch.tensor(val_Pictures,dtype=torch.float32),torch.tensor(val_Labels,dtype=torch.long))


mytrainloader=DataLoader(dataset=mytrainset,
            batch_size=param.batch_size,
            shuffle=True,
            num_workers=0)
myvalloader=DataLoader(dataset=myvalset,
            batch_size=param.batch_size,
            shuffle=True,
            num_workers=0)

# 设置网络参数
net=MyNet()
net=net.to(param.device)
criterion=nn.CrossEntropyLoss()
# optimizer=optim.Adam(net.parameters(),lr=0.005)
optimizer = SAMSGD(net.parameters(),lr = 0.1,rho = 0.05)

# 损失和准确率以及运行时间
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []
keyTime=[time()]

# 运行模型
for epoch in range(param.epoches):
    # 模型训练
    for i,data in enumerate(mytrainloader,0):
        inputs,labels=data
        labels=labels.view(-1)
        inputs=inputs.to(param.device)
        def closure():
            optimizer.zero_grad()
            outputs=net(transformData(inputs))
            loss=criterion(outputs,labels.to(param.device))
            loss.backward()
            return loss
        loss = optimizer.step(closure)

    # 计算正确率
    tmp=calculateAccuracyAndLoss(net,param.device,mytrainloader)
    train_accuracy.append(tmp[0])
    train_loss.append(tmp[1])

    tmp=calculateAccuracyAndLoss(net,param.device,myvalloader)     

    val_accuracy.append(tmp[0])
    val_loss.append(tmp[1])
    keyTime.append(time())
    print("epoch {:3d} totol time {:3.1f} epoch time {:3.1f} train loss {:2.3f} train acc {:3.3f} val loss {:2.3f} val accuracy {:3.3f}".format(
        epoch,keyTime[-1]-keyTime[0],keyTime[-1]-keyTime[-2],train_loss[-1],train_accuracy[-1],val_loss[-1],val_accuracy[-1]
    ))
    