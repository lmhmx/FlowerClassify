from numpy.core.defchararray import index
from MyData import MyData
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from time import time
import torch
from torch import optim
from torch import nn
from torchvision.models import resnet18
# 自编码器降维
class AutoEncoderModel(nn.Module):
    def __init__(self,n_components):
        super().__init__()
        self.n_components = n_components
        self.encoder = nn.Sequential(
            # 80
            nn.Conv2d(3,64,3),
            # 78
            nn.ReLU(),
            nn.AvgPool2d(3),
            # 26
            nn.Conv2d(64,32,3),
            # 24
            nn.ReLU(),
            nn.AvgPool2d(2),
            # 12
            nn.Conv2d(32,3,3),
            # 10
            nn.Flatten(),
            # 3*10*10
            nn.Linear(300,self.n_components)

        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_components,300),
            nn.ReLU(),
            nn.Linear(300,80*80*3)
        )
    def forward(self,X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X
    def getEncoderResult(self,X):
        X = self.encoder(X)
        return X

class AutoEncoder:
    def __init__(self,n_components,epoches = 20, batch_size = 256) -> None:
        
        self.n_components = n_components
        self.epoches = epoches
        self.device = 'cuda' if(torch.cuda.is_available()) else 'cpu'
        self.batch_size = batch_size
        
        
        self.model = AutoEncoderModel(self.n_components)
    def fit(self,X):
        # 输入是 N* 3 * 80*80的图片
        Criterion = nn.MSELoss()
        self.model.to(self.device)
        optimizer = optim.Adam(lr=0.01,params= self.model.parameters())
        N = len(X)
        batch_num = (N-1)//self.batch_size+1
        allLoss=[]
        for epoch in range(self.epoches):
            index_order = np.random.permutation(N)
            for i in range(batch_num):
                tmp_choose = index_order[i*self.batch_size:np.min([N,(i+1)*self.batch_size])]
                aX = X[tmp_choose]
                aX = torch.tensor(aX,dtype=torch.float32)
                aX = aX.to(self.device)
                optimizer.zero_grad()

                result = self.forward(aX)
                aX = torch.reshape(aX,(-1,3*80*80))
                loss = Criterion(aX,result)
                loss.backward()
                allLoss.append(loss.to('cpu').item())
                optimizer.step()
        plt.plot(allLoss)
        plt.show()
    def forward(self,X):
        X=self.model(X)
        return X

    def transform(self,X):
        torch.no_grad()
        N = len(X)
        batch_num = (N-1)//self.batch_size+1
        
        
        index_order = np.random.permutation(N)
        transformResult = np.zeros((N,self.n_components))
        for i in range(batch_num):
            tmp_choose = index_order[i*self.batch_size:np.min([N,(i+1)*self.batch_size])]
            aX = X[tmp_choose]
            aX = torch.tensor(aX,dtype=torch.float32)
            aX = aX.to(self.device)
            result = self.model.getEncoderResult(aX)
            result = result.to('cpu').detach().numpy()
            transformResult[tmp_choose] = result
        
        return transformResult



if (__name__=="__main__"):
    # 是否显示降维结果
    show_plot = True
    # 是否显示各个阶段的用时
    show_plot_Time = True
    # 记录各个关键时刻的时间
    keyTime = []

    keyTime.append(time())

    data = MyData()
    pictures,labels=data.getData()
    vectorPictures=pictures # .reshape((-1,3*80*80))

    keyTime.append(time())

    # 降到二维
    model = AutoEncoder(n_components=2)
    model.fit(vectorPictures)
    
    keyTime.append(time())

    result = model.transform(vectorPictures)

    keyTime.append(time())
    if(show_plot):
        x_min = np.min(result[:,0])
        x_max = np.max(result[:,0])
        y_min = np.min(result[:,1])
        y_max = np.max(result[:,1])
        plt.figure(1)
        for i in range(8):
            indexLabel = labels==i
            tmpResult = result[indexLabel]
            tmpLabel = labels[indexLabel]
            plt.subplot(2,4,i+1)
            plt.scatter(tmpResult[:,0],tmpResult[:,1],s=1)
            plt.axis([x_min-0.1*(x_max-x_min),x_max+0.1*(x_max-x_min),y_min-0.1*(y_max-y_min),y_max+0.1*(y_max-y_min)])
            plt.title(data.Flowers[i])
        plt.show()
        plt.figure(2)
        legends = []
        for i in range(8):
            indexLabel = labels==i
            tmpResult = result[indexLabel]
            tmpLabel = labels[indexLabel]
            plt.scatter(tmpResult[:,0],tmpResult[:,1],s=1)
        
        plt.legend(data.Flowers)
        plt.axis([x_min-0.1*(x_max-x_min),x_max+0.1*(x_max-x_min),y_min-0.1*(y_max-y_min),y_max+0.1*(y_max-y_min)])
        plt.show()

    keyTime.append(time())

    model = AutoEncoder(n_components=3)
    model.fit(vectorPictures)

    keyTime.append(time())

    result = model.transform(vectorPictures)

    keyTime.append(time())
    if(show_plot):
        x_min = np.min(result[:,0])
        x_max = np.max(result[:,0])
        y_min = np.min(result[:,1])
        y_max = np.max(result[:,1])
        z_min = np.min(result[:,2])
        z_max = np.max(result[:,2])
        f=plt.figure(1)
        
        for i in range(8):
            indexLabel = labels==i
            tmpResult = result[indexLabel]
            tmpLabel = labels[indexLabel]
            
            ax=plt.subplot(2,4,i+1,projection='3d')
            # ax=Axes3D(f)
            ax.scatter(tmpResult[:,0],tmpResult[:,1],tmpResult[:,2],s=1)
            ax.set_zlim3d(z_min-0.1*(z_max-z_min),z_max+0.1*(z_max-z_min))
            ax.set_xlim3d(x_min-0.1*(x_max-x_min),x_max+0.1*(x_max-x_min))
            ax.set_ylim3d(y_min-0.1*(y_max-y_min),y_max+0.1*(y_max-y_min))
            
            plt.title(data.Flowers[i])

        plt.show()
        plt.figure(2)
        ax=plt.subplot(projection='3d')
        legends = []
        for i in range(8):
            indexLabel = labels==i
            tmpResult = result[indexLabel]
            tmpLabel = labels[indexLabel]
            ax.scatter(tmpResult[:,0],tmpResult[:,1],tmpResult[:,2],s=1)
        
        plt.legend(data.Flowers)
        ax.set_zlim3d(z_min-0.1*(z_max-z_min),z_max+0.1*(z_max-z_min))
        ax.set_xlim3d(x_min-0.1*(x_max-x_min),x_max+0.1*(x_max-x_min))
        ax.set_ylim3d(y_min-0.1*(y_max-y_min),y_max+0.1*(y_max-y_min))
        plt.show()
    
    print(keyTime)

    if(show_plot_Time):
        tmp_KeyTime = np.array(keyTime)
        # plt.plot(tmp_KeyTime)
        # plt.show()
        yticks = np.diff(tmp_KeyTime)
        plt.plot(yticks)
        plt.yticks(yticks,yticks)
        plt.show()
