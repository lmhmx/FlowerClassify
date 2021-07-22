from MyData import MyData
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from time import time
# PCA降维
if (__name__=="__main__"):
    # 是否显示降维结果
    show_plot = False
    # 是否显示各个阶段的用时
    show_plot_Time = True
    # 记录各个关键时刻的时间
    keyTime = []

    keyTime.append(time())

    data = MyData()
    pictures,labels=data.getData()
    vectorPictures=pictures.reshape((-1,3*80*80))

    keyTime.append(time())

    # 降到二维
    model = PCA(n_components=2)
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

    model = PCA(n_components=3)
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
