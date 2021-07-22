from numpy.core.defchararray import index
from MyData import MyData
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from time import time

# PCA方法和Kmeans
class MyCluster:
    def __init__(self,n_clusters) -> None:
        self.model = KMeans(n_clusters=n_clusters,init='random')
        self.pca = PCA(n_components=200)
    def fit(self,X):
        self.pca.fit(X)
        X = self.pca.transform(X)
        self.model.fit(X)

        
    def fit_transform(self,X):
        X = self.pca.transform(X)
        result=self.model.predict(X)
        return result
def calAccuracy(y_truth,y_pred):
    labels = np.unique(y_pred)
    correct = 0
    totol = 0
    for i in labels:
        choose = y_pred==i

        truth = y_truth[choose]
        truth = truth.astype(np.int64)

        correct = correct + np.max(np.bincount(truth))
        totol = totol+len(truth)
    return correct/totol

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
    N = len(labels)
    # index_choose = np.random.permutation(N)[0:2000]
    # pictures,labels = pictures[index_choose],labels[index_choose]
    vectorPictures=pictures.reshape((-1,3*80*80))

    keyTime.append(time())
    
    
    model = MyCluster(n_clusters=8)
    model.fit(vectorPictures)
    
    keyTime.append(time())

    result = model.fit_transform(vectorPictures)
    keyTime.append(time())

    NMI = metrics.normalized_mutual_info_score(labels,result)
    RI = metrics.rand_score(labels,result)
    FMI = metrics.fowlkes_mallows_score(labels,result)
    Accuracy = calAccuracy(labels,result)
    print("NMI {:.3f} RI {:.3f} FMI {:.3f} Accuracy {:.3f} train time {:.3f} predict time {:.3f}".format(
        NMI,RI,FMI,Accuracy,keyTime[2]-keyTime[1],keyTime[3]-keyTime[2]
    ))
    print(keyTime)
