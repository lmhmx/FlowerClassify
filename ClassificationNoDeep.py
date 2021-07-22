import cv2
import os
import pandas as pd

### Import your package here
import matplotlib.pyplot as plt
import numpy as np
from MyData import MyData
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import joblib
from time import time
# SVM模型
class MyModel:
    def __init__(self,C=1) -> None:
        self.model = SVC(C=C,kernel='rbf')
        
    def fit(self,X,y):
        self.model.fit(X,y)
    def setParam(self,C=1):
        self.model = SVC(C=C,kernel='rbf')

    def predict(self,X):
        y=self.model.predict(X)
        return y
        
    def save(self,fileName):
        joblib.dump(self.model,filename=fileName)
    def load(self,fileName):
        self.model = joblib.load(filename=fileName)
# PCA降维处理
class dataTransform:
    def __init__(self,n=100) -> None:
        self.model = PCA(n_components=n)
    def setParam(self,n=100):
        self.model = PCA(n_components=n)

    def transform(self,X):
        X = np.reshape(X,(-1,3*80*80))
        result = self.model.transform(X)
        return result
    def fit_transform(self,X):
        X = np.reshape(X,(-1,3*80*80))
        result = self.model.fit_transform(X)
        return result




# 定义参数
class paramClass:
    def __init__(self) -> None:
        self.model = MyModel()
        self.transformer = dataTransform()
        self.saveModel = False
        self.ModelFileName = "ModelLM.model"

if __name__ == '__main__':
    param = paramClass()
    all_Time=[]
    all_Train_Accuracy=[]
    all_val_Accuracy=[]
    all_C = []
    all_n_components = []
    # 以两层循环遍历惩罚因子和降维数
    for C in np.power(2,np.linspace(-4,4,9)):
        for n in np.power(2,np.linspace(1,9,9)):
            n=int(n)
            keyTime = []
            keyTime.append(time())

            param.transformer.setParam(n)
            param.model.setParam(C)

            model = param.model
            mydata = MyData()
            

            allPictures,allLabels=mydata.getData()

            keyTime.append(time())

            train_Pictures,val_Pictures,train_Labels,val_Labels = train_test_split(allPictures,allLabels,test_size=0.2)
            
            keyTime.append(time())

            train_Pictures = param.transformer.fit_transform(train_Pictures)

            keyTime.append(time())

            val_Pictures = param.transformer.transform(val_Pictures)

            keyTime.append(time())

            model.fit(train_Pictures,train_Labels)

            keyTime.append(time())

            train_Pred = model.predict(train_Pictures)

            keyTime.append(time())

            val_Pred = model.predict(val_Pictures)

            keyTime.append(time())

            train_Accuracy = accuracy_score(train_Labels,train_Pred)
            val_Accuracy = accuracy_score(val_Labels,val_Pred)
            print([C,n])
            print("train acc {:.4f} val acc {:.4f} ".format(train_Accuracy,val_Accuracy))
            print(np.diff(keyTime))
            all_Train_Accuracy.append(train_Accuracy)
            all_val_Accuracy.append(val_Accuracy)
            all_Time.append(keyTime[-1]-keyTime[0])
            all_C.append(C)
            all_n_components.append(n)

            if(param.saveModel):
                model.save(param.ModelFileName)
    # 将结果显示出来
    f=plt.figure(1)
    
    ax = plt.subplot(1,3,1,projection='3d')
    
    ax.scatter(np.log2(all_C),np.log2(all_n_components),all_Time,s=2)
    ax = plt.subplot(1,3,2,projection='3d')
    ax.scatter(np.log2(all_C),np.log2(all_n_components),all_Train_Accuracy,s=2)
    ax = plt.subplot(1,3,3,projection='3d')
    ax.scatter(np.log2(all_C),np.log2(all_n_components),all_val_Accuracy,s=2)
    
    plt.show()
    print(all_C)
    print(all_n_components)
    print(all_Time)
    print(all_Train_Accuracy)
    print(all_val_Accuracy)