
# 对数据集进行处理，将jpg图片转换成np

import numpy as np
import os
from matplotlib.pyplot import imread
class MyData:
    def __init__(self) -> None:
        self.Flowers = np.array(["clove","flowering peach","cherry","bauhinia","Chrysanthemum peach","Prunus humilis","Violet Orychophragmus","peony"])
        pass
    def getTextLabels(self,label):
        return np.float[label]
    def  getData(self):
        if(os.path.exists("Pictures.npy") and os.path.exists("labels.npy")):
            Pictures = np.load("Pictures.npy")
            labels = np.load("labels.npy")
            return Pictures,labels
        path = "train"
        files = os.listdir(path)
        N = len(files)
        Pictures = np.zeros((N,3,80,80))
        labels = np.zeros(N,)
        for index,fileName in  enumerate(files):
            APicture = imread(path+"/"+fileName)
            APicture = APicture.transpose(2,0,1)
            Pictures[index] = APicture
            labels[index] = int(fileName[0])
        np.save("Pictures",Pictures)
        np.save("labels",labels)
        return Pictures,labels
    