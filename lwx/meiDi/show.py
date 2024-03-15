#运算
import numpy as np
import pandas as pd
import torch
from torch import nn
#张量自动求导
from torch.autograd import Variable
#画图
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
#显示数据
def showAllData():
    #读取数据
    data_csv = pd.read_csv('./mei_di.csv', usecols=[4])
    #去除读入的数据中含有NaN的行。
    #去除读入的数据中含有NaN的行。
    data_csv = data_csv.dropna()
    dataset = data_csv.values
    #对数据类型进行转换
    dataset = dataset.astype('float32')
    plt.plot(dataset)
    plt.xlabel("时间点/h")
    plt.ylabel("负荷/kW")
    plt.show()


if __name__ == '__main__':
    showAllData()