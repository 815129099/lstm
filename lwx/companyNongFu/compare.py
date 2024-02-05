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
def showData():
    #读取数据
    lstm_data_csv = pd.read_csv('./new_data_nong_lstm_v1.csv', usecols=[0,1])
    #去除读入的数据中含有NaN的行。
    lstm_data_csv = lstm_data_csv.dropna()
    lstm_data_set = lstm_data_csv.values
    #对数据类型进行转换
    lstm_data_set = lstm_data_set.astype('float32')

    #读取数据
    bp_data_csv = pd.read_csv('./new_data_nong_bp_v1.csv', usecols=[1])
    #去除读入的数据中含有NaN的行。
    bp_data_csv = bp_data_csv.dropna()
    bp_data_set = bp_data_csv.values
    #对数据类型进行转换
    bp_data_set = bp_data_set.astype('float32')

    #读取数据
    svm_data_csv = pd.read_csv('./new_data_nong_svm.csv', usecols=[1])
    #去除读入的数据中含有NaN的行。
    svm_data_csv = svm_data_csv.dropna()
    svm_data_set = svm_data_csv.values
    #对数据类型进行转换
    svm_data_set = svm_data_set.astype('float32')

    #真实数据
    real_date_set = lstm_data_set[+0:, 0]
    #计算误差取最小的
    min_mape = 1
    min_rmse = 1
    index = 1
    #余数
    flag = int(len(lstm_data_set) % 24)
    # 假设 lstm_data_set
    lstm_data_set = lstm_data_set[flag:]
    bp_data_set = bp_data_set[-len(lstm_data_set):]
    svm_data_set = svm_data_set[-len(lstm_data_set):]

    # 如果想要重置索引，可以使用以下代码
    date_num = int(len(lstm_data_set) / 24)

    lstm_min_data = lstm_data_set
    bp_min_data = bp_data_set
    svm_min_data = svm_data_set

    lstm_total_data = 0
    bp_total_data = 0
    svm_total_data = 0
    real_total_data = 0
    for e in range(date_num):
        if (e < 20):
            continue
        lstm_current_data = lstm_data_set[e*24:(e+1)*24]
        bp_current_data = bp_data_set[e*24:(e+1)*24]
        svm_current_data = svm_data_set[e*24:(e+1)*24]
        temp_real = lstm_current_data[+0:, 0]  # 取真实电量值
        temp_lstm_pred = lstm_current_data[+0:, 1]  # 取lstm预测电量值
        temp_bp_pred = bp_current_data[+0:, 0]  # 取bp预测电量值
        temp_svm_pred = svm_current_data[+0:, 0]  # 取bp预测电量值

        lstm_total_data+=temp_lstm_pred
        bp_total_data+=temp_bp_pred
        svm_total_data+=temp_svm_pred
        real_total_data+=temp_real

        #mape
        lstm_current_mape = mean_absolute_percentage_error(temp_real, temp_lstm_pred)
        bp_current_mape = mean_absolute_percentage_error(temp_real, temp_bp_pred)
        svm_current_mape = mean_absolute_percentage_error(temp_real, temp_svm_pred)

        #mse
        lstm_current_mse = mean_squared_error(temp_real, temp_lstm_pred)
        bp_current_mse = mean_squared_error(temp_real, temp_bp_pred)
        svm_current_mse = mean_squared_error(temp_real, temp_svm_pred)

        #rmse
        lstm_current_rmse = np.sqrt(lstm_current_mse)
        bp_current_rmse = np.sqrt(bp_current_mse)
        svm_current_rmse = np.sqrt(svm_current_mse)
        print('e: {}, lstm_mape: {}, lstm_mse: {}, lstm_rmse: {}'.format(e , lstm_current_mape, lstm_current_mse, lstm_current_rmse))
        print('e: {}, bp_mape: {}, bp_mse: {}, bp_rmse: {}'.format(e , bp_current_mape, bp_current_mse, bp_current_rmse))
        print('e: {}, svm_mape: {}, svm_mse: {}, svm_rmse: {}'.format(e , svm_current_mape, svm_current_mse, svm_current_rmse))

        # if (lstm_current_mape+lstm_current_rmse < min_mape+min_rmse):
        # if (e == 2):
        #     min_mape = lstm_current_mape
        #     min_rmse = lstm_current_rmse
        #     lstm_min_data = lstm_current_data
        #     bp_min_data = bp_current_data
        #     svm_min_data = svm_current_data
        #     index = e

    print("min_mape:{}".format(min_mape))
    print("min_rmse:{}".format(min_rmse))
    print("index:{}".format(index))

    print("lstm_total mape:{}, mse:{},rmse:{}",mean_absolute_percentage_error(real_total_data, lstm_total_data), mean_squared_error(real_total_data, lstm_total_data), np.sqrt(mean_squared_error(real_total_data, lstm_total_data)))
    print("bp_total mape:{}, mse:{},rmse:{}",mean_absolute_percentage_error(real_total_data, bp_total_data), mean_squared_error(real_total_data, bp_total_data), np.sqrt(mean_squared_error(real_total_data, bp_total_data)))
    print("svm_total mape:{}, mse:{},rmse:{}",mean_absolute_percentage_error(real_total_data, svm_total_data), mean_squared_error(real_total_data, svm_total_data), np.sqrt(mean_squared_error(real_total_data, svm_total_data)))
    real = lstm_min_data[+0:, 0]  #取电量值
    lstm_pred = lstm_min_data[+0:, 1]  #取电量值
    bp_pred = bp_min_data[+0:, 0]  #取电量值
    svm_pred = svm_min_data[+0:, 0]  #取电量值
    plt.plot(real, 'r', label='真实值', linewidth=0.5)
    plt.plot(lstm_pred, 'b', label='mRMR-LSTM', linewidth=0.5)
    plt.plot(bp_pred, 'g', label='bp', linewidth=0.5)
    plt.plot(svm_pred, 'Purple', label='svm', linewidth=0.5)
    plt.legend(loc='best')
    plt.xlabel("时间点/h")
    plt.ylabel("负荷/kW")
    plt.show()


if __name__ == '__main__':
    #显示数据
    showData()