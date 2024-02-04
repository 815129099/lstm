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

# 这里定义好模型，模型的第一部分是一个两层的 RNN，每一步模型接受两个月的输入作为特征，
# 得到一个输出特征。接着通过一个线性层将 RNN 的输出回归到流量的具体数值，这里我们需要用 view 来重新排列，
# 因为 nn.Linear 不接受三维的输入，所以我们先将前两维合并在一起，然后经过线性层之后再将其分开，最后输出结果。
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归，用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是二维张量

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape
        #view()相当于reshape、resize，重新调整Tensor的形状。
        x = x.view(s * b, h)
        x = self.reg(x)
        #view中一个参数定为-1，代表自动调整这个维度上的元素个数，以保证元素的总数不变。
        x = x.view(s, b, -1)
        return x
    # def forward(self, input_seq):
    #     batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
    #     h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
    #     c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
    #     # output(batch_size, seq_len, num_directions * hidden_size)
    #     output, _ = self.lstm(input_seq, (h_0, c_0))  # output(5, 30, 64)
    #     pred = self.linear(output)  # (5, 30, 1)
    #     pred = pred[:, -1, :]  # (5, 1)
    #     return pred

#显示数据
def showData():
    #读取数据
    lstm_data_csv = pd.read_csv('./new_data_nong_lstm_0903.csv', usecols=[2,3])
    #去除读入的数据中含有NaN的行。
    lstm_data_csv = lstm_data_csv.dropna()
    lstm_data_set = lstm_data_csv.values
    #对数据类型进行转换
    lstm_data_set = lstm_data_set.astype('float32')

    #读取数据
    bp_data_csv = pd.read_csv('./new_data_nong_bp.csv', usecols=[1])
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
    date_num = int(len(lstm_data_set) / 24)

    lstm_min_data = lstm_data_set
    bp_min_data = bp_data_set
    svm_min_data = svm_data_set
    for e in range(date_num):
        lstm_current_data = lstm_data_set[e*24:(e+1)*24]
        bp_current_data = bp_data_set[e*24:(e+1)*24]
        svm_current_data = svm_data_set[e*24:(e+1)*24]
        temp_real = lstm_current_data[+0:, 0]  # 取真实电量值
        temp_lstm_pred = lstm_current_data[+0:, 1]  # 取lstm预测电量值
        temp_bp_pred = bp_current_data[+0:, 0]  # 取bp预测电量值
        temp_svm_pred = svm_current_data[+0:, 0]  # 取bp预测电量值

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
        if (e == 2):
            min_mape = lstm_current_mape
            min_rmse = lstm_current_rmse
            lstm_min_data = lstm_current_data
            bp_min_data = bp_current_data
            svm_min_data = svm_current_data
            index = e

    print("min_mape:{}".format(min_mape))
    print("min_rmse:{}".format(min_rmse))
    print("index:{}".format(index))
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