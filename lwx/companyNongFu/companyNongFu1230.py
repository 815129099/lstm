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

def inverse_transform_col(scaler, y, n_col):
    '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
    y = y.copy()
    y -= scaler.min_[n_col]
    y /= scaler.scale_[n_col]
    return y

def run():
    #读取数据
    data_csv = pd.read_csv('./data_nong_1.csv', usecols=[2,3,4])

    #去除读入的数据中含有NaN的行。
    data_csv = data_csv.dropna()
    dataset = data_csv.values
    #对数据类型进行转换
    dataset = dataset.astype('float32')
    #所有的数据
    # real_set = dataset[+0:, 4]
    real_set = dataset

    # 划分训练集和测试集，75% 作为训练集
    train_size = int(len(real_set) * 0.75)
    test_size = len(real_set) - train_size
    train_data = real_set[:(train_size+24)]
    test_data = real_set[train_size:]

    # 测试数据处理，归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data = min_max_scaler.fit_transform(train_data)
    train_X, train_Y = create_dataset(train_data)

    #最后，我们需要将数据改变一下形状，因为 RNN 读入的数据维度是 (seq, batch, feature)，
    # 所以要重新改变一下数据的维度，这里只有一个序列，所以 batch 是 1，而输入的 feature
    # 就是我们希望依据的几个月份，这里我们定的是两个月份，所以 feature 就是 2.

    #序列长度、序列个数、特征数
    train_X = train_X.reshape(len(train_X),1, 3)
    train_Y = train_Y.reshape(len(train_Y),1, 1)
    #将数组array转换为张量Tensor,Tensors 类似于 NumPy 的 ndarrays ，同时 Tensors 可以使用 GPU 进行计算。

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda')
    train_x = torch.from_numpy(train_X).to(device)
    train_y = torch.from_numpy(train_Y).to(device)

    net = lstm_reg(3, 6).to(device)

    #均方差损失函数
    criterion = nn.MSELoss()
    #优化
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    # 开始训练
    for e in range(1000):
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        # 前向传播
        out = net(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e + 1) % 100 == 0:  # 每 100 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

    net = net.eval()  # 转换成测试模式
    #测试数据归一化
    test_data = min_max_scaler.fit_transform(test_data)
    test_data = test_data.reshape(len(test_data), 1, 3)
    test_data = torch.from_numpy(test_data).to(device)
    var_data = Variable(test_data)
    pred_test = net(var_data)  # 测试集的预测结果

    # 改变输出的格式
    pred_test = pred_test.cpu().view(-1).data.numpy()
    # 画出实际结果和预测的结果
    # pred_test = list(map(lambda x: x * scalar, pred_test))
    pred_test = inverse_transform_col(min_max_scaler, pred_test, n_col=2)  # 对预测值反归一化

    # data_csv = data_csv.drop(data_csv.index[0:train_size-1])
    #追加
    real_data_csv = pd.DataFrame(data=real_set[train_size:],columns=['hour','peak_flat_valley','real'])
    pred_data_csv = pd.DataFrame(data=pred_test,columns=['pred'])
    dataframe = real_data_csv.join(pred_data_csv)
    dataframe.to_csv('./new_data_nong_lstm_0903.csv',index=False,mode='w',sep=',')

    # pred_test = pred_test[-24:]
    # real_set = list(map(lambda x: x * scalar, real_set))
    real_set = real_set[train_size:]
    real_set = real_set[+0:, 2]  #取电量值
    plt.plot(pred_test, 'r', label='LSTM', linewidth=0.5)
    plt.plot(real_set, 'b', label='real', linewidth=0.5)
    plt.legend(loc='best')
    plt.show()


    #保存模型
    torch.save(net.state_dict(),"D:\lunwengit\LTSF-Linear-main\lwx\companyNongFu\companyNongFu.pth")
    # plt.plot(data_csv)
    # plt.show()

# 接着我们进行数据集的创建，我们想通过前面几个月的流量来预测当月的流量，比如我们希望通过前两个月的流量来预测当月的流量，我们可以将前两个月的流量当做输入，当月的流量当做输出。同时我们需要将我们的数据集分为训练集和测试集，通过测试集的效果来测试模型的性能，这里我们简单的将前面几年的数据作为训练集，后面两年的数据作为测试集。
def create_dataset(dataset,look_back=24):
    dataX,dataY = [],[]
    for i in range(len(dataset) - look_back):
        a = dataset[i]
        dataX.append(a)
        dataY.append(dataset[i+look_back][2])
    return np.array(dataX),np.array(dataY)


#加载现有的模型
def load():
    # 读取数据
    data_csv = pd.read_csv('./data_nong_8_20.csv', usecols=[0, 1, 2, 3, 4, 5])

    # 数据处理，归一化
    # 去除读入的数据中含有NaN的行。
    data_csv = data_csv.dropna()
    dataset = data_csv.values
    # 对数据类型进行转换
    dataset = dataset.astype('float32')
    real_set = dataset[+0:, 4]
    np_max = np.max(real_set)
    np_min = np.min(real_set)
    scalar = np_max - np_min

    dataset = dataset / (dataset.max(axis=0) - dataset.min(axis=0))
    # 创建好输入输出
    # data_X, data_Y = create_dataset(dataset)
    test_X = dataset[:,2: 5]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda')

    #加载模型
    # net = lstm_reg(6, 4).to(device)
    net = lstm_reg(3, 4).to(device)
    net.load_state_dict(torch.load("D:\lunwengit\LTSF-Linear-main\lwx\companyNongFu\companyNongFu.pth"))

    net = net.eval()  # 转换成测试模式
    test_X = test_X.reshape(len(test_X), 1, 3)
    test_X = torch.from_numpy(test_X).to(device)
    var_data = Variable(test_X)
    pred_test = net(var_data)  # 测试集的预测结果

    # 改变输出的格式
    pred_test = pred_test.cpu().view(-1).data.numpy()
    # 画出实际结果和预测的结果
    pred_test = list(map(lambda x: x * scalar, pred_test))
    # pred_test = pred_test[-48:]
    # real_set = list(map(lambda x: x * scalar, real_set))
    # real_set = real_set[-48:]
    plt.plot(pred_test, 'r', label='LSTM')
    plt.plot(real_set, 'b', label='real')
    plt.legend(loc='best')
    plt.show()

    MAPE = np.mean(np.abs((pred_test - real_set) / real_set)) * 100
    print('MAPE')
    print(MAPE)
    #追加
    pred_data_csv = pd.DataFrame(pred_test)
    pred_data_csv.columns=["pred"]
    dataframe = data_csv.join(pred_data_csv)
    dataframe.to_csv('./new_data_nong_8_21.csv',index=False,mode='w',sep=',')

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

    #真实数据
    real_date_set = lstm_data_set[+0:, 0]
    #计算误差取最小的
    min_mape = 1
    min_rmse = 1
    index = 1
    date_num = int(len(lstm_data_set) / 24)

    lstm_min_data = lstm_data_set
    bp_min_data = bp_data_set
    for e in range(date_num):
        lstm_current_data = lstm_data_set[e*24:(e+1)*24]
        bp_current_data = bp_data_set[e*24:(e+1)*24]
        temp_real = lstm_current_data[+0:, 0]  # 取真实电量值
        temp_lstm_pred = lstm_current_data[+0:, 1]  # 取lstm预测电量值
        temp_bp_pred = bp_current_data[+0:, 0]  # 取bp预测电量值

        #mape
        lstm_current_mape = mean_absolute_percentage_error(temp_real, temp_lstm_pred)
        bp_current_mape = mean_absolute_percentage_error(temp_real, temp_bp_pred)

        #mse
        lstm_current_mse = mean_squared_error(temp_real, temp_lstm_pred)
        bp_current_mse = mean_squared_error(temp_real, temp_bp_pred)

        #rmse
        lstm_current_rmse = np.sqrt(lstm_current_mse)
        bp_current_rmse = np.sqrt(bp_current_mse)
        print('e: {}, lstm_mape: {}, lstm_mse: {}, lstm_rmse: {}'.format(e , lstm_current_mape, lstm_current_mse, lstm_current_rmse))
        print('e: {}, bp_mape: {}, bp_mse: {}, bp_rmse: {}'.format(e , bp_current_mape, bp_current_mse, bp_current_rmse))

        # if (lstm_current_mape+lstm_current_rmse < min_mape+min_rmse):
        if (e == 2):
            min_mape = lstm_current_mape
            min_rmse = lstm_current_rmse
            lstm_min_data = lstm_current_data
            bp_min_data = bp_current_data
            index = e

    print("min_mape:{}".format(min_mape))
    print("min_rmse:{}".format(min_rmse))
    print("index:{}".format(index))
    real = lstm_min_data[+0:, 0]  #取电量值
    lstm_pred = lstm_min_data[+0:, 1]  #取电量值
    bp_pred = bp_min_data[+0:, 0]  #取电量值
    plt.plot(real, 'r', label='real', linewidth=0.5)
    plt.plot(lstm_pred, 'b', label='LSTM', linewidth=0.5)
    plt.plot(bp_pred, 'g', label='bp', linewidth=0.5)
    plt.legend(loc='best')
    plt.xlabel("时间点/h")
    plt.ylabel("负荷/kW")
    plt.show()


if __name__ == '__main__':
    #训练并保存模型
    # run()
    #加载模型
    # load()
    #显示数据
    showData()