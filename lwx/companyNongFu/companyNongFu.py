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

def run():
    #读取数据
    data_csv = pd.read_csv('./data_nong_1.csv', usecols=[0,1,2,3,4])

    # 数据处理，归一化
    #去除读入的数据中含有NaN的行。
    data_csv = data_csv.dropna()
    dataset = data_csv.values
    #对数据类型进行转换
    dataset = dataset.astype('float32')
    real_set = dataset[+0:, 4]
    np_max = np.max(real_set)
    np_min = np.min(real_set)
    scalar = np_max - np_min
    # dataset = dataset[:, 0: 5]
    dataset = dataset / (dataset.max(axis=0) - dataset.min(axis=0))
    # 创建好输入输出
    data_X, data_Y = create_dataset(dataset)
    # 划分训练集和测试集，75% 作为训练集
    train_size = int(len(data_X) * 0.75)
    test_size = len(data_X) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[test_size:]
    test_Y = data_Y[train_size:]

    #最后，我们需要将数据改变一下形状，因为 RNN 读入的数据维度是 (seq, batch, feature)，
    # 所以要重新改变一下数据的维度，这里只有一个序列，所以 batch 是 1，而输入的 feature
    # 就是我们希望依据的几个月份，这里我们定的是两个月份，所以 feature 就是 2.

    #序列长度、序列个数、特征数
    train_X = train_X.reshape(len(train_X),1, 5)
    train_Y = train_Y.reshape(len(train_Y),1, 1)
    #将数组array转换为张量Tensor,Tensors 类似于 NumPy 的 ndarrays ，同时 Tensors 可以使用 GPU 进行计算。

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda')
    train_x = torch.from_numpy(train_X).to(device)
    train_y = torch.from_numpy(train_Y).to(device)


    net = lstm_reg(5, 4).to(device)

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
    test_X = test_X.reshape(len(test_X), 1, 5)
    test_X = torch.from_numpy(test_X).to(device)
    var_data = Variable(test_X)
    pred_test = net(var_data)  # 测试集的预测结果

    # 改变输出的格式
    pred_test = pred_test.cpu().view(-1).data.numpy()
    # 画出实际结果和预测的结果
    pred_test = list(map(lambda x: x * scalar, pred_test))
    #追加
    pred_data_csv = pd.DataFrame(pred_test)
    pred_data_csv.columns=["pred"]

    dataframe = data_csv.join(pred_data_csv)
    dataframe.to_csv('./new_data_nong_1.csv',index=False,mode='w',sep=',')

    # pred_test = pred_test[-24:]
    # real_set = list(map(lambda x: x * scalar, real_set))
    real_set = real_set[test_size:]
    # real_set = real_set[-24:]
    plt.plot(pred_test, 'r', label='LSTM')
    plt.plot(real_set, 'b', label='real')
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
        dataY.append(dataset[i+look_back][4])
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
    data_csv = pd.read_csv('./data_nong_1.csv', usecols=[4])

    # 数据处理，归一化
    #去除读入的数据中含有NaN的行。
    data_csv = data_csv.dropna()
    dataset = data_csv.values
    #对数据类型进行转换
    dataset = dataset.astype('float32')

    plt.plot(dataset)
    plt.show()


if __name__ == '__main__':
    #训练并保存模型
    run()
    #加载模型
    # load()
    #显示数据
    # showData()