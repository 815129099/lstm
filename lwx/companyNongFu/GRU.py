import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from keras.models import Sequential
from keras.layers import GRU, Dense

def create_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(32, input_shape=(input_shape[1], input_shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def run():
    # 加载和预处理数据
    df1 = pd.read_csv('./data_nong_1.csv')
    min_max_scaler = MinMaxScaler()
    df0 = min_max_scaler.fit_transform(df1.iloc[:, 4].values.reshape(-1, 1))
    df = pd.DataFrame(df0, columns=['power_num'])

    # 为GRU准备数据
    sequence_length = 24  # 一天的数据
    x, y = [], []
    for i in range(len(df) - sequence_length + 1):
        x.append(df.iloc[i:i + sequence_length, :].values)
        y.append(df.iloc[i + sequence_length - 1, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # 增加一个维度

    # 将数据分为训练集和测试集
    train_size = int(len(df) * 0.75)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 创建GRU模型
    model = create_gru_model(x_train.shape)

    # 打印模型摘要
    print(model.summary())

    # 训练模型
    model.fit(x_train, y_train, epochs=8, batch_size=32)

    # 进行预测
    y_predict = model.predict(x_test)
    y_predict = y_predict.reshape(-1, 1)

    # 对预测值和实际值进行反归一化
    y_predict = min_max_scaler.inverse_transform(np.hstack([np.zeros((len(y_predict), 1)), y_predict]))
    y_predict = y_predict[:, -1]

    # 反序列化实际值
    y_test = min_max_scaler.inverse_transform(np.hstack([np.zeros((len(y_test), 1)), y_test.reshape(-1, 1)]))
    y_test = y_test[:, -1]

    # 绘制结果
    draw = pd.concat([pd.DataFrame(y_test), pd.DataFrame(y_predict)], axis=1)
    draw.iloc[:, 0].plot(figsize=(12, 6))
    draw.iloc[:, 1].plot(figsize=(12, 6))
    plt.legend(('real', 'gru'), loc='upper right', fontsize='15')
    plt.title("test", fontsize='30')
    plt.show()

    # 输出评估指标
    mape = mean_absolute_percentage_error(y_predict, y_test)
    rmse = np.sqrt(mean_squared_error(y_predict, y_test))
    print("MAPE: {}, RMSE: {}".format(mape, rmse))

    # 将结果保存到新的CSV文件
    real_data_csv = pd.DataFrame(data=y_test, columns=['real'])
    pred_data_csv = pd.DataFrame(data=y_predict, columns=['gru'])
    dataframe = real_data_csv.join(pred_data_csv)
    dataframe.to_csv('./new_data_nong_gru_v1.csv', index=False, mode='w', sep=',')

# 运行脚本
if __name__ == '__main__':
    run()
