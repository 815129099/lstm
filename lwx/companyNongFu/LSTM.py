import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

def run():
    # 导入数据
    df = pd.read_csv('./data_nong_1.csv', usecols=[4])
    dataset = df.values.astype('float32')

    # 归一化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 创建输入数据集
    look_back = 5  # 可调整的时间步长，根据实际情况设置
    X, Y = create_dataset(dataset, look_back)

    # 将数据集重塑为符合LSTM要求的形状 [样本数, 时间步长, 特征数]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(look_back, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 拟合模型
    model.fit(X, Y, epochs=100, batch_size=1, verbose=2)

    # 使用训练好的模型进行预测
    test_data = dataset[len(dataset) - look_back:, :]
    test_data = np.reshape(test_data, (1, len(test_data), 1))
    predicted_values = model.predict(test_data)
    predicted_values = scaler.inverse_transform(predicted_values)

    # 画图
    train_data = scaler.inverse_transform(dataset)
    plt.plot(train_data, label='True Data')
    plt.plot(range(len(train_data), len(train_data) + len(predicted_values[0])), predicted_values[0], label='Predicted Data')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run()
