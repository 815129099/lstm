# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def inverse_transform_col(scaler, y, n_col):
    y = y.copy()
    y -= scaler.min_[n_col]
    y /= scaler.scale_[n_col]
    return y


def run():
    # 导入数据
    df1 = pd.read_csv('./data_nong_1.csv', usecols=[0, 1, 2, 3, 4])
    df1 = df1.astype('float32')


    # 仅对第5列进行数据归一化
    min_max_scaler = MinMaxScaler()
    df1[4] = min_max_scaler.fit_transform(df1[[df1.columns[4]]])  # 只对第5列进行归一化

    x = df1.iloc[:, :-1]
    y = df1.iloc[:, -1]

    # 划分训练集测试集
    train_size = int(len(df1) * 0.25)
    x_train, x_test = x.iloc[:-train_size], x.iloc[-train_size:]
    y_train, y_test = y.iloc[:-train_size], y.iloc[-train_size:]

    # 支持向量机回归模型
    svm_regressor = SVR(kernel='rbf', C=1.0, epsilon=0.15)
    svm_regressor.fit(x_train, y_train)

    # 预测
    y_predict = svm_regressor.predict(x_test)

    # 反归一化
    y_predict = inverse_transform_col(min_max_scaler, y_predict, n_col=-1)
    y_test = inverse_transform_col(min_max_scaler, y_test.values, n_col=-1)

    # 画图
    draw = pd.concat([pd.DataFrame(y_test), pd.DataFrame(y_predict)], axis=1)
    draw.iloc[:, 0].plot(figsize=(12, 6))
    draw.iloc[:, 1].plot(figsize=(12, 6))
    plt.legend(('real', 'predict'), loc='upper right', fontsize='15')
    plt.title("Test Data", fontsize='30')
    plt.show()

    # 输出精度指标
    mape = mean_absolute_percentage_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    print("MAPE:{}, RMSE:{}".format(mape, rmse))

    # 保存预测结果
    real_data_csv = pd.DataFrame(data=y_test, columns=['real'])
    pred_data_csv = pd.DataFrame(data=y_predict, columns=['pred'])
    dataframe = real_data_csv.join(pred_data_csv)
    dataframe.to_csv('./new_data_nong_svm.csv', index=False, mode='w', sep=',')


# 运行
if __name__ == '__main__':
    run()
