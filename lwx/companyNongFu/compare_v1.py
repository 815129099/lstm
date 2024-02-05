import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt



def calculate_metrics(real, pred):
    mape = mean_absolute_percentage_error(real, pred)
    mse = mean_squared_error(real, pred)
    rmse = np.sqrt(mse)
    return mape, mse, rmse

def evaluate_model(data, model_name):
    min_mape_rmse = float('inf')
    min_mape_rmse_day = None

    date_num = int(len(data) / 24)

    for e in range(date_num):
        current_data = data[e * 24:(e + 1) * 24]
        real = current_data[:, 0]
        pred = current_data[:, 1]

        mape, mse, rmse = calculate_metrics(real, pred)

        print('Day {}: {} MAPE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}'.format(e, model_name, mape, mse, rmse))

        if mape + rmse < min_mape_rmse:
            min_mape_rmse = mape + rmse
            min_mape_rmse_day = current_data

    print("Minimum MAPE + RMSE for {}: {:.4f}".format(model_name, min_mape_rmse))

    return min_mape_rmse_day

def show_comparison(lstm_data, bp_data, svm_data):
    plt.plot(lstm_data[:, 0], 'r', label='真实值', linewidth=0.5)
    plt.plot(lstm_data[:, 1], 'b', label='mRMR-LSTM', linewidth=0.5)
    plt.plot(bp_data[:, 0], 'g', label='bp', linewidth=0.5)
    plt.plot(svm_data[:, 0], 'Purple', label='svm', linewidth=0.5)
    plt.legend(loc='best')
    plt.xlabel("时间点/h")
    plt.ylabel("负荷/kW")
    plt.show()

def main():
    lstm_data_csv = pd.read_csv('./new_data_nong_lstm_v1.csv', usecols=[0, 1]).dropna().values.astype('float32')
    bp_data_csv = pd.read_csv('./new_data_nong_bp.csv', usecols=[1]).dropna().values.astype('float32')
    svm_data_csv = pd.read_csv('./new_data_nong_svm.csv', usecols=[1]).dropna().values.astype('float32')

    lstm_min_data = evaluate_model(lstm_data_csv, 'mRMR-LSTM')
    bp_min_data = evaluate_model(bp_data_csv, 'BP')
    svm_min_data = evaluate_model(svm_data_csv, 'SVM')

    show_comparison(lstm_min_data, bp_min_data, svm_min_data)

if __name__ == '__main__':
    main()
