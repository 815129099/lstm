"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time
import sys
#画图
import matplotlib.pyplot as plt

np.random.seed(2)  # reproducible

TIME_DICT = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:1,9:2,10:2,11:2,12:1,13:1,14:1,15:1,16:1,17:1,18:1,19:2,20:2,21:2,22:1,23:1}  #峰平谷
N_STATES = 24   # 状态 t时刻
PREDICT_ACTIONS = np.array([0.29, 0.64, 0.89]) #批发电价
BEFORE_ACTIONS = np.array([0.3667, 0.6805, 0.991]) #原来电价
ACTIONS = np.round(np.linspace(0.3, 1.1, num=18), decimals=3)     # 动作，离散化   零售价
EPSILON = 0.9   # 90%取奖励最大的动作
LEARNING_RATE = 0.1     # 10%取随机动作 学习率
GAMMA = 0.9    # 奖励折扣
MAX_EPISODES = 1000000  # maximum episodes  最大回合
WEIGHT_FACTOR = 0.4  #权值因子
ALPHA = 0.01 #  用户不满意成本偏好参数
BETA = 0.01 #   用户不满意成本预设参数
D_MIN = 0.1 #  可需求响应负荷最小占比
D_MAX = 0.5 #  可需求响应负荷最大占比
K = 0.3  # 不可参与需求响应负荷占总负荷的占比
ELASTIC_COEFFICIENT = np.array([-0.25, -0.58, -0.82])  #谷、平、峰需求响应的弹性系数
ACTIONS_LIST = []  #回报最高的零售价
PARTICIPATE_POWER_LIST = [0 for _ in range(N_STATES)]  #回报最高的可参与需求响应的负荷
NO_PARTICIPATE_POWER_LIST = [0 for _ in range(N_STATES)]  #回报最高的不参与需求响应的负荷
TOTAL_POWER_LIST = [0 for _ in range(N_STATES)]  #回报最高的不参与需求响应的负荷
PREDICT_ACTIONS_LIST = [0 for _ in range(N_STATES)]
N_STATES_LIST = [0 for _ in range(N_STATES)]
dataset = [0 for _ in range(N_STATES)]
CONVERGENCE_THRESHOLD = 0.001  #判断是否收敛

#批发价、零售价、优化前负荷，优化后负荷
def show (PREDICT_ACTIONS_LIST, ACTIONS_LIST, dataset, TOTAL_POWER_LIST, N_STATES_LIST):
    # 用户总成本-优化前
    customer_total_cost = 0
    # 用户总用电量-优化前
    customer_total_power = 0
    # 售电商总利润-优化前
    total_profit = 0

    # 用户总成本-优化后
    after_customer_total_cost = 0
    # 用户总用电量-优化后
    after_customer_total_power = 0
    # 售电商总利润-优化后
    after_total_profit = 0

    for i in range(N_STATES):
        #批发价
        PREDICT_ACTIONS_LIST[i] = PREDICT_ACTIONS[TIME_DICT[i]]
        # 用户总用电量 - 优化前
        customer_total_power += dataset[i]
        # 用户总成本-优化前
        customer_total_cost += dataset[i] * BEFORE_ACTIONS[TIME_DICT[i]]
        #优化前 电价-批发价*负荷量
        total_profit += (BEFORE_ACTIONS[TIME_DICT[i]] - PREDICT_ACTIONS[TIME_DICT[i]]) * dataset[i]

        after_customer_total_power += TOTAL_POWER_LIST[i]
        after_customer_total_cost += TOTAL_POWER_LIST[i] * ACTIONS_LIST[i]
        after_total_profit += (ACTIONS_LIST[i] - PREDICT_ACTIONS_LIST[i]) * TOTAL_POWER_LIST[i]

    print("优化前负荷总量：" + str(customer_total_power))
    print("优化前用户总成本：" + str(customer_total_cost))
    print("优化前总利润：" + str(total_profit))

    print("优化后负荷总量：" + str(after_customer_total_power))
    print("优化后用户总成本：" + str(after_customer_total_cost))
    print("优化后总利润：" + str(after_total_profit))

    print("负荷总量变化：" + str(after_customer_total_power - customer_total_power))
    print("用户总成本变化：" + str(after_customer_total_cost - customer_total_cost))
    print("总利润变化：" + str(after_total_profit - total_profit))



    # 汉字字体，优先使用楷体，找不到则使用黑体
    plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

    x = np.arange(24)
    # 创建一个图形和两个y轴

    # 价格，柱状图
    bar_width = 0.35
    plt.bar(x, PREDICT_ACTIONS_LIST, bar_width, align='center', color='#66c2a5', label='批发价')
    plt.bar(x + bar_width, ACTIONS_LIST, bar_width, align='center', color='#8da0cb', label='零售价')
    plt.xlabel('时间点/h')
    plt.ylabel('价格/元')
    plt.xticks(x + bar_width / 2, N_STATES_LIST)
    # 在左侧显示图例
    plt.legend(bbox_to_anchor=(0.22, 0.88))

    ax2 = plt.twinx()
    # 折线图 负荷
    plt.plot(x, dataset, 'b', label='优化前负荷', marker='o')
    plt.plot(x, TOTAL_POWER_LIST, 'r', label='优化后负荷', marker='o')
    ax2.set_ylabel('负荷/kWh')
    plt.legend(bbox_to_anchor=(0.27, 1.0))
    plt.show()



# 0.9
def show1() :
    #批发价
    PREDICT_ACTIONS_LIST = np.array(
        [0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.64, 0.89, 0.89, 0.89, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64,
         0.64, 0.89, 0.89, 0.89, 0.64, 0.64])
    #零售价
    ACTIONS_LIST = np.array(
        [0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.734, 1.2, 1.2, 1.2, 0.734, 0.734, 0.734, 0.734, 0.734, 0.734,
         0.734, 1.2, 1.2, 1.2, 0.734, 0.734])
    #优化前的负荷
    dataset = np.array([5259.4316,5368.9204,5297.236,5102.1284,5014.8633,5236.271,5139.4067,6540.1816,9415.462,11047.648,11387.518,10667.572,9239.147,
                        11440.6045,10866.549,9850.347,10381.247,11244.644,11005.515,10164.266,10059.705,9503.475,9299.003,7698.291])
    #优化后的负荷
    TOTAL_POWER_LIST = np.array([4911.221,5013.461,4946.5225,4764.3325,4682.8447,4889.594,4799.1426,6107.177,8773.799,10370.391,10689.425,10013.614,8609.499,10660.928,10125.994,9179.045,9673.766,10478.321,10255.489,9541.162,9443.012,8920.88,8665.276,7173.6523])

    N_STATES_LIST = np.arange(24)
    show(PREDICT_ACTIONS_LIST, ACTIONS_LIST, dataset, TOTAL_POWER_LIST, N_STATES_LIST)

# 0.6
def show2() :
    #批发价
    PREDICT_ACTIONS_LIST = np.array(
        [0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.64, 0.89, 0.89, 0.89, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64,
         0.64, 0.89, 0.89, 0.89, 0.64, 0.64])
    #零售价
    ACTIONS_LIST = np.array(
        [0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.682, 1.045, 1.045, 1.045, 0.682, 0.682, 0.682, 0.682, 0.682, 0.682,
         0.682, 1.045, 1.045, 1.045, 0.682, 0.682])
    #优化前的负荷
    dataset = np.array([5259.4316,5368.9204,5297.236,5102.1284,5014.8633,5236.271,5139.4067,6540.1816,9415.462,11047.648,11387.518,10667.572,9239.147,
                        11440.6045,10866.549,9850.347,10381.247,11244.644,11005.515,10164.266,10059.705,9503.475,9299.003,7698.291])
    #优化后的负荷
    TOTAL_POWER_LIST = np.array([4954.7476,5057.893,4990.3613,4806.557,4724.3467,4932.9287,4841.676,6161.3027,8773.799,10555.098,10879.814,10191.966,8609.499,10660.928,10125.994,9179.045,9673.766,10478.321,10255.489,9711.1,9611.2,9079.77,8665.276,7173.6523])

    N_STATES_LIST = np.arange(24)
    show(PREDICT_ACTIONS_LIST, ACTIONS_LIST, dataset, TOTAL_POWER_LIST, N_STATES_LIST)

# 0.3
def show3() :
    #批发价
    PREDICT_ACTIONS_LIST = np.array([0.29,0.29,0.29,0.29,0.29,0.29,0.29,0.29,0.64,0.89,0.89,0.89,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.89,0.89,0.89,0.64,0.64])
    #零售价
    ACTIONS_LIST = np.array([0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.682,0.993,0.993,0.993,0.682,0.682,0.682,0.682,0.682,0.682,0.682,0.993,0.993,0.993,0.682,0.682])
    #优化前的负荷
    dataset = np.array([5259.4316,5368.9204,5297.236,5102.1284,5014.8633,5236.271,5139.4067,6540.1816,9415.462,11047.648,11387.518,10667.572,9239.147,
                        11440.6045,10866.549,9850.347,10381.247,11244.644,11005.515,10164.266,10059.705,9503.475,9299.003,7698.291])
    #优化后的负荷
    TOTAL_POWER_LIST = np.array([4954.7476,5057.893,4990.3613,4806.557,4724.3467,4932.9287,4841.676,6161.3027,8773.799,10515.773,10839.28,10153.994,8609.499,10660.928,10125.994,9179.045,9673.766,10478.321,10255.489,9674.919,9575.393,9045.941,8665.276,7173.6523])

    N_STATES_LIST = np.arange(24)
    show(PREDICT_ACTIONS_LIST, ACTIONS_LIST, dataset, TOTAL_POWER_LIST, N_STATES_LIST)

if __name__ == "__main__":
    #0.9
    print("---------0.9--------------")
    show1()
    # 0.6
    print("---------0.6--------------")
    show2()
    #
    print("---------0.3--------------")
    show3()

