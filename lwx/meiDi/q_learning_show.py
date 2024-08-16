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
BEFORE_ACTIONS = np.array([0.3647, 0.6805, 1.0911]) #原来电价
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
        [0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.631, 0.838, 1.096, 1.096, 1.096, 0.838, 0.838, 0.838, 0.838, 0.838, 0.838,
         0.838, 1.096, 1.096, 1.096, 0.838, 0.838])
    #优化前的负荷
    dataset = np.array([5259.4316,5368.9204,5297.236,5102.1284,5014.8633,5236.271,5139.4067,6540.1816,9415.462,11047.648,11387.518,10667.572,9239.147,
                        11440.6045,10866.549,9850.347,10381.247,11244.644,11005.515,10164.266,10059.705,9503.475,9299.003,7698.291])
    #优化后的负荷
    TOTAL_POWER_LIST = np.array([4022.5586,4106.2983,4051.4722,3902.2488,3835.5059,4004.8447,3930.76,5236.6562,8063.872,9370.193,10086.526,9047.827,7912.8677,9798.306,9306.656,8436.329,8891.02,9630.475,9425.674,8620.942,8532.258,8060.484,7964.1313,6308.557])

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
        [0.527, 0.527, 0.527, 0.527, 0.527, 0.527, 0.527, 0.527, 0.786, 1.045, 1.045, 1.045, 0.786, 0.786, 0.786, 0.786, 0.786, 0.786,
         0.786, 1.045, 1.045, 1.045, 0.786, 0.786])
    #优化前的负荷
    dataset = np.array([5259.4316,5368.9204,5297.236,5102.1284,5014.8633,5236.271,5139.4067,6540.1816,9415.462,11047.648,11387.518,10667.572,9239.147,
                        11440.6045,10866.549,9850.347,10381.247,11244.644,11005.515,10164.266,10059.705,9503.475,9299.003,7698.291])
    #优化后的负荷
    TOTAL_POWER_LIST = np.array([4399.7866,4491.3794,4431.4116,4268.1943,4015.349,4380.4116,4299.3794,5471.2,8418.836,9785.486,10086.526,9448.832,8261.184,10229.616,9716.325,8807.6875,9282.393,10054.398,9840.582,9003.027,8910.412,8417.7295,8314.704,6593.201])

    N_STATES_LIST = np.arange(24)
    show(PREDICT_ACTIONS_LIST, ACTIONS_LIST, dataset, TOTAL_POWER_LIST, N_STATES_LIST)

# 0.3
def show3() :
    #批发价
    PREDICT_ACTIONS_LIST = np.array([0.29,0.29,0.29,0.29,0.29,0.29,0.29,0.29,0.64,0.89,0.89,0.89,0.64,0.64,0.64,0.64,0.64,0.64,0.64,0.89,0.89,0.89,0.64,0.64])
    #零售价
    ACTIONS_LIST = np.array([0.475,0.475,0.475,0.475,0.475,0.475,0.475,0.475,0.734,0.993,0.993,0.993,0.734,0.734,0.734,0.734,0.734,0.734,0.734,0.993,0.993,0.993,0.734,0.734])
    #优化前的负荷
    dataset = np.array([5259.4316,5368.9204,5297.236,5102.1284,5014.8633,5236.271,5139.4067,6540.1816,9415.462,11047.648,11387.518,10667.572,9239.147,
                        11440.6045,10866.549,9850.347,10381.247,11244.644,11005.515,10164.266,10059.705,9503.475,9299.003,7698.291])
    #优化后的负荷
    TOTAL_POWER_LIST = np.array([4588.401,4683.92,4621.3813,4451.167,4375.036,4568.1953,4483.6895,5705.745,8773.799,10208.922,10522.988,9857.699,8609.499,10660.928,10125.994,9179.045,9673.766,10478.321,10255.489,9392.604,9295.981,8781.9795,8665.276,6883.4272])

    N_STATES_LIST = np.arange(24)
    show(PREDICT_ACTIONS_LIST, ACTIONS_LIST, dataset, TOTAL_POWER_LIST, N_STATES_LIST)

if __name__ == "__main__":
    #0.9
    show1()
    # 0.6
    # show2()
    #
    # show3()

