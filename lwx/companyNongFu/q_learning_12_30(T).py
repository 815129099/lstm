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

TIME_DICT = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:1,9:1,10:1,11:1,12:1,13:1,14:2,15:2,16:2,17:1,18:1,19:2,20:2,21:2,22:1,23:1}  #峰平谷
N_STATES = 24   # 状态 t时刻
PREDICT_ACTIONS = np.array([0.22, 0.53, 0.89]) #批发电价
ACTIONS = np.round(np.linspace(0.3, 1.1, num=12), decimals=3)     # 动作，离散化   零售价
EPSILON = 0.9   # 90%取奖励最大的动作
LEARNING_RATE = 0.1     # 10%取随机动作 学习率
GAMMA = 0.9    # 奖励折扣
MAX_EPISODES = 1000  # maximum episodes  最大回合
FRESH_TIME = 0.3    # fresh time for one move  移动间隔
WEIGHT_FACTOR = 0.8  #权值因子
ALPHA = 0.01 #  用户不满意成本偏好参数
BETA = 0.01 #   用户不满意成本预设参数
D_MIN = 0.1 #  可需求响应负荷最小占比
D_MAX = 0.5 #  可需求响应负荷最大占比
K = 0.3  # 不可参与需求响应负荷占总负荷的占比
ELASTIC_COEFFICIENT = np.array([-0.25, -0.53, -0.82])  #谷、平、峰需求响应的弹性系数
ACTIONS_LIST = []  #回报最高的零售价
PARTICIPATE_POWER_LIST = [0 for _ in range(N_STATES)]  #回报最高的可参与需求响应的负荷
NO_PARTICIPATE_POWER_LIST = [0 for _ in range(N_STATES)]  #回报最高的不参与需求响应的负荷
TOTAL_POWER_LIST = [0 for _ in range(N_STATES)]  #回报最高的不参与需求响应的负荷
PREDICT_ACTIONS_LIST = [0 for _ in range(N_STATES)]
N_STATES_LIST = [0 for _ in range(N_STATES)]
dataset = [0 for _ in range(N_STATES)]
CONVERGENCE_THRESHOLD = 0.001  #判断是否收敛


#初始化q表
#t时刻，零售电价
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    print(table)    # show table
    print("---------------------------------table  end---------------------------------------------")
    return table

# 选择回报最大的action
def choose_action(S, q_table,dataset):
    # This is how to choose an action
    #获取该state那一行
    state_actions = q_table.iloc[S, :]
    if (np.random.uniform() > EPSILON):  # act non-greedy or state-action have no value
        #随机选择
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        #取最大的
        mix_r = -sys.maxsize-1
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
        for i in range(len(ACTIONS)):
            S_, R = get_env_feedback(S, ACTIONS[i], dataset, 0)
            if (mix_r < R):
                mix_r = R
                action_name = ACTIONS[i]
    return action_name

# S指时刻，A->S时刻时电价，dataset->负荷集合
#获取当前status的奖励和下一个state
def get_env_feedback(S, A, dataset, type):
    # This is how agent will interact with the environment
    if S == N_STATES - 1:  # terminate
        S_ = 'terminal'
    else:
        S_ = S + 1

    #S时刻处于峰平谷
    flag = TIME_DICT[S]
    # 获取批发电价
    wholesale_price = PREDICT_ACTIONS[flag]

    if wholesale_price > A :
        return S_, -sys.maxsize-1

    #预测的用电量
    predict_power = dataset[S]
    #可参与需求响应负荷- 预计
    predict_participate_power = predict_power*(1-K)
    predict_no_participate_power = predict_power*K

    #可参与需求响应负荷- 实际
    actual_participate_power = predict_participate_power*(1+ELASTIC_COEFFICIENT[flag]*((A-wholesale_price)/wholesale_price))
    actual_no_participate_power = predict_no_participate_power
    actual_power = actual_participate_power+actual_no_participate_power

    #不满意成本
    unsatisfied_cost = ALPHA*(predict_participate_power - actual_participate_power)*(predict_participate_power - actual_participate_power) + BETA*(predict_participate_power - actual_participate_power)

    #售电商利润
    profit = (A - wholesale_price)*actual_power

    #居民用电成本
    total_cost = A*actual_power +unsatisfied_cost

    #WEIGHT_FACTOR 权值因子
    #总回报
    R = WEIGHT_FACTOR * profit -(1-WEIGHT_FACTOR) * total_cost

    if (type == 1) :
        PARTICIPATE_POWER_LIST[S] = actual_participate_power
        NO_PARTICIPATE_POWER_LIST[S] = actual_no_participate_power
        TOTAL_POWER_LIST[S] = actual_no_participate_power+actual_participate_power
    return S_, R

#更新环境
def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def load():
    # 读取数据
    data_csv = pd.read_csv('./new_data_nong_lstm_24_01_07.csv', usecols=[3])
    # 去除读入的数据中含有NaN的行。
    data_csv = data_csv.dropna()
    dataset = data_csv.values
    # 对数据类型进行转换
    dataset = dataset.astype('float32')
    return dataset

def test():
    # 初始化Q值表
    q_table = build_q_table(N_STATES, ACTIONS)
    #预测的用电量
    dataset = load();
    #开始循环
    prev_q_table = q_table.copy()  # 复制初始的Q值表
    for episode in range(MAX_EPISODES):
        # step_counter = 0
        S = 0
        is_terminated = False
        # update_env(S, episode, step_counter)
        while not is_terminated:
            #获取动作
            A = choose_action(S, q_table,dataset)
            #获取下一个状态，计算奖励
            S_, R = get_env_feedback(S, A, dataset, 1)  # take action & get next state and reward
            #估计值
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                #未结束
                #真实值
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += LEARNING_RATE * (q_target - q_predict)  # update
            S = S_  # move to next state
            #更新环境
            # update_env(S, episode, step_counter+1)
            # step_counter += 1

        # 判断是否收敛
        # 在Q值更新后，检查是否收敛
        if np.max(np.abs(q_table - prev_q_table).values) < CONVERGENCE_THRESHOLD:
            print(f"Q-values have converged. Stopping training at episode {episode + 1}.")
            break

        prev_q_table = q_table.copy()  # 更新前的Q值表

    #打印回报最大的价格
    ACTIONS_LIST = q_table.idxmax(axis=1)
    # print(ACTIONS_LIST)
    # print(PARTICIPATE_POWER_LIST)
    # print(NO_PARTICIPATE_POWER_LIST)
    print("-------------------最合适零售价--------------------------")
    #用户总成本-优化前
    customer_total_cost = 0
    #用户总用电量-优化前
    customer_total_power = 0
    #售电商总利润-优化前
    total_profit = 0

    #用户总成本-优化后
    after_customer_total_cost = 0
    #用户总用电量-优化后
    after_customer_total_power = 0
    #售电商总利润-优化后
    after_total_profit = 0

    for i in range(N_STATES) :
        N_STATES_LIST[i] = i
        PREDICT_ACTIONS_LIST[i] = PREDICT_ACTIONS[TIME_DICT[i]]
        customer_total_power += dataset[i]
        after_customer_total_power += TOTAL_POWER_LIST[i]
        after_customer_total_cost += TOTAL_POWER_LIST[i]*ACTIONS_LIST[i]

        print("t:"+str(i)+
              "， 批发价"+str(PREDICT_ACTIONS_LIST[i])+
              ",  零售价:"+str(ACTIONS_LIST[i])+
              ",  预测电量:"+str(dataset[i])+
              "  实际电量："+str(TOTAL_POWER_LIST[i])+
              "  可参与负荷："+str(PARTICIPATE_POWER_LIST[i])+
              "  不可参与负荷："+str(NO_PARTICIPATE_POWER_LIST[i]))

    print("优化前负荷总量："+str(customer_total_power))
    print("优化后负荷总量："+str(after_customer_total_power))
    print("优化后总利润："+str(after_customer_total_cost))

    # 汉字字体，优先使用楷体，找不到则使用黑体
    plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

    x = np.arange(24)
    # 创建一个图形和两个y轴


    #价格，柱状图
    bar_width = 0.35
    plt.bar(x, PREDICT_ACTIONS_LIST, bar_width, align='center', color='#66c2a5', label='批发价')
    plt.bar(x + bar_width, ACTIONS_LIST, bar_width, align='center', color='#8da0cb', label='零售价')
    plt.xlabel('时间点/h')
    plt.ylabel('价格/元')
    plt.xticks(x + bar_width / 2, N_STATES_LIST)
    # 在左侧显示图例
    plt.legend(loc="upper left")


    ax2 = plt.twinx()
    # 折线图 负荷
    plt.plot(x, dataset, 'b', label='优化前的负荷', marker='o')
    plt.plot(x, TOTAL_POWER_LIST, 'r', label='优化后的负荷', marker='o')
    ax2.set_ylabel('负荷/kWh')
    plt.legend(loc="upper right")
    plt.show()

    return q_table

def show():
    dataset = load();
    # 汉字字体，优先使用楷体，找不到则使用黑体
    plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

    x = np.arange(24)
    # 创建一个图形和两个y轴

    # 折线图 负荷
    plt.plot(x, dataset, 'b', marker='o',label='负荷')
    plt.xlabel('时间点/h')
    plt.ylabel('负荷/kWh')
    # plt.legend(loc="upper center")
    plt.show()


if __name__ == "__main__":
    # q_table = test()
    # 单日负荷
    show()

