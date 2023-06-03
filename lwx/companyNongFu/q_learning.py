"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 48   # 状态
ACTIONS = np.linspace(10.0, 100.0, num=5)     # 动作，离散化
EPSILON = 0.9   # 90%取奖励最大的动作
ALPHA = 0.1     # 10%取随机动作 学习率
GAMMA = 0.9    # 奖励折扣
MAX_EPISODES = 100  # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move

#初始化q表
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    #获取该state那一行
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name

#获取当前status的奖励和下一个state
def get_env_feedback(S, A, dataset):
    # This is how agent will interact with the environment
    if S == N_STATES - 1:  # terminate
        S_ = 'terminal'
    else:
        S_ = S + 1

    #预测的用电量
    powerNum = dataset[S]
    #计算奖励
    #居民用电成本
    CUS = A*powerNum
    #售电公司的销售额

    R = CUS
    return S_, R


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


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    #预测的用电量
    dataset = load();
    for episode in range(MAX_EPISODES):
        # step_counter = 0
        S = 0
        is_terminated = False
        # update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            #获取下一个状态，计算奖励
            S_, R = get_env_feedback(S, A, dataset)  # take action & get next state and reward
            #估计值
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                #真实值
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            # update_env(S, episode, step_counter+1)
            # step_counter += 1
    return q_table

def load():
    # 读取数据
    data_csv = pd.read_csv('./Australian_electricity_data_2.csv', usecols=[6])
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
    for episode in range(MAX_EPISODES):
        # step_counter = 0
        S = 0
        is_terminated = False
        # update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            #获取下一个状态，计算奖励
            S_, R = get_env_feedback(S, A, dataset)  # take action & get next state and reward
            #估计值
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                #真实值
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            # update_env(S, episode, step_counter+1)
            # step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = test()
    print('\r\nQ-table:\n')
    print(q_table)
