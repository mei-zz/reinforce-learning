import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible

N_STATES = 6   # the length of the 1 dimensional world【离宝藏的距离】
ACTIONS = ['left', 'right']     # available actions【两个动作】
EPSILON = 0.9   # greedy police 【90%选择reward最高的，10%选择随机的】
ALPHA = 0.1     # learning rate 【学习率】
GAMMA = 0.9    # discount factor 【对未来奖励的衰减】
MAX_EPISODES = 13   # maximum episodes  【一共进行几回合】
FRESH_TIME = 0.3    # fresh time for one move   【每次移动所消耗的时间】


# q_table initial values【初始化都为0】
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,    # actions's name    【columns：actions的内容作为表头】
    )
    # print(table)
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        # act non-greedy or state-action have no value【随机选择一个actions执行，10%的部分】
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()
        """python有.argmax()和.idxmax()两个函数，前者用于Series中，后者用于DataFrame中。
           .idxmin()+返回的是DataFrame中每一列最小值的索引
           replace argmax to idxmax as argmax means a different function in newer version of pandas
        """
    return action_name


def get_env_feedback(S, A):
    #【S表示当前的位置，A表示动作，R表示判断是不是terminal状态】
    # This is how agent will interact with the environment  【改变环境】
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
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
    for episode in range(MAX_EPISODES):
        step_counter = 0    #【计步】
        S = 0   #【当前位置】
        is_terminated = False   #【是否到终点】
        update_env(S, episode, step_counter)    #【初始化环境】
        while not is_terminated:

            A = choose_action(S, q_table)
            S_,R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)