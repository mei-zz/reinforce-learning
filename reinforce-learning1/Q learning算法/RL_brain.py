import numpy as np
import pandas as pd

class QLearningTable:
    """
    reward_decay:回报衰减
    greedy:贪婪
    """
    def __init__(self,actions,learning_rate=100,reward_decay=0.9,e_greedy=0.9):
        self.actions = actions  #a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self,observation):
        self.check_state_exist(observation)
        #action selection
        if np.random.uniform() < self.epsilon:
            # np.random.uniform:随机生成0-1之间的随机数，这儿用来表示概率
            #choose the best actions
            state_action = self.q_table.loc[observation,:]
            #reindex:pandas的用法：重新索引。
            # 目的：当有两个不同的索引但是对应的值相同，argmax永远只能够得到第一个的索引，所以用reindex来打乱

            #修改之处！！！！！！！！！！！！
            #state_action = state_action.reindex(np.random.permutation(state_action.index))
            #action = state_action.argmax()
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)

        else:
            #choose random acitons
            action = np.random.choice(self.actions)
        return action

    def learn(self,s,a,r,s_):
        #s:state;   a:action;   r:reward;   s_:state_
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != "terminal":
            #对比下修改之处：把原来这儿的iloc改成了loc
            q_target = r + self.gamma * self.q_table.loc[s_,:].max()   #next state
        else:
            q_target = r    #the terminal

        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)   #update


    def check_state_exist(self,state):
        if state not in self.q_table.index:
            #append new state to q_table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )