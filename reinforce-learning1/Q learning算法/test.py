from maze_env import Maze   #【Maze：迷宫】
from RL_brain import QLearningTable

def update():
    for episode in range(100):
        #initial observation
        observation = env.reset()

        while True:
            #fresh env【刷新环境】
            env.render()

            #RL choose actions based on observation
            action = RL.choose_action(str(observation))
            #RL take action and get next observation and reward
            #observation_表示一个状态后的下一个状态；done：是否拿到宝藏或者跳进陷阱

            observation_,reward,done = env.step(action)
            #RL learning from this transition【过渡】       通过这4个因数学习

            RL.learn(str(observation),action,reward,str(observation_))
            #swap【交换】 observation   【把下一次的observation重新返回，连续起来】

            observation = observation_
            #break the loop when end of this episode【一集】
            if done:
                break

    #end of game
    print("game over")
    env.destroy()

if __name__ == '__main__':
    env = Maze()    #创建一个迷宫环境
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100,update)
    env.mainloop()

