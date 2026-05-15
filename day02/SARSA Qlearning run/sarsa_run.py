import numpy as np
import gymnasium as gym
import gridworld as gw
import time


class Sarsa_Agent():
    """ 由于使用包装器文件重新定义了离散动作为{0,1,2,3}，我们可以简单地用一个数字表示动作。"""

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 epsilon: float = 0.1,
                 lr: float = 0.1,
                 gamma: float = 0.9) -> None:
        # 初始化Sarsa_Agent对象
        # obs_dim: 状态空间的维度
        # action_dim: 动作空间的维度
        # epsilon: epsilon-greedy策略中的探索率
        # lr: 学习率
        # gamma: 折扣因子
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.Q = np.zeros((self.obs_dim, self.action_dim))  # 初始化状态动作值函数 Q 表
        print(self.Q)

        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma

    def get_greedy_action(self, obs: int) -> int:
        # 通过策略改进确定策略的动作
        Q_list = self.Q[obs, :]
        # 选择动作，使用贪婪策略（选择具有最大 Q 值的动作）
        # 注意：如果有多个最大 Q 值，则随机选择其中一个
        action = np.random.choice(
            np.flatnonzero(Q_list == Q_list.max()))  # 对于这种方法，[0,0,0,0]将以随机方式选择action[0,1,2,3]
        return action

    def get_action(self, obs: int) -> int:
        # epsilon-greedy策略
        if np.random.uniform(0, 1) < self.epsilon:
            # 以 epsilon 的概率随机选择动作
            action = np.random.choice(self.action_dim)
        else:
            # 以 (1 - epsilon) 的概率使用改进的策略（贪婪策略）
            action = self.get_greedy_action(obs)

        return action

    def policy_evaluation(self,
                          obs: int,
                          action: int,
                          reward: float,
                          next_obs: int,
                          next_action: int,
                          done: bool) -> None:
        # 策略评估更新 Q 表
        current_Q = self.Q[obs, action]
        # 计算 TD 目标(时序差分)
        TD_target = reward + (1 - float(done)) * self.gamma * self.Q[next_obs, next_action]
        # 使用 Sarsa 更新规则更新 Q 值
        self.Q[obs, action] -= self.lr * (current_Q - TD_target)


class TrainManager():

    def __init__(self,
                 env: gym.Env,
                 episode_num: int = 1000,
                 lr: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1) -> None:
        self.env = env
        self.episode_num = episode_num
        obs_dim = env.observation_space.n
        print('obs_dim', obs_dim)
        action_dim = env.action_space.n
        print('action_dim', action_dim)
        self.agent = Sarsa_Agent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            epsilon=epsilon,
            lr=lr,
            gamma=gamma
        )

    def train_episode(self, is_render: bool = False) -> float:
        total_reward = 0  # 记录一个episode的总奖励
        obs, _ = self.env.reset()  # 重置环境并获取初始状态
        while True:
            action = self.agent.get_action(obs)  # 使用学到的epsilon-greedy策略获取动作
            next_obs, reward, terminated, truncated, _ = self.env.step(
                action)  # 执行动作并获取下一个状态，奖励，是否终止，信息
            """在Gymnasium或Gym v0.26中，当terminated或truncated时，done为True。
               在Gym早期版本中，有done但没有terminated，truncated在这里。
               根据你使用的Gym版本，你应该修改代码。"""
            done = terminated or truncated
            total_reward += reward
            # 对于Sarsa，我们需要使用当前策略获得a'
            next_action = self.agent.get_action(next_obs)
            # 使用数据进行策略评估
            self.agent.policy_evaluation(obs, action, reward, next_obs, next_action, done)
            # 更新状态和动作
            obs = next_obs
            if is_render:
                self.env.render()  # !! 你可以在任务栏中找到游戏窗口 !!
                time.sleep(0.1)  # 为可视化设置速度

            print(11111)

            if done:
                break
            print(22222)

        return total_reward

    def test_episode(self) -> float:
        total_reward = 0  # 记录一个episode的总奖励
        obs, _ = self.env.reset()  # 重置环境并获取初始状态
        while True:
            action = self.agent.get_greedy_action(obs)  # 使用学到的贪婪策略获取动作
            next_obs, reward, terminated, truncated, _ = self.env.step(
                action)  # 执行动作并获取下一个状态，奖励，是否终止，信息
            done = terminated or truncated
            obs = next_obs
            total_reward += reward
            self.env.render()
            time.sleep(0.1)
            if done:
                break

        return total_reward

    def train(self) -> None:
        is_render = True
        for e in range(self.episode_num):
            episode_reward = self.train_episode(is_render)
            print('Episode %s: Total Reward = %.2f' % (e, episode_reward))

            if e % 50 == 0:
                is_render = True
            else:
                is_render = False

        test_reward = self.test_episode()
        print('Test Total Reward = %.2f' % (test_reward))


# env = gym.make('CliffWalking-v0')
# env = gw.CliffWalkingWapper(env)

"""这是另一个你可以尝试的游戏。"""

env = gym.make("FrozenLake-v1", is_slippery=False)
env = gw.FrozenLakeWapper(env)

Manager = TrainManager(env=env,
                       episode_num=500,
                       lr=0.1,
                       gamma=0.9,
                       epsilon=0.1
                       )
Manager.train()
