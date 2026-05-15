import numpy as np
import gymnasium as gym
import gridworld as gw
import time


class Q_Agent():
    """由于使用包装器文件重新定义了离散动作为{0,1,2,3}，我们可以简单地用一个数字表示动作。"""

    def __init__(self, obs_dim: int, action_dim: int, epsilon: float = 0.1, lr: float = 0.1,
                 gamma: float = 0.9) -> None:
        self.obs_dim = obs_dim  # 从智能体的角度来看，状态就是观察
        self.action_dim = action_dim
        self.Q_table = np.zeros((self.obs_dim, self.action_dim))  # Q表的初始化

        self.epsilon = epsilon  # epsilon-greedy策略的探索参数
        self.lr = lr  # 学习率
        self.gamma = gamma  # 折扣因子

    def get_target_action(self, obs: int) -> int:
        """通过选择具有最高Q值的动作来确定目标策略的动作。该方法可用于测试。"""
        Q_list = self.Q_table[obs, :]
        # 注意，如果使用 action = np.argmax(Q_list)，[0,0,0,0] 将始终选择 action[0] 作为 argmax，这对于探索不好。
        # 所以我们使用以下方法。在这种方法中，[0,0,0,0] 将以随机方式选择 action[0,1,2,3]。
        action = np.random.choice(np.flatnonzero(Q_list == Q_list.max()))
        return action

    def get_behavior_action(self, obs: int) -> int:
        """对于这样一种离策略算法，我们只是为探索从目标策略修改了一个epsilon-greedy策略。"""
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.get_target_action(obs)
        return action

    def BOE_iterative_solver(self, obs: int, action: int, reward: float, next_obs: int, done: bool) -> None:
        """在这里，我们计算TD误差并使用学习率为lr的随机逼近算法更新Q表。
        因此我们称之为BOE_iterative_solver，而不仅仅是学习。"""
        current_Q = self.Q_table[obs, action]
        """注意，如果终止为True，那么将没有next_state和next_action。在这种情况下，target_Q就是reward。
        这里，我们使用一个清晰的布尔表示来避免if-else语句。"""
        TD_target = reward + (1 - float(done)) * self.gamma * self.Q_table[next_obs,
                                                              :].max()  # 与Sarsa不同，这里我们使用下一个状态的最大Q值。

        self.Q_table[obs, action] -= self.lr * (current_Q - TD_target)  # 使用学习率lr更新Q值


class TrainManager():
    def __init__(self, env: gym.Env, episode_num: int = 1000, lr: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 0.1) -> None:
        self.env = env
        self.episode_num = episode_num
        obs_dim = env.observation_space.n  # 对于这样的离散环境，我们使用env.observation_space.n来获取状态的数量
        action_dim = env.action_space.n
        self.agent = Q_Agent(
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
            action = self.agent.get_behavior_action(obs)  # 使用学到的epsilon-greedy策略获取动作
            next_state, reward, terminated, truncated, _ = self.env.step(
                action)  # 执行动作并获取下一个状态，奖励，是否终止，信息
            done = terminated or truncated
            total_reward += reward
            self.agent.BOE_iterative_solver(obs, action, reward, next_state, done)
            # 更新状态
            obs = next_state
            if is_render:
                self.env.render()
                time.sleep(0.1)

            if done:
                break

        return total_reward

    def test_episode(self) -> float:
        """在测试时，我们不需要更新Q表，所以我们只需使用目标策略获取动作。"""
        total_reward = 0
        obs, _ = self.env.reset()
        while True:
            action = self.agent.get_target_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            obs = next_obs
            total_reward += reward
            self.env.render()
            time.sleep(0.1)
            if done: break

        return total_reward

    def train(self) -> None:
        is_render = False
        for e in range(self.episode_num):  # 对于每个episode
            episode_reward = self.train_episode(is_render)
            print('Episode %s: Total Reward = %.2f' % (e, episode_reward))

            """在这里，每50个episode渲染一次环境（即使用目标策略玩游戏）以查看智能体的性能"""
            if e % 50 == 0:
                is_render = True
            else:
                is_render = False

        """训练后，我们对智能体进行一次测试"""
        test_reward = self.test_episode()
        print('Test Total Reward = %.2f' % (test_reward))


# 创建环境并进行包装
env = gym.make('CliffWalking-v1')
env = gw.CliffWalkingWapper(env)
# 创建训练管理器
Manager = TrainManager(env=env,
                       episode_num=1000,
                       lr=0.1,
                       gamma=0.9,
                       epsilon=0.1
                       )
# 进行训练
Manager.train()
