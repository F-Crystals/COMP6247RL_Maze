import gym
import read_maze as rm
from gym import spaces
from gym.utils import seeding
# from gym.envs.classic_control import rendering
import time
from PIL import Image
import numpy as np
import cv2


MAX_STEPS_NUM = 40000
STEP_PENALTY = 1
FAIL_PENALTY = 10000
SUCCESS_REWARD = 200000
STAY_PENALTY = 20
MOVE_TO_FIRE_PENALTY = 1
MOVE_TO_WALL_PENALTY = 100
MOVE_AWAY_FROM_TARGET_PENALTY = 2
DEAD_POINT_PENALTY = 1000
VISITED_PENALTY = 100


class Env_Maze(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    # 将会初始化动作空间与状态空间，便于强化学习算法在给定的状态空间中搜索合适的动作
    # 环境中会用的全局变量可以声明为类（self.）的变量
    def __init__(self):
        self.maze_length = 201
        self.maze_width = 201
        rm.load_maze()
        self.init_maze()
        self.action_space = spaces.Discrete(5)  # 动作空间，0, 1, 2, 3, 4: 不动, left, up, right, down
        # self.observation_space = rm.maze_cells  # maze_size
        self.maze_space = np.zeros((self.maze_length, self.maze_width))
        self.observation_all_space = rm.maze_cells  # maze_size
        self.maze_dead_points = np.zeros((self.maze_length, self.maze_width))
        self.n_actions = self.action_space.n  # 动作个数
        self.state = [1, 1]  # 当前状态
        self.target = [199, 199]  # 安全/目标状态 final point of maze
        self.get_dead_point_of_maze()

        # self.viewer = rendering.Viewer(1000, 1000)  # 初始化一张画布

    def step(self, action):
        # 接收一个动作，执行这个动作
        # 用来处理状态的转换逻辑
        # 返回动作的回报、下一时刻的状态、以及是否结束当前episode及调试信息
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        x = self.state
        self.maze_space[x[0], x[1]] = 1
        around = rm.get_local_maze_information(x[0], x[1])
        reward = 0
        if action == 0:  # 不动
            x = x
            reward -= STAY_PENALTY
        elif action == 1:
            if self.observation_all_space[x[0] - 1, x[1], 0]:
                if not self.observation_all_space[x[0] - 1, x[1], 1]:  # 左
                    x[0] = x[0] - 1
                    reward -= MOVE_AWAY_FROM_TARGET_PENALTY  # 让ai有向右走的倾向
                    if self.maze_space[x[0] - 1, x[1]]:
                        reward -= VISITED_PENALTY  # 让ai避免走走过的路
                else:
                    reward -= MOVE_TO_FIRE_PENALTY
            else:
                reward -= MOVE_TO_WALL_PENALTY  # 让ai避免选择撞墙的action
        elif action == 2:
            if self.observation_all_space[x[0], x[1] - 1, 0]:
                if not self.observation_all_space[x[0], x[1] - 1, 1]:  # 左
                    x[1] = x[1] - 1
                    reward -= MOVE_AWAY_FROM_TARGET_PENALTY  # 让ai有向下走的倾向
                    if self.maze_space[x[0], x[1] - 1]:
                        reward -= VISITED_PENALTY  # 让ai避免走走过的路
                else:
                    reward -= MOVE_TO_FIRE_PENALTY
            else:
                reward -= MOVE_TO_WALL_PENALTY  # 让ai避免选择撞墙的action
        elif action == 3:
            if self.observation_all_space[x[0] + 1, x[1], 0]:
                if not self.observation_all_space[x[0] + 1, x[1], 1]:  # 左
                    x[0] = x[0] + 1
                    if self.maze_space[x[0] + 1, x[1]]:
                        reward -= VISITED_PENALTY  # 让ai避免走走过的路
                else:
                    reward -= MOVE_TO_FIRE_PENALTY
            else:
                reward -= MOVE_TO_WALL_PENALTY  # 让ai避免选择撞墙的action
        elif action == 4:
            if self.observation_all_space[x[0], x[1] + 1, 0]:
                if not self.observation_all_space[x[0], x[1] + 1, 1]:  # 左
                    x[1] = x[1] + 1
                    if self.maze_space[x[0], x[1] + 1]:
                        reward -= VISITED_PENALTY  # 让ai避免走走过的路
                else:
                    reward -= MOVE_TO_FIRE_PENALTY
            else:
                reward -= MOVE_TO_WALL_PENALTY  # 让ai避免选择撞墙的action

        # 在这里做一下限定，如果下一个动作导致智能体越过了环境边界（即不在状态空间中），则无视这个动作
        next_state = x
        self.state = next_state
        self.counts += 1

        # 如果到达了终点，给予一个回报
        # 在复杂环境中多种状态的反馈配比很重要
        if self.state == self.target:
            reward += SUCCESS_REWARD
            done = True
        else:  # 如果是普通的一步，给予一个小惩罚，目的是为了减少起点到终点的总路程长度
            if self.maze_dead_points[self.state[0], self.state[1]]:
                reward -= DEAD_POINT_PENALTY
                done = False
            else:
                reward -= STEP_PENALTY
                done = False

        if self.counts > MAX_STEPS_NUM:
            reward -= FAIL_PENALTY
            done = True

        return self.state, reward, done

    # 用于在每轮开始之前重置智能体的状态，把环境恢复到最开始
    # 在训练的时候，可以不指定startstate，随机选择初始状态，以便能尽可能全的采集到的环境中所有状态的数据反馈
    def reset(self):
        # if startstate == None:
        #     self.state_new = self.observation_space.sample()
        # else:  # 在训练完成测试的时候，可以根据需要指定从某个状态开始
            # if self.observation_space.contains(startstate):
            #     self.state_new = startstate
            # else:
            #     self.state_new = self.observation_space.sample()
        self.state = [1, 1]
        self.counts = 0
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # openCV画图
    def init_maze(self):
        self.maze = np.zeros((201, 201, 3), dtype=np.uint8)
        for i in range(rm.maze_cells.shape[0]):
            for j in range(rm.maze_cells.shape[1]):
                if rm.maze_cells[i][j][0]:
                    self.maze[i][j] = (255, 255, 255)
                else:
                    self.maze[i][j] = (0, 0, 0)

    # render()绘制可视化环境的部分都写在这里
    def render(self, mode='human'):

        ai = self.state
        for i in range(rm.maze_cells.shape[0]):
            for j in range(rm.maze_cells.shape[1]):
                if rm.maze_cells[i, j, 0]:
                    if rm.maze_cells[i, j, 1]:
                        self.maze[i, j] = (0, 0, 255)
                    else:
                        self.maze[i][j] = (255, 255, 255)
                else:
                    self.maze[i][j] = (0, 0, 0)
        self.maze[1, 1] = (255, 128, 128)
        self.maze[199, 199] = (0, 128, 255)
        self.maze[ai[0], ai[1]] = (255, 255, 0)

        img = Image.fromarray(self.maze, 'RGB')
        cv2.namedWindow("Maze", 0)
        cv2.resizeWindow("Maze", 1000, 1000)
        cv2.imshow("Maze", np.array(img))
        cv2.waitKey(1)
        self.maze[ai[0], ai[1]] = (255, 255, 255)
        return self.maze

    def get_dead_point_of_maze(self):
        count = 0
        for i in range(1, rm.maze_cells.shape[0] - 1):
            for j in range(1, rm.maze_cells.shape[1] - 1):
                if rm.maze_cells[i + 1, j, 0]:
                    count += 1
                elif rm.maze_cells[i - 1, j, 0]:
                    count += 1
                elif rm.maze_cells[i, j + 1, 0]:
                    count += 1
                elif rm.maze_cells[i, j - 1, 0]:
                    count += 1
                if not count > 1:
                    self.maze_dead_points[i, j] = 1

if __name__ == '__main__':
    env = Env_Maze()
    for epoch in range(5):
        env.reset()
        print('Epoch', epoch + 1, ': ', end='')
        print(env.state, end='')
        # env.render()  # 刷新画面

        for i in range(50):

            env.step(env.action_space.sample())  # 随机选择一个动作执行

            env.render()
            print(' -> ', env.state, end='')
            # env.render()  # 刷新画面
            time.sleep(0.2)
        print()
    env.close()
