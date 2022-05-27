import gym
import read_maze as rm
from gym import spaces
from gym.utils import seeding
# from gym.envs.classic_control import rendering
import time
from PIL import Image
import numpy as np
import cv2

MAX_STEPS_NUM = 10000               # game over if steps larger than 10000
STEP_PENALTY = 0.1                  # every step costs in order to minimize the path, the time consumption
FAIL_PENALTY = 100                  # negative reward if fail
SUCCESS_REWARD = 20000              # big positive reward if win
STAY_PENALTY = 0.5                  # expect agent not to stay
# MOVE_TO_FIRE_PENALTY = 1          # at first I thought if agent see the fire and still move to the fire,
                                    # it should be penalized. However, I changed my idea after,
                                    # because I thought this may cause agent learn the wrong logic.
MOVE_TO_WALL_PENALTY = 1            # expect agent not to move to the wall
MOVE_TO_TARGET_REWARD = 0.2         # samll positive reward if move towards the target
DEAD_POINT_PENALTY = 10             # expect agent not to stay and frequently visit the dead point
VISITED_PENALTY = 1                 # expect agent to try others ways to the target


# define my environment
class Env_Maze(gym.Env):
    # initialization
    def __init__(self):
        # load maze for only once
        rm.load_maze()
        self.maze_length = 201
        self.maze_width = 201
        self.maze_space = np.zeros((self.maze_length, self.maze_width))
        self.observation_all_space = rm.maze_cells  # this is used to make judgement, is or isn't wall, fire, etc.
        self.action_space = spaces.Discrete(5)  # action space，0, 1, 2, 3, 4: stay, left, up, right, down
        self.n_actions = self.action_space.n  # the number of actions
        self.init_maze()

        self.state = [1, 1]  # initial location
        self.target = [199, 199]  # target location of maze
        # get dead points in the maze
        self.maze_dead_points = np.zeros((self.maze_length, self.maze_width))
        self.get_dead_point_of_maze()

        # used to print and analyse the trace information
        self.stopped_by_fire = False
        self.stopped_by_wall = False

    def step(self, action):
        # get the reward based on the type of action and conditions for this action, and get new environment
        # judge whether the action get to the end
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        x = self.state

        # set visited point
        self.maze_space[x[0], x[1]] = 1

        reward = 0
        # stay
        if action == 0:
            x = x
            reward -= STAY_PENALTY
            self.stopped_by_fire = False
            self.stopped_by_wall = False
        # left
        elif action == 1:
            if self.observation_all_space[x[0] - 1, x[1], 0]:
                self.stopped_by_wall = False
                if not self.observation_all_space[x[0] - 1, x[1], 1]:
                    x[0] = x[0] - 1
                    self.stopped_by_fire = False
                    if self.maze_space[x[0] - 1, x[1]]:
                        reward -= VISITED_PENALTY  # avoid visiting the visited point so much times
                else:
                    self.stopped_by_fire = True
                    # reward -= MOVE_TO_FIRE_PENALTY
            else:
                self.stopped_by_wall = True
                reward -= MOVE_TO_WALL_PENALTY  # avoid moving to the wall
        # up
        elif action == 2:
            if self.observation_all_space[x[0], x[1] - 1, 0]:
                self.stopped_by_wall = False
                if not self.observation_all_space[x[0], x[1] - 1, 1]:
                    x[1] = x[1] - 1
                    self.stopped_by_fire = False
                    if self.maze_space[x[0], x[1] - 1]:
                        reward -= VISITED_PENALTY  # avoid visiting the visited point so much times
                else:
                    self.stopped_by_fire = True
                    # reward -= MOVE_TO_FIRE_PENALTY
            else:
                self.stopped_by_wall = True
                reward -= MOVE_TO_WALL_PENALTY  # avoid moving to the wall
        # right
        elif action == 3:
            if self.observation_all_space[x[0] + 1, x[1], 0]:
                self.stopped_by_wall = False
                if not self.observation_all_space[x[0] + 1, x[1], 1]:
                    x[0] = x[0] + 1
                    reward += MOVE_TO_TARGET_REWARD  # expect agent go to the target faster (e.g. right)
                    self.stopped_by_fire = False
                    if self.maze_space[x[0] + 1, x[1]]:
                        reward -= VISITED_PENALTY
                else:
                    self.stopped_by_fire = True
                    # reward -= MOVE_TO_FIRE_PENALTY
            else:
                self.stopped_by_wall = True
                reward -= MOVE_TO_WALL_PENALTY  # avoid moving to the wall
        # down
        elif action == 4:
            if self.observation_all_space[x[0], x[1] + 1, 0]:
                self.stopped_by_wall = False
                if not self.observation_all_space[x[0], x[1] + 1, 1]:
                    x[1] = x[1] + 1
                    self.maze_space[x[0] - 1, x[1]] = 1
                    reward += MOVE_TO_TARGET_REWARD  # expect agent go to the target faster (e.g. downward)
                    self.stopped_by_fire = False
                    if self.maze_space[x[0], x[1] + 1]:
                        reward -= VISITED_PENALTY
                else:
                    self.stopped_by_fire = True
                    # reward -= MOVE_TO_FIRE_PENALTY
            else:
                self.stopped_by_wall = True
                reward -= MOVE_TO_WALL_PENALTY  # avoid moving to the wall

        # assign new location to next state
        next_state = x
        self.state = next_state
        self.counts += 1

        reward -= STEP_PENALTY

        # positive reward if success
        if self.state == self.target:
            reward += SUCCESS_REWARD
            done = True
        # for each step, small penalty to make agent use less steps to get to the target point
        else:
            if self.maze_dead_points[self.state[0], self.state[1]]:
                reward -= DEAD_POINT_PENALTY
            done = False
        # if steps larger than MAX_STEPS_NUM, the agent failed in this episode.
        if self.counts > MAX_STEPS_NUM:
            reward -= FAIL_PENALTY
            done = True

        # get new environment after this action
        around = rm.get_local_maze_information(x[0], x[1])

        return self.state, reward, done

    # reset the maze game, all parameters, environments, etc.
    def reset(self):
        self.state = [1, 1]
        self.counts = 0
        return self.state

    # openCV to draw the maze
    def init_maze(self):
        self.maze = np.zeros((201, 201, 3), dtype=np.uint8)
        for i in range(rm.maze_cells.shape[0]):
            for j in range(rm.maze_cells.shape[1]):
                if rm.maze_cells[i][j][0]:
                    self.maze[i][j] = (255, 255, 255)
                else:
                    self.maze[i][j] = (0, 0, 0)

    # render() visualization of environment
    def render(self, mode='human'):
        # get current state
        agent = self.state
        for i in range(rm.maze_cells.shape[0]):
            for j in range(rm.maze_cells.shape[1]):
                if rm.maze_cells[i, j, 0]:
                    if rm.maze_cells[i, j, 1]:
                        # draw red fire
                        self.maze[i, j] = (0, 0, 255)
                    else:
                        # change to white if no fire
                        self.maze[i][j] = (255, 255, 255)
                else:
                    # black wall
                    self.maze[i][j] = (0, 0, 0)
        # start point
        self.maze[1, 1] = (255, 128, 128)
        # target point
        self.maze[199, 199] = (0, 128, 255)
        # agent
        self.maze[agent[0], agent[1]] = (255, 255, 0)

        img = Image.fromarray(self.maze, 'RGB')
        cv2.namedWindow("Maze", 0)
        cv2.resizeWindow("Maze", 1000, 1000)
        cv2.imshow("Maze", np.array(img))
        cv2.waitKey(1)
        # clear the last color, keep one agent color to show where the agent is
        self.maze[agent[0], agent[1]] = (255, 255, 255)
        return self.maze

    # get the dead point
    # the location is the dead point when there is only one way to go
    def get_dead_point_of_maze(self):
        for i in range(1, rm.maze_cells.shape[0] - 1):
            for j in range(1, rm.maze_cells.shape[1] - 1):
                count = 0
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
import gym
import read_maze as rm
from gym import spaces
from gym.utils import seeding
# from gym.envs.classic_control import rendering
import time
from PIL import Image
import numpy as np
import cv2

MAX_STEPS_NUM = 10000               # game over if steps larger than 10000
STEP_PENALTY = 0.1                  # every step costs in order to minimize the path, the time consumption
FAIL_PENALTY = 100                  # negative reward if fail
SUCCESS_REWARD = 20000              # big positive reward if win
STAY_PENALTY = 0.5                  # expect agent not to stay
# MOVE_TO_FIRE_PENALTY = 1          # at first I thought if agent see the fire and still move to the fire,
                                    # it should be penalized. However, I changed my idea after,
                                    # because I thought this may cause agent learn the wrong logic.
MOVE_TO_WALL_PENALTY = 1            # expect agent not to move to the wall
MOVE_TO_TARGET_REWARD = 0.2         # samll positive reward if move towards the target
DEAD_POINT_PENALTY = 10             # expect agent not to stay and frequently visit the dead point
VISITED_PENALTY = 1                 # expect agent to try others ways to the target


# define my environment
class Env_Maze(gym.Env):
    # initialization
    def __init__(self):
        # load maze for only once
        rm.load_maze()
        self.maze_length = 201
        self.maze_width = 201
        self.maze_space = np.zeros((self.maze_length, self.maze_width))
        self.observation_all_space = rm.maze_cells  # this is used to make judgement, is or isn't wall, fire, etc.
        self.action_space = spaces.Discrete(5)  # action space，0, 1, 2, 3, 4: stay, left, up, right, down
        self.n_actions = self.action_space.n  # the number of actions
        self.init_maze()

        self.state = [1, 1]  # initial location
        self.target = [199, 199]  # target location of maze
        # get dead points in the maze
        self.maze_dead_points = np.zeros((self.maze_length, self.maze_width))
        self.get_dead_point_of_maze()

        # used to print and analyse the trace information
        self.stopped_by_fire = False
        self.stopped_by_wall = False

    def step(self, action):
        # get the reward based on the type of action and conditions for this action, and get new environment
        # judge whether the action get to the end
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        x = self.state

        # set visited point
        self.maze_space[x[0], x[1]] = 1

        reward = 0
        # stay
        if action == 0:
            x = x
            reward -= STAY_PENALTY
            self.stopped_by_fire = False
            self.stopped_by_wall = False
        # left
        elif action == 1:
            if self.observation_all_space[x[0] - 1, x[1], 0]:
                self.stopped_by_wall = False
                if not self.observation_all_space[x[0] - 1, x[1], 1]:
                    x[0] = x[0] - 1
                    self.stopped_by_fire = False
                    if self.maze_space[x[0] - 1, x[1]]:
                        reward -= VISITED_PENALTY  # avoid visiting the visited point so much times
                else:
                    self.stopped_by_fire = True
                    # reward -= MOVE_TO_FIRE_PENALTY
            else:
                self.stopped_by_wall = True
                reward -= MOVE_TO_WALL_PENALTY  # avoid moving to the wall
        # up
        elif action == 2:
            if self.observation_all_space[x[0], x[1] - 1, 0]:
                self.stopped_by_wall = False
                if not self.observation_all_space[x[0], x[1] - 1, 1]:
                    x[1] = x[1] - 1
                    self.stopped_by_fire = False
                    if self.maze_space[x[0], x[1] - 1]:
                        reward -= VISITED_PENALTY  # avoid visiting the visited point so much times
                else:
                    self.stopped_by_fire = True
                    # reward -= MOVE_TO_FIRE_PENALTY
            else:
                self.stopped_by_wall = True
                reward -= MOVE_TO_WALL_PENALTY  # avoid moving to the wall
        # right
        elif action == 3:
            if self.observation_all_space[x[0] + 1, x[1], 0]:
                self.stopped_by_wall = False
                if not self.observation_all_space[x[0] + 1, x[1], 1]:
                    x[0] = x[0] + 1
                    reward += MOVE_TO_TARGET_REWARD  # expect agent go to the target faster (e.g. right)
                    self.stopped_by_fire = False
                    if self.maze_space[x[0] + 1, x[1]]:
                        reward -= VISITED_PENALTY
                else:
                    self.stopped_by_fire = True
                    # reward -= MOVE_TO_FIRE_PENALTY
            else:
                self.stopped_by_wall = True
                reward -= MOVE_TO_WALL_PENALTY  # avoid moving to the wall
        # down
        elif action == 4:
            if self.observation_all_space[x[0], x[1] + 1, 0]:
                self.stopped_by_wall = False
                if not self.observation_all_space[x[0], x[1] + 1, 1]:
                    x[1] = x[1] + 1
                    self.maze_space[x[0] - 1, x[1]] = 1
                    reward += MOVE_TO_TARGET_REWARD  # expect agent go to the target faster (e.g. downward)
                    self.stopped_by_fire = False
                    if self.maze_space[x[0], x[1] + 1]:
                        reward -= VISITED_PENALTY
                else:
                    self.stopped_by_fire = True
                    # reward -= MOVE_TO_FIRE_PENALTY
            else:
                self.stopped_by_wall = True
                reward -= MOVE_TO_WALL_PENALTY  # avoid moving to the wall

        # assign new location to next state
        next_state = x
        self.state = next_state
        self.counts += 1

        reward -= STEP_PENALTY

        # positive reward if success
        if self.state == self.target:
            reward += SUCCESS_REWARD
            done = True
        # for each step, small penalty to make agent use less steps to get to the target point
        else:
            if self.maze_dead_points[self.state[0], self.state[1]]:
                reward -= DEAD_POINT_PENALTY
            done = False
        # if steps larger than MAX_STEPS_NUM, the agent failed in this episode.
        if self.counts > MAX_STEPS_NUM:
            reward -= FAIL_PENALTY
            done = True

        # get new environment after this action
        around = rm.get_local_maze_information(x[0], x[1])

        return self.state, reward, done

    # reset the maze game, all parameters, environments, etc.
    def reset(self):
        self.state = [1, 1]
        self.counts = 0
        return self.state

    # openCV to draw the maze
    def init_maze(self):
        self.maze = np.zeros((201, 201, 3), dtype=np.uint8)
        for i in range(rm.maze_cells.shape[0]):
            for j in range(rm.maze_cells.shape[1]):
                if rm.maze_cells[i][j][0]:
                    self.maze[i][j] = (255, 255, 255)
                else:
                    self.maze[i][j] = (0, 0, 0)

    # render() visualization of environment
    def render(self, mode='human'):
        # get current state
        agent = self.state
        for i in range(rm.maze_cells.shape[0]):
            for j in range(rm.maze_cells.shape[1]):
                if rm.maze_cells[i, j, 0]:
                    if rm.maze_cells[i, j, 1]:
                        # draw red fire
                        self.maze[i, j] = (0, 0, 255)
                    else:
                        # change to white if no fire
                        self.maze[i][j] = (255, 255, 255)
                else:
                    # black wall
                    self.maze[i][j] = (0, 0, 0)
        # start point
        self.maze[1, 1] = (255, 128, 128)
        # target point
        self.maze[199, 199] = (0, 128, 255)
        # agent
        self.maze[agent[0], agent[1]] = (255, 255, 0)

        img = Image.fromarray(self.maze, 'RGB')
        cv2.namedWindow("Maze", 0)
        cv2.resizeWindow("Maze", 1000, 1000)
        cv2.imshow("Maze", np.array(img))
        cv2.waitKey(1)
        # clear the last color, keep one agent color to show where the agent is
        self.maze[agent[0], agent[1]] = (255, 255, 255)
        return self.maze

    # get the dead point
    # the location is the dead point when there is only one way to go
    def get_dead_point_of_maze(self):
        for i in range(1, rm.maze_cells.shape[0] - 1):
            for j in range(1, rm.maze_cells.shape[1] - 1):
                count = 0
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
