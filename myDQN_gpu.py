import os

import myGym                                               # import my maze env
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# initialize maze env
env = myGym.Env_Maze()

# use gpu if could
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------------------------------------------
# Experience Replay Memory

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# -------------------------------------------------------------------------------------------------------------
# DQN

class DQN(nn.Module):

    def __init__(self, h, w, actions):
        super(DQN, self).__init__()

        # design layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)

        # compute map size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(w, 2, 1), 1, 1)
        convh = conv2d_size_out(conv2d_size_out(h, 2, 1), 1, 1)
        linear_input_size = convw * convh * 64

        self.l1 = nn.Linear(linear_input_size, 128)
        self.l2 = nn.Linear(128, actions)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        return self.l2(x.view(-1, 128))


# -------------------------------------------------------------------------------------------------------------
# Input extraction

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize((3, 3), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor()])


def get_screen():
    # transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    location = env.state
    screen = torch.from_numpy(screen[:, location[0] - 1:location[0] + 1, location[1] - 1:location[1] + 1])
    # resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


# -------------------------------------------------------------------------------------------------------------
# Training

# hyperparameters initialization
BATCH_SIZE = 128
GAMMA = 0.9                 # the parameter in the updating formula
EPS_START = 1.0             # greedy policy
EPS_END = 0.1
EPS_DECAY = 5000
TARGET_UPDATE = 5
REPLAY_MEMORY_SIZE = 5000   # experience memory size

# read maze screen, only agent observation (field : 3 * 3)
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# get number of actions from gym action space
n_actions = env.action_space.n

# initialize networks and load checkpoint if need
eval_net = DQN(screen_height, screen_width, n_actions).to(device)
# eval_net.load_state_dict(torch.load('weights/eval_net_weights_0.pth'))
# eval_net.load_state_dict(torch.load('weights/final_net_weights.pth'))
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(eval_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(eval_net.parameters())

# set length of Replay Memory
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

# count agent's steps
steps_done = 0


# next action
def select_action(state):
    global steps_done
    # greedy policy
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return eval_net(state).max(1)[1].view(1, 1)
    else:
        # choose a random action to explore more with small ratio
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_total_rewards = []


# analyse the performance of the current model
def plot_rewards():
    plt.figure(1)
    plt.clf()
    rewards = torch.tensor(episode_total_rewards, dtype=torch.float)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards.numpy())
    # take 100 episode averages and plot them too
    if len(rewards) >= 100:
        means = rewards.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = eval_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in eval_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# get first several frame of random actions and observations
def random_start(skip_steps=3, m=1):
    env.reset()
    state_queue = deque([], maxlen=m)
    next_state_queue = deque([], maxlen=m)
    done = False
    for i in range(skip_steps):
        if (i + 1) <= m:
            state_queue.append(get_screen())
        elif m < (i + 1) <= 2 * m:
            next_state_queue.append(get_screen())
        else:
            state_queue.append(next_state_queue[0])
            next_state_queue.append(get_screen())

        action = env.action_space.sample()
        _, _, done = env.step(action)
        if done:
            break
    return done, state_queue, next_state_queue


# -------------------------------------------------------------------------------------------------------------
# Start Training

num_episodes = 2000
for i_episode in range(num_episodes):
    # initialize the environment and location
    done, state_queue, next_state_queue = random_start()

    state = torch.cat(tuple(state_queue), dim=1)
    for step in count():
        total_reward = 0
        reward = 0
        m_reward = 0

        action = select_action(state)
        # print steps, actions, trace of locations, etc.
        print(f'Epi:{i_episode} | Step: {step} | Location: {env.state}\t'
              f'| Action: {["stay", "left", "up", "right", "down"][action]}\t ', end='')

        # take action and get new obervation
        _, reward, done = env.step(action.item())

        print(f'(Stopped by Fire)' if env.stopped_by_fire else '' + f'(Stopped by Wall)' if env.stopped_by_wall else '')

        # if not done, read next state, compute reward
        if not done:
            next_state_queue.append(get_screen())
            next_state = torch.cat(tuple(next_state_queue), dim=1)
            total_reward += reward
            m_reward += reward
        else:
            next_state = None
            total_reward += reward
            m_reward += reward
        m_reward = torch.tensor([m_reward], device=device)

        # store replay memory
        memory.push(state, action, next_state, m_reward)

        # update the state
        state = next_state
        optimize_model()

        # plot reward in order to analyse the variation of rewards in the end
        if done:
            episode_total_rewards.append(total_reward)
            plot_rewards()
            break

    # update the target network, copying all weights and biases from eval net to target net
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(eval_net.state_dict())
        if os.path.exists('./weights/'):
            continue
        else:
            os.makedirs('./weights/')
        torch.save(eval_net.state_dict(), 'weights/eval_net_weights_{0}.pth'.format(i_episode))

print('Complete')
env.close()
torch.save(eval_net.state_dict(), 'weights/final_net_weights.pth')
