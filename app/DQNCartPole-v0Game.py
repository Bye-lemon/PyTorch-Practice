import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchnet as tnt
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import os
import tqdm
import itertools
import math
import gym

from PIL import Image

from models.DQN import *

LOG = print
WATCH = lambda x: print(x.shape)

# Hyper Parameters
BATCH_SIZE = 32
BASIC_EPISODES = 100
NUM_EPISODES = 200
LEARNING_RATE = 0.01  # learning rate
EPSILON_START = 0.9  # 最优选择动作百分比
EPSILON_END = 0.05
EPSILON_DECAY = 200
GAMMA = 0.9  # 奖励递减参数
TARGET_REPLACE_ITER = 100  # Q 现实网络的更新频率
MEMORY_CAPACITY = 10000  # 记忆库大小
MODE = "train"
MODEL_PATH = "../logs/RL/Epoch_300.pth"
LOG_PATH = "../logs/RL/"

# Environment Definition
env = gym.make('CartPole-v0')  # 立杆子游戏
env = env.unwrapped
SCREEN_WIDTH = 600

# Load Data
resize_img = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(40, interpolation=Image.CUBIC),
    transforms.ToTensor()
])


def get_cart_location():
    world_width = env.x_threshold * 2
    scale = SCREEN_WIDTH / world_width
    return int(env.state[0] * scale + SCREEN_WIDTH / 2.0)


def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (SCREEN_WIDTH - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize_img(screen).unsqueeze(0).type(FloatTensor)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        plt.plot(means.numpy())

    plt.pause(0.001)


# Device Settings
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
BoolTensor = torch.cuda.BoolTensor if torch.cuda.is_available() else torch.BoolTensor
LOG(f"[DEVICE] Device {device} is ready.")

# Log Directory Checking
if not op.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
LOG(f"[LOGGER] Logging at {LOG_PATH}.")


# DQN Definition
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.target_net = DQNNet()
        self.eval_net = DQNNet()
        self.step_cnt = 0
        self.memory = ReplyMemory(capacity=MEMORY_CAPACITY)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def choose_action(self, x):
        epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-self.step_cnt / EPSILON_DECAY)
        if random.random() > epsilon_threshold:
            with torch.no_grad():
                return self.eval_net(torch.autograd.Variable(x).type(FloatTensor)).data.max(1)[1].cpu().numpy()[0]
        else:
            return np.random.randint(0, 2)

    def store_transition(self, *args):
        self.memory.push(*args)

    def learn(self):
        if self.step_cnt % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.step_cnt += 1

        if len(self.memory) < BATCH_SIZE:
            return
        memory = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*memory))

        non_final_mask = BoolTensor(tuple(map(lambda x: x is not None, batch.next_state)))
        non_final_next_state = torch.autograd.Variable(
            torch.cat([state for state in batch.next_state if state is not None])).to(device)
        state_batch = torch.autograd.Variable(torch.cat(batch.state)).to(device)
        action_batch = torch.autograd.Variable(torch.cat(batch.action)).to(device)
        reward_batch = torch.autograd.Variable(torch.cat(batch.reward)).to(device)

        state_action_values = self.eval_net(state_batch).gather(1, action_batch)

        next_state_values = torch.autograd.Variable(torch.zeros(BATCH_SIZE)).type(FloatTensor)
        next_state_values[non_final_mask] = self.eval_net(non_final_next_state).max(1)[0]

        expected_state_values = next_state_values * GAMMA + reward_batch

        loss = self.criterion(state_action_values, expected_state_values.view(-1, 1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Model Definition
model = DQN().to(device)
if MODEL_PATH is not None:
    model.target_net.load_state_dict(torch.load(MODEL_PATH))
    model.eval_net.load_state_dict(torch.load(MODEL_PATH))
LOG(f"[MODEL]  Model building complete.")

# Train
if MODE == "train":
    for episode in range(NUM_EPISODES):
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in itertools.count():
            action = model.choose_action(state)
            _, reward, done, _ = env.step(action)
            action = LongTensor([[action]])
            reward = FloatTensor([reward])

            last_screen, current_screen = current_screen, get_screen()
            next_state = current_screen - last_screen if not done else None

            model.store_transition(state, action, next_state, reward)

            state = next_state

            model.learn()

            if done:
                episode_durations.append(t + 1)
                print(f"Episode {BASIC_EPISODES + episode + 1} Time {t + 1}")
                break

    torch.save(model.target_net.state_dict(), op.join(LOG_PATH, f"Epoch_{BASIC_EPISODES + NUM_EPISODES}.pth"))
    env.render()
    env.close()
    plot_durations()
    plt.ioff()
    plt.show()

# Test
if MODE == "test":
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in itertools.count():
        action = model.choose_action(state)
        _, reward, done, _ = env.step(action)
        action, reward = LongTensor([[action]]), FloatTensor([reward])

        last_screen, current_screen = current_screen, get_screen()
        next_state = current_screen - last_screen if not done else None

        model.store_transition(state, action, next_state, reward)

        state = next_state

        if done:
            print(f"Time {t + 1}")
            break

    env.render()
    env.close()

