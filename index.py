import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import random
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')


def plot_res(values, title=''):
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    # clear_output(wait=True)

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(4, 128).cuda()
        self.fc2 = nn.Linear(128, 64).cuda()
        self.fc3 = nn.Linear(64, 2).cuda()

    def forward(self, image, observation):
        x = torch.tensor(observation).float().cuda()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze()
        return x


target_DQN = (Net1()).cuda()
policy_DQN = (Net1()).cuda()

# Use Huber's loss to get rid of exploding gradients
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(policy_DQN.parameters(), lr=0.01)

target_DQN.load_state_dict(policy_DQN.state_dict())
done = False

# initial observation
observation = env.reset()
memory = np.array([])


def reflect():
    gamma = 0.9
    random_batch = np.array(random.sample(list(memory), 80))
    states = np.array([])
    expected_vals = np.array([])

    for state, action, reward, next_state, done in random_batch:
        action_vals = policy_DQN.forward(state[0], state[1])
        assert not torch.isnan(action_vals).any()
        action_vals[action] = reward
        if done == False:
            action_vals[action] = reward + \
                (gamma *
                 torch.max(target_DQN.forward(next_state[0], next_state[1])).detach().item())

        # Using only the "observation" as state input. And, neglecting the image pixel input.
        if(states.shape[0] > 0):
            states = np.append(states, np.array([state[1]]), axis=0)
            expected_vals = np.append(
                expected_vals, [action_vals.cpu().detach().numpy()], axis=0)
        else:
            states = np.array([state[1]])
            expected_vals = np.array([action_vals.cpu().detach().numpy()])

    optimizer.zero_grad()
    predicted_vals = policy_DQN.forward([], states)
    # print('FINAL', predicted_vals.shape, expected_vals.shape)
    loss = criterion(predicted_vals, torch.from_numpy(expected_vals).cuda())
    loss.backward()
    optimizer.step()


def rgb_to_gray(rgb):
    gray = (0.2989 * rgb[:, 0, :, :]) + (0.5870 *
                                         rgb[:, 1, :, :]) + (0.1140 * rgb[:, 2, :, :])
    return torch.unsqueeze(gray, 0)


epsilon = 1
total_rewards = []

for episode in range(200):
    print("New Episode", episode, epsilon, memory.shape)
    state = rgb_to_gray(torch.tensor([env.render(
        mode='rgb_array').transpose(2, 0, 1)]).cuda()).float()
    done = False
    observation = env.reset()
    epsilon *= 0.98
    total_reward = 0
    for transition in range(0, 200):

        # Q(s,a)
        Q_vals = policy_DQN.forward(state, [observation])
        Q_expected = torch.max(Q_vals)

        action_probs = np.ones(2) * (epsilon/2)
        action_probs[np.argmax(Q_vals.cpu().detach().numpy())
                     ] += (1 - (epsilon))

        action_to_take = np.random.choice([0, 1], p=action_probs)

        next_observation, reward, done, info = env.step(action_to_take.item())
        next_state = rgb_to_gray(torch.tensor([env.render(
            mode='rgb_array').transpose(2, 0, 1)]).cuda()).float()
        total_reward += reward

        if(episode == 0 and transition == 0):
            memory = np.array(
                [[[state.cpu().detach().numpy(), observation], action_to_take, reward, [next_state.cpu().detach().numpy(), next_observation], done]])
        else:
            memory = np.append(
                memory, [[[state.cpu().detach().numpy(), observation], action_to_take, reward, [next_state.cpu().detach().numpy(), next_observation], done]], axis=0)

        if memory.shape[0] > 100:
            # Reflect only if there's enough memory
            reflect()
        if(memory.shape[0] > 5000):
            # Remove old memory
            memory = memory[100:, :]
        state = next_state
        observation = next_observation
        if(done):
            break
    print("Reward for an episode: ", total_reward)
    total_rewards.append(total_reward)
    if(episode % 10 == 0):
        target_DQN.load_state_dict(policy_DQN.state_dict())

plot_res(total_rewards, "Double DQN (W) Experience Replay")
