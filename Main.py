import os
import yaml
import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import argparse

# === Utility Functions ===
def clip(x, min_x, max_x):
    return max(min(x, max_x), min_x)

# === Environment ===
class IrrigationEnv(gym.Env):
    """
    Gym-like environment for irrigation scheduling with:
     - layered soil moisture
     - dynamic evapotranspiration
     - crop growth model
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super().__init__()
        self.season_length = config['season_length']
        self.crop_params = config['crop']
        self.soil_layers = config['soil_layers']
        self.day = 0
        # observation: [layer1, layer2, ..., precip, temp, growth_stage, day]
        low = np.array([0.0]*self.soil_layers + [0.0, 0.0, 0.0, 0])
        high = np.array([1.0]*self.soil_layers + [1.0, 50.0, 1.0, self.season_length])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # continuous irrigation [0, max_rate]
        self.action_space = spaces.Box(
            low=np.array([0.0]), high=np.array([config['max_irrigation']]), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.day = 0
        self.soil = np.full(self.soil_layers, 0.5)
        self.precip = 0.0
        self.temp = 25.0
        self.growth = 0.0  # 0-1 fraction
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.soil, [self.precip, self.temp, self.growth, self.day]], axis=0)

    def step(self, action):
        # Clip irrigation
        water = float(clip(action[0], 0.0, self.action_space.high[0]))
        # simulate precip and temp from real or synthetic data
        self.precip = np.random.rand() * 0.1
        self.temp = 10 + np.random.rand() * 25
        # evapotranspiration depending on temp
        et = self.crop_params['et_base'] * (1 + (self.temp - 20)/30)
        # update soil layers (simple percolation)
        self.soil[0] = clip(self.soil[0] + water + self.precip - et, 0.0, 1.0)
        for i in range(1, self.soil_layers):
            percolation = self.crop_params['perc_rate'] * self.soil[i-1]
            self.soil[i] = clip(self.soil[i] + percolation - et*0.5, 0.0, 1.0)
        # update crop growth
        moisture = np.mean(self.soil)
        growth_rate = self.crop_params['growth_coeff'] * moisture
        self.growth = clip(self.growth + growth_rate, 0.0, 1.0)
        # reward: maximize growth minus water cost
        reward = (growth_rate * 10) - water * self.crop_params['water_cost']

        self.day += 1
        done = self.day >= self.season_length
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"Day: {self.day}, Soil Layers: {self.soil}, Growth: {self.growth:.3f}")

# === Actor-Critic Network ===
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden = 128
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.fc(x)
        return torch.tanh(self.actor(x)), self.critic(x)

# === A2C Agent ===
class A2CAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.gamma = cfg['gamma']

    def select_action(self, state):
        state = torch.FloatTensor(state)
        mean, value = self.model(state)
        dist = torch.distributions.Normal(mean, 0.1)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.detach().numpy(), log_prob, value

    def update(self, rewards, log_probs, values, next_value, done):
        Qvals = []
        Qval = next_value
        for r, d in zip(reversed(rewards), reversed(done)):
            Qval = r + self.gamma * Qval * (1 - d)
            Qvals.insert(0, Qval)
        Qvals = torch.FloatTensor(Qvals)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        advantage = Qvals - values.squeeze()

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# === Training ===
def train(config_path):
    # load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # setup
    env = IrrigationEnv(cfg['env'])
    writer = SummaryWriter(log_dir=cfg['logging']['log_dir'])
    agent = A2CAgent(env.observation_space.shape[0], env.action_space.shape[0], cfg['agent'])
    save_path = cfg['logging']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for ep in range(cfg['training']['episodes']):
        state = env.reset()
        log_probs, values, rewards, dones = [], [], [], []
        total_reward = 0
        done = False
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(reward, dtype=torch.float))
            dones.append(done)
            state = next_state
            total_reward += reward

        # last value
        _, next_value = agent.model(torch.FloatTensor(state))
        loss = agent.update(rewards, log_probs, values, next_value, dones)

        writer.add_scalar('Reward/episode', total_reward, ep)
        writer.add_scalar('Loss/episode', loss, ep)

        if ep % cfg['logging']['save_freq'] == 0:
            torch.save(agent.model.state_dict(), save_path)
            print(f"Ep {ep}, Reward: {total_reward:.2f}, Loss: {loss:.4f}")

    writer.close()

# === Main ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    train(args.config)
