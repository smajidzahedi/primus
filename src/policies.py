import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
import torch.nn.functional as fun


class Critic(nn.Module):
    def __init__(self, input_size, h1_size, lr):
        super().__init__()
        c_l1_size = input_size ** 2 + 4 * input_size
        self.critic_layer1 = nn.Linear(c_l1_size, h1_size)
        self.critic_layer2 = nn.Linear(h1_size, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        x4 = x ** 4
        x3 = x ** 3
        x2 = x ** 2
        xx = torch.outer(x, x).flatten()
        x = torch.cat((x4, x3, x2, xx, x))
        x = torch.relu(self.critic_layer1(x))
        state_value = self.critic_layer2(x)
        return state_value


class Actor(nn.Module):
    def __init__(self, input_size, h1_size, lr):
        super().__init__()
        # Initialize Actor network
        a_l1_size = input_size ** 2 + input_size
        self.actor_layer1 = nn.Linear(a_l1_size, h1_size)
        self.actor_layer2_mean = nn.Linear(h1_size, 1)
        self.actor_layer2_std = nn.Linear(h1_size, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        x2 = x ** 2
        x = torch.cat((x2, x))
        x = torch.relu(self.actor_layer1(x))
        mean = self.actor_layer2_mean(x)
        std = torch.exp(self.actor_layer2_std(x))
        std = torch.clamp(std, min=0, max=0.01)
        dist = Normal(loc=mean, scale=std)
        u = dist.sample()
        # a = torch.tanh(u)
        log_prob = dist.log_prob(u)
        # log_prob -= torch.log(1 - a.pow(2) + 1e-6)
        # log_prob -= 2 * (np.log(2) - u - fun.softplus(-2 * u))
        # log_prob = log_prob.sum(dim=-1, keepdim=True)
        # return a * 0.5 + 0.5, log_prob
        return u, log_prob


class Policy:
    def get_new_action(self, state):
        pass


class ACPolicy(Policy):
    def __init__(self, a_input_size, c_input_size, a_h1_size, c_h1_size, a_lr, c_lr, df):
        self.actor = Actor(a_input_size, a_h1_size, a_lr)
        self.critic = Critic(c_input_size, c_h1_size, c_lr)
        self.log_prob = None
        self.discount_factor = df
        self.state_value = torch.tensor([0.0], requires_grad=True)

    def get_new_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action, self.log_prob = self.actor(state_tensor)
        return action.item()

    def update_policy(self, next_state, reward, update_actor):
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        next_state_value = self.critic(next_state_tensor)
        estimate = reward + self.discount_factor * next_state_value.detach()
        advantage = estimate - self.state_value

        if update_actor:
            actor_loss = - self.log_prob * advantage.detach()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

        critic_loss = advantage.pow(2).mean()
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.state_value = self.critic(next_state_tensor)


class ThrPolicy(Policy):
    def __init__(self, threshold):
        self.threshold = threshold

    def get_new_action(self, state):
        return self.threshold
