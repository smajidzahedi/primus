import torch
from torch import nn, optim
from torch.distributions import Normal


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
        a_l1_size = 2 * input_size
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
    def __init__(self, a_input_size, c_input_size, a_h1_size, c_h1_size, a_lr, c_lr, df, mini_batch_size=1):
        self.actor = Actor(a_input_size, a_h1_size, a_lr)
        self.critic = Critic(c_input_size, c_h1_size, c_lr)
        self.log_prob = None
        self.discount_factor = df
        self.state_value = torch.tensor([0.0], requires_grad=True)
        self.c_values = []
        self.a_values = []
        self.rewards = []
        self.log_probs = []
        self.masks = []
        self.iteration = 0
        self.mini_batch_size = mini_batch_size

    def get_new_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action, self.log_prob = self.actor(state_tensor)
        return action.item()

    def compute_returns(self, next_state_value):
        r = next_state_value
        c_returns = []
        a_returns = []
        for step in reversed(range(len(self.rewards))):
            r = self.rewards[step] + self.discount_factor * r
            c_returns.insert(0, r)
            if self.masks[step]:
                a_returns.insert(0, r)
        return c_returns, a_returns

    def update_policy(self, next_state, reward, update_actor):
        self.c_values.append(self.state_value)
        self.rewards.append(reward)
        self.masks.append(update_actor)

        if update_actor:
            self.log_probs.append(self.log_prob)
            self.a_values.append(self.state_value)

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        self.iteration += 1

        if self.iteration == self.mini_batch_size:
            next_state_value = self.critic(next_state_tensor)
            c_returns, a_returns = self.compute_returns(next_state_value)

            if len(self.a_values) > 0:
                a_returns = torch.cat(a_returns).detach()
                a_values = torch.cat(self.a_values)
                a_advantage = a_returns - a_values

                log_probs = torch.cat(self.log_probs)

                actor_loss = -(log_probs * a_advantage.detach()).mean()

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            c_returns = torch.cat(c_returns).detach()
            c_values = torch.cat(self.c_values)
            c_advantage = c_returns - c_values

            critic_loss = c_advantage.pow(2).mean()

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            self.rewards = []
            self.c_values = []
            self.a_values = []
            self.log_probs = []
            self.masks = []
            self.iteration = 0

        self.state_value = self.critic(next_state_tensor)


class ThrPolicy(Policy):
    def __init__(self, threshold):
        self.threshold = threshold

    def get_new_action(self, state):
        return self.threshold
