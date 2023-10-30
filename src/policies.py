import numpy as np
import torch
from torch import nn, optim

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("This method should be overridden.")


"""
Actor Critic Policy
Actor network and Critic network, both of them have two fully connected layers.
"""


class ACPolicy(Policy):
    def __init__(self, a_input_size, c_input_size, a_h1_size, c_h1_size, a_lr, c_lr):
        super().__init__()
        # Initialize Actor network
        a_l1_size = a_input_size ** 2 + 2 * a_input_size
        self.actor_layer1 = nn.Linear(a_l1_size, a_h1_size)
        self.actor_layer2 = nn.Linear(a_h1_size, 2)
        self.a_lr = a_lr
        actor_params = [self.actor_layer1.weight, self.actor_layer1.bias,
                        self.actor_layer2.weight, self.actor_layer2.bias]
        self.actor_optimizer = optim.Adam(actor_params, lr=self.a_lr)

        # Initialize Critic network
        self.c_lr = c_lr
        c_l1_size = c_input_size ** 2 + 4 * c_input_size
        self.critic_layer1 = nn.Linear(c_l1_size, c_h1_size)
        self.critic_layer2 = nn.Linear(c_h1_size, 1)
        critic_params = [self.critic_layer1.weight, self.critic_layer1.bias,
                         self.critic_layer2.weight, self.critic_layer2.bias]
        self.critic_optimizer = optim.Adam(critic_params, lr=self.c_lr)

    def forward_actor(self, x):
        x2 = x ** 2
        xx = torch.outer(x, x).flatten()
        x = torch.cat((x2, xx, x))
        x = torch.relu(self.actor_layer1(x))
        # x_max = torch.max(x)
        # x = x - x_max
        softmax = torch.softmax(self.actor_layer2(x), dim=-1)
        return softmax

    def forward_critic(self, x):
        x4 = x ** 4
        x3 = x ** 3
        x2 = x ** 2
        xx = torch.outer(x, x).flatten()
        x = torch.cat((x4, x3, x2, xx, x))
        x = torch.relu(self.critic_layer1(x))
        state_value = self.critic_layer2(x)
        return state_value

    def forward(self, x):
        x_actor, x_critic = x[0], x[1]
        action_prob = self.forward_actor(x_actor)
        state_value = self.forward_critic(x_critic)
        return action_prob, state_value


"""
Threshold Policy: Comparing given threshold value and current utility value.
If current utility value is higher than the given threshold value --> sprint.
Else --> not sprint
"""


class ThrPolicy(Policy):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        if x.item() > self.threshold:  # The first element of x is the utility from sprinting
            action_prob = torch.tensor([1.0, 0.0])
        else:
            action_prob = torch.tensor([0.0, 1.0])
        return action_prob, 0