# Start with necessary imports
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time

class MarkovApp:
    def __init__(self, app_states, transition_matrix, utilities):
        self.app_states = app_states
        self.transition_matrix = transition_matrix
        self.utilities = utilities

    def get_utility(self, app_state):
        index = self.app_states.index(app_state)
        return self.utilities[index]

    def get_next_app_state(self, current_app_state):
        index = self.app_states.index(current_app_state)
        probabilities = self.transition_matrix[index]
        next_app_state = np.random.choice(self.app_states, p=probabilities)
        return next_app_state
    
# Define the Actor and Critic models
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2_actor = nn.Linear(64, 64)
        self.fc2_critic = nn.Linear(64, 64)
        self.fc3_actor = nn.Linear(64, 2)
        self.fc3_critic = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x_actor = F.relu(self.fc2_actor(x))
        x_critic = F.relu(self.fc2_critic(x))
        action_prob = F.softmax(self.fc3_actor(x_actor), dim=-1)
        state_value = self.fc3_critic(x_critic)
        return action_prob, state_value

# Define the Server and Environment classes
class Server:
    def __init__(self, id, policy, policy_optimizer, markov_app):
        self.id = id
        self.rack_state = 0  # initial state: Normal
        self.server_state = 0 # inital state: Active
        self.app_state = np.random.choice(markov_app.app_states)  # initial state in Markov Game
        self.action = np.random.choice([0, 1], p=[0.5, 0.5])
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.markov_app = markov_app

    # Modify get_reward method to use MarkovGame
    def get_reward(self, trip):
        base_reward = self.markov_app.get_utility(self.app_state)
        if trip:
            return -1
        else:
            if self.rack_state == 0 and self.server_state == 0 and self.action == 0:  # normal and active state and sprinting
                return base_reward
            elif self.rack_state == 1:
                return -1 # recovery state
            elif self.rack_state == 0 and (self.server_state == 1 or self.action == 1):
                return 0  # cooling or choose not sprinting
            
            
    def transition_state(self, num_servers, trip, recovery_flag, back_active_count):
        self.app_state = self.markov_app.get_next_app_state(self.app_state)
        p_cooling = 0.5
        if trip:
            self.rack_state = 1
        else:
            if self.rack_state == 1 and recovery_flag == True:
                if back_active_count < 0.25*num_servers:
                    self.rack_state = 0
                    back_active_count += 1
            elif self.rack_state == 0:
                if self.server_state == 0 and self.action == 0:
                    self.server_state = 1 # go to cooling
                elif self.server_state == 1: # get chance back to active
                    if np.random.rand() <= p_cooling:
                        self.server_state = 0
        return back_active_count
        
class Environment:
    def __init__(self, num_servers, lr, markov_app):
        self.num_servers = num_servers
        self.servers = []
        self.ewma_sprinting = 0.0  # initial value for EWMA of num_sprinting
        self.decay_factor = 0.99
        self.active_count = 0  # count number of visited active state for bias correction
        self.markov_app = markov_app  # Markov game object
        self.num_sprinting = 0  # Initialize num_sprinting
        self.recovery_flag = False
        self.total_reward = 0
        self.step_count = 0
        self.bias_corrected_ewma_sprinting = 0
        self.trip = False  # Initialize the trip flag
        for i in range(num_servers):
            policy = Policy()
            policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
            self.servers.append(Server(i, policy, policy_optimizer, self.markov_app))

    
    def calculate_trip_probability(self, num_sprinting):
        if num_sprinting <= self.num_servers * 0.25:
            return 0
        elif num_sprinting <= self.num_servers * 0.75:
            return (1/(0.5 * self.num_servers)) * num_sprinting - 0.5
        else:
            return 1

    def step(self):
        self.step_count += 1
        num_sprinting = 0  # Reset num_sprinting at the start of each step
        rewards = {}
        self.trip = False
        back_active_count = 0
        p_trip = 0.12
        log_prob_dict = {}
        state_values = {}
        next_state_value = 0
        
        for server in self.servers:
            input_tensor = torch.tensor([server.rack_state, server.server_state, self.bias_corrected_ewma_sprinting/self.num_servers, server.markov_app.get_utility(server.app_state)], dtype=torch.float32)
            action_probabilities, state_value = server.policy(input_tensor)
            state_values[server.id] = state_value
            if server.rack_state == 0 and server.server_state == 0: # server not in recovery state and in active state
                server.action = np.random.choice([0, 1], p=action_probabilities.detach().numpy())
                log_prob = torch.log(action_probabilities[int(server.action)])
                log_prob_dict[server.id] = log_prob
                #print(f"server {i} at app {server.app} with sprinting prob: {action_probabilities}, action is {server.action}")
                if server.action == 0:  # if server decides to sprint
                    num_sprinting += 1
            else:
                server.action = 1
                log_prob = torch.tensor(0.0)
                log_prob_dict[server.id] = log_prob
        # Calculate trip probability and trip status before proceeding
        prob_trip = self.calculate_trip_probability(num_sprinting)
        self.trip = np.random.rand() <= prob_trip  # determine if system trips

        if not all(server.rack_state == 1 for server in self.servers):
            self.active_count += 1
            self.ewma_sprinting = self.decay_factor * self.ewma_sprinting + (1 - self.decay_factor) * num_sprinting
            self.bias_corrected_ewma_sprinting = self.ewma_sprinting / (1 - self.decay_factor ** self.active_count)

        if all(server.rack_state == 1 for server in self.servers):
            if self.recovery_flag == False:
                self.recovery_flag = np.random.rand() <= p_trip #p_trip
        elif any(server.rack_state == 1 for server in self.servers) and not all(server.rack_state == 1 for server in self.servers):
            self.recovery_flag = True
        elif all(server.rack_state == 0 for server in self.servers):
            self.recovery_flag = False
            
        for server in self.servers:
            rewards[server.id] = server.get_reward(self.trip)
            back_active_count = server.transition_state(self.num_servers, self.trip, self.recovery_flag, back_active_count)
        back_active_count = 0
        
        print(f'Round {self.step_count}: EWMA of num_sprinting is {self.bias_corrected_ewma_sprinting:.1f}')
        self.total_reward = sum(rewards.values())
        
        for server in self.servers:
            state_value = torch.tensor([state_values[server.id]], requires_grad=True)
            # Calculate the next state value
            next_input = torch.tensor([server.rack_state, server.server_state, self.bias_corrected_ewma_sprinting/self.num_servers, server.markov_app.get_utility(server.app_state)], dtype=torch.float32)
            _, next_state_value = server.policy(next_input)
            # Detach the next_state_value from its computational graph
            next_state_value = next_state_value.detach()
            # Calculate the advantage
            advantage = rewards[server.id] + 0.99 * next_state_value - state_value
            action_loss = -log_prob_dict[server.id] * advantage
            # Calculate the critic loss (value loss)
            value_loss = advantage.pow(2)
            # Calculate the total loss
            policy_loss = value_loss + action_loss
            server.policy_optimizer.zero_grad()
            policy_loss.backward()
            server.policy_optimizer.step()

        return num_sprinting, self.trip, self.bias_corrected_ewma_sprinting, self.total_reward
 
start_time = time.time()
# Initialize the Markov game
app_states = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
utilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
transition_matrix = [ 
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
]
markov_app = MarkovApp(app_states, transition_matrix, utilities)
num_servers = 10
# Initialize the environment with 100 servers and Markov game
env = Environment(num_servers, 1e-3, markov_app)
"""num_sprinting_list = []
total_reward_list = []
ewma_sprinting_list = []
average_reward_list = []"""  # List to store average reward per server
# Run the game for 1000 rounds
num_sprinting = 0
trip = False
ewma_sprinting = 0
total_reward = 0
for i in range(100):
    num_sprinting, trip, ewma_sprinting, total_reward = env.step()
    # Record the number of sprinting servers, total reward, and Markov game reward for this round
    """num_sprinting_list.append(num_sprinting)
    total_reward_list.append(total_reward)
    ewma_sprinting_list.append(ewma_sprinting)
    average_reward_list.append(total_reward / env.num_servers)  # Calculate average reward per server
    print(f"Round {i+1}:")
    print(f"Number of sprinting servers: {num_sprinting}")
    print(f"System tripped: {'Yes' if trip else 'No'}")
    print(f"Average reward per server: {total_reward / env.num_servers}\n")  # Display average reward per server"""

"""# Plot the number of sprinting servers over rounds
plt.figure(figsize=(30, 20))
plt.subplot(4, 1, 1)
plt.plot(num_sprinting_list, label='Number of sprinting servers')
plt.xlabel('Round')
plt.ylabel('Number of Sprinting servers')
plt.title('Number of Sprinting servers over Rounds')
plt.legend()
plt.grid(True)

# Plot the total reward over rounds
plt.subplot(4, 1, 2)
plt.plot(average_reward_list, label='Average reward per server')
plt.xlabel('Round')
plt.ylabel('Average Reward per server')
plt.title('Average Reward per server over Rounds')
plt.legend()
plt.grid(True)

# Plot the exponent weighted moving average sprinting over rounds
plt.subplot(4, 1, 3)
plt.plot(ewma_sprinting_list, label='ewma_sprinting')
plt.xlabel('Round')
plt.ylabel('EWMA of Number of Sprinting servers')
plt.title('EWMA of Number of Sprinting servers over Rounds')
plt.legend()
plt.grid(True)

plt.show()
"""

reward_sprinting_prob = {}
for i, server in enumerate(env.servers):
    for utility in utilities:
        input_tensor = torch.tensor([0, 0, ewma_sprinting/num_servers, utility], dtype=torch.float32)
        prob, _ = server.policy(input_tensor)
        print(f"Server {i} with utility {utility}, prob of sprinting {prob.detach().numpy()[0]}")
        if utility in reward_sprinting_prob:
            reward_sprinting_prob[utility].append(prob.detach().numpy()[0])
        else:
            reward_sprinting_prob[utility] = [prob.detach().numpy()[0]]

# Compute average sprinting probabilities for each reward
average_sprinting_prob = {}
for utility, probs in reward_sprinting_prob.items():
    average_sprinting_prob[utility] = sum(probs) / len(probs)

# Print average sprinting probabilities
for utility, avg_prob in average_sprinting_prob.items():
    print(f"Utility: {utility}, Average Sprinting Probability: {avg_prob}")
end_time = time.time()
total_time = end_time - start_time
print(f"Total running time: {total_time} seconds")
