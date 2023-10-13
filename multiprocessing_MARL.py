from multiprocessing import Process, Queue
import numpy as np
from scipy.stats import norm
import torch
from torch import nn, optim
import time
import os
from tqdm import tqdm
import json

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

class App:
    def __init__(self):
        self.app_state_history = []
        self.current_state_ewma = None

    def get_sprinting_utility(self):
        raise NotImplementedError("This method should be overridden.")
    
    def get_cooling_utility(self):
        return 0
    
    def get_recovery_utility(self):
        return 0
    
    def get_current_state(self):
        return float(self.current_state)

    def update_state(self, action):
        self.app_state_history.append(self.current_state_ewma)

    def print_state(self, server_id, path):
        file_path = os.path.join(path, f"server_{server_id}_app_state.txt")
        with open(file_path, 'w') as file:
            for h in self.app_state_history:
                file.write(f"{str(h)}\n")


"""
State transition follows Markov process
"""
class MarkovApp(App):
    def __init__(self, app_states, transition_matrix, utilities, initial_state):
        super().__init__()
        self.app_states = app_states
        self.transition_matrix = transition_matrix
        self.utilities = utilities
        self.current_state = initial_state

    def get_sprinting_utility(self):
        index = self.app_states.index(self.current_state)
        return self.utilities[index]

    def update_state(self, action):
        #super().update_state(action)
        index = self.app_states.index(self.current_state)
        probabilities = self.transition_matrix[index]
        self.current_state = np.random.choice(self.app_states, p=probabilities)


"""
State transition follows Unifrom distribution
"""
class UniformApp(App):
    def __init__(self, app_states, utilities):
        super().__init__()
        self.app_states = app_states
        self.utilities = utilities
        self.current_state = np.random.choice(self.app_states)

    def get_sprinting_utility(self):
        return self.utilities[self.app_states.index(self.current_state)]

    def update_state(self, action):
        #super().update_state(action)
        self.current_state = np.random.choice(self.app_states)


"""
State transition follows Normal distribution
"""
class NormalApp(App):
    def __init__(self, app_states, utilities):
        super().__init__()
        self.app_states = app_states
        self.utilities = utilities
        self.loc = (self.utilities[0] + self.utilities[-1]) / 2 # mean
        self.scale = (self.utilities[-1] - self.utilities[0]) / 4   # standard deviation
        # Calculate probability from normal distribution
        self.prob = []
        prev_cdf = 0
        for utility in self.utilities[:len(self.utilities)-1]:
            current_cdf = norm.cdf(utility, self.loc, self.scale)
            self.prob.append(current_cdf - prev_cdf)
            prev_cdf = current_cdf
        self.prob.append(1 - prev_cdf)
        self.current_state =  np.random.choice(self.app_states, p=self.prob)
        
    def get_sprinting_utility(self):
        return self.utilities[self.app_states.index(self.current_state)]
    
    def update_state(self, action):
        #super().update_state(action)
        self.current_state = np.random.choice(self.app_states, p=self.prob)


"""
Queue application uses Poisson distribution to model events that occur randomly and independently.
"""
class QueueApp(App):
    def __init__(self, utilities, arrival_tps, sprinting_tps, nominal_tps,
                 max_queue_length=1000):
        super().__init__()
        self.utilities = utilities
        self.arrival_tps = arrival_tps
        self.current_state = 0
        self.current_state_ewma = None
        self.sprinting_tps = sprinting_tps
        self.nominal_tps = nominal_tps
        self.max_queue_length = max_queue_length
        self.decay_factor = 0.99
    
    #   current state for queue app is the current queue length in range 0 to 1
    def get_current_state(self):
        return min(self.current_state / self.max_queue_length, 1)
    
    def get_sprinting_utility(self):
        return -min(self.current_state + self.arrival_tps - min(self.current_state, self.sprinting_tps), self.max_queue_length) / self.max_queue_length
    
    def get_cooling_utility(self):
        return -min(self.current_state + self.arrival_tps - min(self.current_state, self.nominal_tps), self.max_queue_length) / self.max_queue_length
    
    def get_recovery_utility(self):
        return self.get_cooling_utility()


    def update_state(self, action):
        arrived_tasks = np.random.poisson(self.arrival_tps)
        departed_tasks = min(self.current_state, np.random.poisson(self.nominal_tps))
        if action == 0:
            departed_tasks = min(self.current_state, np.random.poisson(self.sprinting_tps))
        self.current_state = self.current_state + arrived_tasks - departed_tasks
        if self.current_state_ewma is None:  # If this is the first state
            self.current_state_ewma = self.current_state
        else:
            self.current_state_ewma = self.decay_factor * self.current_state_ewma + (1 - self.decay_factor) * self.current_state
        super().update_state(action)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("This method should be overridden.")


"""
Deep Q Network with two fully connected layers.
Return q values of sprint and not sprint.
"""
class QNetwork(nn.Module):
    def __init__(self, l1_in, l1_out_l2_in):
        super().__init__()
        self.fc1= nn.Linear(l1_in, l1_out_l2_in)
        self.fc2 = nn.Linear(l1_out_l2_in, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

"""
Actor Critic Policy
Actor network and Critic network, both of them have two fully connected layers.
Actor network returns a threshold value.
Critic network returns a state value.
The class is tailored to handle two types of applications, 
defined by the variable app_type: queue_app and other three app_type.
For queue_app, the actor network return mean and standard deviation for a normal distribution.
For other three types of application, it return a class of probabilities.
"""
class ACPolicy(nn.Module):
    def __init__(self, l1_in_actor, l1_in_critic, l1_out_l2_in_actor, l1_out_l2_in_critic, app_type):
        super().__init__()
        self.app_type = app_type
        
        # Initialize Actor network
        self.actor_layer1 = nn.Linear(l1_in_actor, l1_out_l2_in_actor)
        
        if self.app_type == "queue_app":
            self.actor_layer2_mean = nn.Linear(l1_out_l2_in_actor, 1)
            self.actor_layer2_std = nn.Linear(l1_out_l2_in_actor, 1)
        else:
            self.actor_layer2 = nn.Linear(l1_out_l2_in_actor, 10)
        
        # Initialize Critic network
        self.critic_layer1 = nn.Linear(l1_in_critic, l1_out_l2_in_critic)
        self.critic_layer2 = nn.Linear(l1_out_l2_in_critic, 1)

    def forward_actor(self, x):
        x = torch.relu(self.actor_layer1(x))
        if self.app_type == "queue_app":
            mean = self.actor_layer2_mean(x)
            std = torch.exp(self.actor_layer2_std(x))
            std = torch.clamp(std, min=0.001, max=1)  # Limit the standard deviation
            return mean, std
        else:
            threshold_probs = torch.softmax(self.actor_layer2(x), dim=-1)
            return threshold_probs

    def forward_critic(self, x):
        x = torch.relu(self.critic_layer1(x))
        state_value = self.critic_layer2(x)
        return state_value

    def forward(self, x):
        x_actor, x_critic = x[0], x[1]
        if self.app_type == "queue_app":
            mean, std = self.forward_actor(x_actor)
            action_dist = torch.distributions.Normal(mean, std)
            threshold = action_dist.sample()
        else:
            threshold_probs = self.forward_actor(x_actor)
            threshold = torch.tensor([torch.argmax(threshold_probs).item() / 10])  # Take the argmax to get the discrete threshold value
        # Critic forward pass
        state_value = self.forward_critic(x_critic)
        
        return threshold, state_value

    
"""
Threshold Policy
Comparing given threshold value and current utility value.
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
        return action_prob

class Server:
    def __init__(self, server_id, policy, app, path, server_config, 
                 num_servers, servers_per_worker, game_type):
        self.server_id = server_id
        self.rack_state = 0  # initial state: Normal
        self.server_state = 0   # initial state: Active
        self.policy = policy    # learning policy
        self.app = app  # application
        self.path = path    # location of storing files
        self.num_servers = num_servers
        self.servers_per_worker = servers_per_worker
        self.game_type = game_type  # Mean Field game or Non-Mean Field game
        self.recovery_cost = server_config["recovery_cost"]
        self.cooling_prob = server_config["cooling_prob"]   # prob. of staying cooling state
        self.reward = 0
        self.action = 1
        self.discount_factor = server_config["discount_factor"]
        self.iter = 0
        if self.game_type == "MF":
            self.frac_sprinters = torch.zeros(1)
        else:
            self.frac_sprinters = torch.zeros(self.num_servers-1)
        self.info_from_worker = [self.frac_sprinters, self.rack_state, self.iter]

    # Sending information to worker 
    def set_info_from_worker(self, info_from_worker):
        self.info_from_worker = info_from_worker

    def get_action_reward(self, network_return):
        pass

    # update application state, rack_state, server_state, and fractional number of sprinters.
    def update_state(self, rack_state, frac_sprinters):
        self.app.update_state(self.action)
        self.rack_state = rack_state
        if self.game_type == "NMF":
            self.frac_sprinters = torch.tensor(np.delete(frac_sprinters, self.server_id), dtype=torch.float32)
        else:
            self.frac_sprinters = torch.tensor(frac_sprinters)
        if self.server_state == 1:
            if np.random.rand() > self.cooling_prob:    # stay in cooling
                self.server_state = 0
        elif self.action == 0:     # go to cooling
            self.server_state = 1

    def update_policy(self, next_tensor):
        return

    def take_action(self, input_tensor):
        return

    def run_server(self):
        pass

    # write reward into files
    def print_reward(self, rewards):
        file_path = os.path.join(self.path, f"server_{self.server_id}_reward.txt")
        with open(file_path, 'w+') as file:
            for r in rewards:
                file.write(f"{str(r)}\n")
    
    def print_policy(self, policies):
        file_path = os.path.join(self.path, f"server_{self.server_id}_reward.txt")
        with open(file_path, 'w+') as file:
            for p in policies:
                file.write(f"{str(p)}\n")

#   server with Actor-Critic policy
class AC_server(Server):
    def __init__(self, server_id, policy, optimizer, app, path, server_config, 
                 num_servers, servers_per_worker, game_type):
        super().__init__(server_id, policy, app, path, server_config, 
                 num_servers, servers_per_worker, game_type)
        self.actor_optimizer = optimizer[0]
        self.critic_optimizer = optimizer[1]
        self.state_value = torch.tensor([0.0])
        self.threshold = torch.tensor([0.0])

    # Update Actor and Critic networks' parameters
    def update_policy(self, next_tensor):
        if isinstance(self.app, QueueApp):
            mean, std = self.policy.forward_actor(next_tensor[0])
            threshold_distribution = torch.distributions.Normal(mean, std)
            log_prob = threshold_distribution.log_prob(self.threshold)
        else:
            threshold_probs = self.policy.forward_actor(next_tensor[0])
            log_prob = torch.log(threshold_probs[torch.argmax(threshold_probs).item()])

        next_state_value = self.policy.forward_critic(next_tensor[1])
        advantage = self.reward + self.discount_factor * next_state_value - self.state_value

        action_loss = -log_prob * advantage.detach()
        loss_fn = nn.MSELoss()
        value_loss = loss_fn(self.state_value, self.reward + self.discount_factor * next_state_value)

        # Update the actor
        self.actor_optimizer.zero_grad()
        action_loss.backward()
        self.actor_optimizer.step()

        # Update the critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

    # get threshold value and state value from AC_Policy network, choose sprint or not and get immediate reward
    def take_action(self, input_tensor):
        self.threshold, self.state_value = self.policy(input_tensor)
        self.action, self.reward = self.get_action_reward(self.threshold)

    # Given threshold value, choose sprint or not, and get immediate reward
    def get_action_reward(self, network_return):
        threshold = network_return.item()
        if self.app.get_sprinting_utility() > threshold:
            action = 0
        else:
            action = 1
        if self.rack_state == 1:    # rack is in recovery
            return 1, -self.recovery_cost + self.app.get_recovery_utility()
        elif self.server_state == 0 and action == 0:
            return 0, self.app.get_sprinting_utility()  # server sprints
        else:
            return 1, self.app.get_cooling_utility()
    
    # Run server function, return server's id, action and reward which are send to worker.
    def run_server(self):
        frac_sprinters = self.info_from_worker[0]
        rack_state = self.info_from_worker[1]
        self.iter = self.info_from_worker[2]
        if self.iter == 20000 and isinstance(self.app, QueueApp):
            self.app.arrival_tps = 18 # increase 20%
        self.update_state(rack_state, frac_sprinters)
        rack_state_tensor = torch.tensor([self.rack_state])
        server_state_tensor = torch.tensor([self.server_state])
        current_utility_tensor = torch.tensor([self.app.get_sprinting_utility()])
        next_tensor_actor = self.frac_sprinters
        next_tensor_critic = torch.cat((rack_state_tensor, server_state_tensor, 
                                        current_utility_tensor, self.frac_sprinters))
        
        next_tensor = [next_tensor_actor, next_tensor_critic]
        self.update_policy(next_tensor)
        self.take_action(next_tensor)
        return self.server_id, self.action, self.reward
            
#   Server with DQN
class Q_server(Server):
    def __init__(self, server_id, policy, target_dqn, optimizer, app, path, server_config, 
                 num_servers, servers_per_worker, game_type):
        super().__init__(server_id, policy, app, path, server_config, 
                 num_servers, servers_per_worker, game_type)
        self.optimizer = optimizer
        self.q_value = torch.tensor([0.0, 1.0], requires_grad=True)
        self.dqn = policy   # DQN for getting q_value, has backpropagation
        self.target_dqn = target_dqn    #   target DQN for getting next_q_value, no backpropagation
        self.tau = 0.005

    # Update DQN parameters
    def update_policy(self, next_tensor):
        next_q_values = self.target_dqn(next_tensor).detach()
        max_next_q_value = torch.max(next_q_values)
        target_q_value = self.reward + self.discount_factor * max_next_q_value

        loss_fn = nn.MSELoss()
        loss = loss_fn(self.q_value[int(self.action)], target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # get q value from DQN, choose sprint or not and get immediate reward
    def take_action(self, input_tensor):
        self.q_value = self.dqn(input_tensor)
        self.action, self.reward = self.get_action_reward()

    # Choose sprint or not by given q values, and get immediate reward
    def get_action_reward(self):
        if self.rack_state == 1:    # rack is in recovery
            return 1, -self.recovery_cost + self.app.get_recovery_utility()
        elif self.server_state == 0 and torch.argmax(self.q_value).item() == 0:
            return 0, self.app.get_sprinting_utility()  # server sprints
        else:
            return 1, self.app.get_cooling_utility()
                

    # Run server function, return server's id, action and reward which are send to worker.
    def run_server(self):
        frac_sprinters = self.info_from_worker[0]
        rack_state = self.info_from_worker[1]
        self.update_state(rack_state, frac_sprinters)
        
        rack_state_tensor = torch.tensor([self.rack_state], dtype=torch.float)
        server_state_tensor = torch.tensor([self.server_state], dtype=torch.float)
        current_utility_tensor = torch.tensor([self.app.get_sprinting_utility()])
        next_state = torch.cat((rack_state_tensor, server_state_tensor, current_utility_tensor, self.frac_sprinters))
        self.update_policy(next_state)
        self.take_action(next_state)
        
        for target_param, policy_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        return self.server_id, self.action, self.reward


#  server with threshold policy.
#  it is a fixed policy, so it doesn't need update policy
class Thr_server(Server):
    def __init__(self, server_id, policy, app, path, server_config, 
                 num_servers, servers_per_worker, game_type):
        super().__init__(server_id, policy, app, path, server_config, 
                 num_servers, servers_per_worker, game_type)

    # Get sprinting probability from Thr_Policy, and choose sprint or not by this probability, and get immediate reward
    def take_action(self, input_tensor):
        action_probs = self.policy(input_tensor)
        self.action, self.reward = self.get_action_reward(action_probs)

    # Given sprinting probability, choose sprint or not, and get immediate reward
    def get_action_reward(self, network_return):
        action_probs = network_return
        if self.rack_state == 1:    # rack is in recovery
            return 1, -self.recovery_cost + self.app.get_recovery_utility()
        elif self.server_state == 0 and np.random.choice([0, 1], p=action_probs.detach().numpy()) == 0:
            return 0, self.app.get_sprinting_utility()  # server sprints
        else:
            return 1, self.app.get_cooling_utility()
    
    # Run server function, return server's id, action and reward which are send to worker.
    def run_server(self):
        frac_sprinters = self.info_from_worker[0]
        rack_state = self.info_from_worker[1]
        self.update_state(rack_state, frac_sprinters)
        current_utility_tensor = torch.tensor([self.app.get_sprinting_utility()])
        self.take_action(current_utility_tensor)
        return self.server_id, self.action, self.reward

"""
An independent processor. Get information from each server, 
and send rack state and frac_sprinters back to each of them. 
Determining system trip or not.
"""
class Coordinator:
    def __init__(self, coordinator_config, w2c_queues, c2w_queues, path, num_workers, num_servers, game_type, 
                 app_type, policy_type, threshold, alpha, epsilon_bar):
        self.num_sprinters = 0  # Initialize num_sprinting
        self.num_recovery = 0 # store total number of recovery happens over rounds
        self.num_active = 0 # store total number of active rounds
        self.num_workers = num_workers
        self.num_servers = num_servers
        self.game_type = game_type
        self.app_type = app_type
        self.decay_factor = coordinator_config["decay_factor"]  #   for fictitious play
        self.w2c_queues = w2c_queues
        self.c2w_queues = c2w_queues
        self.recovery_prob = coordinator_config["recovery_prob"]    # prob. for staying recovery state
        self.in_recovery = False    # flag for recovery
        self.active_count = 0   # count number of active iterations
        self.min_frac = coordinator_config["min_frac"]  # lower bound of sprinters of system trip
        self.max_frac = coordinator_config["max_frac"]  # higher bound of sprinters of system trip
        self.rack_states = np.zeros(self.num_servers)
        self.release_frac = coordinator_config["release_frac"]  #   prob. of servers that can leave recovery state
        self.num_iterations = coordinator_config["num_iterations"]
        self.path = path
        self.policy_type = policy_type
        self.threshold = threshold  # Given threshold value, when policy_type == Thr_Policy
        if self.game_type == "MF":
            self.frac_sprinters = torch.zeros(1)
            self.avg_frac_sprinters = torch.zeros(1) # initial value for exponential moving average of num_sprinting
        else:
            self.frac_sprinters = torch.zeros(self.num_servers)
            self.avg_frac_sprinters = torch.zeros(self.num_servers) # initial value for exponential moving average of num_sprinting
        self.sensitivity = 1/self.num_servers # sensitivity for differential privacy
        self.alpha = alpha  # Rényi Divergence Order 
        self.epsilon_bar = epsilon_bar # Privacy budget
        self.noise = coordinator_config["noise"]
            


    #   whether system trip or not
    def is_tripped(self):
        prob = min(max((np.mean(self.frac_sprinters.tolist()) - self.min_frac) / (self.max_frac - self.min_frac), 0), 1)
        return np.random.rand() < prob

    # Calculate number of sprinters in this round, determining whether system trip or not.
    # Calculate the fractional number of sprinters by Bias-Corrected Exponential Weighted Moving Average
    # Add noise on the fraction number of sprinters in this round (# of sprinters / total # of servers)
    def aggregate_actions(self, actions):
        self.num_sprinters = self.num_servers - actions.sum()
        if self.in_recovery:
            assert(self.num_sprinters == 0)
            if np.random.rand() > self.recovery_prob:   #   whether servers are allowed start to leave recovery state
                self.in_recovery = False
        else:
            self.in_recovery = self.is_tripped()
            self.active_count += 1
            self.avg_frac_sprinters *= self.decay_factor
            if self.game_type == "MF":
                if self.noise == 1:
                    self.avg_frac_sprinters += (1 - self.decay_factor) * self.gaussian_mech_RDP(self.num_sprinters / self.num_servers)
                else:
                    self.avg_frac_sprinters += (1 - self.decay_factor) * (self.num_sprinters / self.num_servers)
            else:
                if self.noise == 1:
                    self.avg_frac_sprinters += (1 - self.decay_factor) * self.gaussian_mech_RDP(1 - actions) # since action=0 means sprint, flip 0 and 1 to calculate the frac_sprint for each server
                else:
                    self.avg_frac_sprinters += (1 - self.decay_factor) * (1 - actions) # since action=0 means sprint, flip 0 and 1 to calculate the frac_sprint for each server
            self.frac_sprinters = self.avg_frac_sprinters / (1 - self.decay_factor ** self.active_count)    #   Exponantial weighted moving average
        # release servers from recovery state randomly
        if self.in_recovery:
            self.rack_states = np.ones(self.num_servers)
        elif np.count_nonzero(self.rack_states) >= self.num_servers * self.release_frac: # randomly select servers leave recovery state
            recovers = np.random.choice(np.where(self.rack_states == 1)[0], size=int(self.num_servers *
                                                                                    self.release_frac), replace=False)
            self.rack_states[recovers] = 0
        else:   #   all servers leave recovery state
            self.rack_states = np.zeros(self.num_servers)
    
    # Rényi Differential Privacy
    def gaussian_mech_RDP(self, scalar):
        sigma = np.sqrt((self.sensitivity ** 2 * self.alpha) / (2 * self.epsilon_bar))
        return scalar + np.random.normal(loc=0, scale=sigma)

    # Main function for coordinator
    def run_coordinator(self):
        actions_array = np.zeros(self.num_servers)
        frac_sprinters_list = []
        i = 0
        num_active = 0
        num_recovery = 0
        pbar = tqdm(total=self.num_iterations, desc="Iterations")
        while i < self.num_iterations:   
            rack_state_list = []
            # Split the array into self.num_workers parts
            reshaped_states = np.array_split(self.rack_states, self.num_workers)
            # Now, iterate over the queues and reshaped states
            for q, states in zip(self.c2w_queues, reshaped_states):
                q.put((self.frac_sprinters.tolist(), states.tolist(), i))
                rack_state_list += states.tolist()
            if not all(rack_state_list) == 1: # count number of non-recovery iterations
                num_active += 1
            elif all(rack_state_list) == 1:
                num_recovery += 1   # count number of recovery iterations
            # get information from workers
            for q in self.w2c_queues:
                info = q.get()
                server_ids, actions = info
                for j, index in enumerate(server_ids):
                    actions_array[index] = actions[j]
            self.aggregate_actions(actions_array)
            frac_sprinters_list.append(self.frac_sprinters)
            i += 1
            pbar.update(1)
        pbar.close()
        self.num_recovery = num_recovery
        self.num_active = num_active
        print(f"total active round: {self.num_active}")
        # Send stop to all
        for q in self.c2w_queues:
            q.put('stop')
        self.print_frac_sprinters(frac_sprinters_list)
        self.print_num_recovery()

    # Record fractional number of sprinters in each iterations
    def print_frac_sprinters(self, frac_sprinters_list):
        file_path = os.path.join(self.path, "frac_sprinters.txt")
        with open(file_path, 'w+') as file:
            for fs in frac_sprinters_list:
                #fs_num = round(np.mean(fs.tolist()), 2)
                file.write(f"{fs.tolist()[0]}\n")
    
    # Record total number of recovery iterations 
    def print_num_recovery(self):
        file_path = os.path.join(self.path, f"{self.app_type}_num_recovery.txt")
        with open(file_path, 'a+') as file:
            if self.policy_type == "thr_policy":
                file.write(f"{self.policy_type}_{str(self.threshold)}_{self.app_type}\t{self.num_recovery}\n")
            else:
                file.write(f"{self.policy_type}_{self.game_type}_{self.app_type}\t{self.num_recovery}\n")

"""
Workers are also independent processors. Each worker manages several servers. 
Workers get information from their servers, and send servers' information to coordinator, and vice versa.
"""
class Worker:
    def __init__(self, servers, w2c_queue, c2w_queue):
        self.servers = servers
        self.w2c_queue = w2c_queue
        self.c2w_queue = c2w_queue
        self.rewards = {}

    def run_worker(self):
        while True:
            server_ids = []
            actions = []
            info = self.c2w_queue.get()
            if info == 'stop':
                for server in self.servers:
                    server.print_reward(self.rewards[server.server_id])
                    if isinstance(server.app, QueueApp):
                        server.app.print_state(server.server_id, server.path)
                break
            frac_sprinters, rack_state, iter = info
            for i, server in enumerate(self.servers):
                server.set_info_from_worker([frac_sprinters, rack_state[i], iter])
                server_id, action, reward = server.run_server()
                server_ids.append(server_id)
                actions.append(action)
                if server_id in self.rewards:
                    self.rewards[server_id].append(reward)
                else:
                    self.rewards[server_id] = []
                    self.rewards[server_id].append(reward)
            self.w2c_queue.put((server_ids, actions))


def main(config_file_name):
    start_time = time.time()
    with open(config_file_name, 'r') as f:
        config = json.load(f)
    folder_name = config["folder_name"]
    coordinator_config = config["coordinator_config"]
    servers_config = config["servers_config"]
    app_states = config["app_states"]
    utilities = config["utilities"]
    num_workers = config["num_workers"]
    num_servers = config["num_servers"]
    game_type = config["game_type"]
    app_type = config["app_type"]
    policy_type = config["policy_type"]
    threshold = config["threshold"]
    alpha = config["alpha"]
    epsilon_bar = config["epsilon_bar"]
    path = f"{folder_name}/{num_servers}_server/{app_type}"
    if not os.path.exists(path):
        os.makedirs(path)
    w2c_queues = [Queue() for _ in range(num_workers)]
    c2w_queues = [Queue() for _ in range(num_workers)]
    servers = []
    worker_processors = []
    servers_per_worker = num_servers // num_workers
    coordinator = Coordinator(coordinator_config, w2c_queues, c2w_queues, path, num_workers, num_servers, game_type, app_type, policy_type, threshold, alpha, epsilon_bar)
    policies = []
    for i in range(num_servers):
        app = None
        if "markov_app" == app_type:
            transition_matrix = config["transition_matrix"]
            app = MarkovApp(app_states, transition_matrix, utilities, np.random.choice(app_states))
        elif "uniform_app" == app_type:
            app = UniformApp(app_states, utilities)
        elif "normal_app" == app_type:
            app = NormalApp(app_states, utilities)
        elif "queue_app" == app_type:
            arrival_tps = config["arrival_tps"]
            sprinting_tps = config["sprinting_tps"]
            nominal_tps = config["nominal_tps"]
            max_queue_length = config["max_queue_length"]
            app = QueueApp(utilities, arrival_tps, sprinting_tps, nominal_tps, max_queue_length)
        if "ac_policy" == policy_type:
            lr_actor = config["lr_actor"]
            lr_critic = config["lr_critic"]
            l1_in_actor = config["l1_in_actor"]
            l1_in_critic = config["l1_in_critic"]
            l1_out_l2_in_actor = config["l1_out_l2_in_actor"]
            l1_out_l2_in_critic = config["l1_out_l2_in_critic"]
            policy = ACPolicy(l1_in_actor, l1_in_critic, l1_out_l2_in_actor, l1_out_l2_in_critic, app_type)
            # Create an Adam optimizer for the actor
            if app_type == "queue_app":
                actor_params = [policy.actor_layer1.weight, policy.actor_layer1.bias,
                                policy.actor_layer2_mean.weight, policy.actor_layer2_mean.bias,
                                policy.actor_layer2_std.weight, policy.actor_layer2_std.bias]
            else:
                actor_params = [policy.actor_layer1.weight, policy.actor_layer1.bias,
                                policy.actor_layer2.weight, policy.actor_layer2.bias]

            actor_optimizer = optim.Adam(actor_params, lr=lr_actor)

            # Create an Adam optimizer for the critic
            critic_params = [policy.critic_layer1.weight, policy.critic_layer1.bias,
                            policy.critic_layer2.weight, policy.critic_layer2.bias]

            critic_optimizer = optim.Adam(critic_params, lr=lr_critic)
            optimizer = [actor_optimizer, critic_optimizer]
            server = AC_server(i, policy, optimizer, app, path, servers_config, num_servers, servers_per_worker, game_type)
        elif "thr_policy" == policy_type:
            policy = ThrPolicy(threshold)
            server = Thr_server(i, policy, app, path, servers_config, num_servers, servers_per_worker, game_type)
        elif "q_learning" == policy_type:
            lr_q = config["lr_q"]
            l1_in_q = config["l1_in_q"]
            l1_out_l2_in_q = config["l1_out_l2_in_q"]
            dqn = QNetwork(l1_in_q, l1_out_l2_in_q)
            target_dqn = QNetwork(l1_in_q, l1_out_l2_in_q)
            optimizer = optim.Adam(dqn.parameters(), lr_q)
            server = Q_server(i, dqn, target_dqn, optimizer, app, path, servers_config, num_servers, servers_per_worker, game_type)
        servers.append(server)
        policies.append(policy)
    
    split_servers = np.array_split(servers, num_workers)
    for i, s in enumerate(split_servers):
        worker = Worker(s, w2c_queues[i], c2w_queues[i])
        worker_processor = Process(target=worker.run_worker)
        worker_processors.append(worker_processor)
        worker_processor.start()

    coordinator_processor = Process(target=coordinator.run_coordinator)
    coordinator_processor.start()

    for worker_processor in worker_processors:
        worker_processor.join()
    
    coordinator_processor.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total running time: {total_time} seconds")
    

if __name__ == "__main__":
    config_file = "/Users/jingyiwu/Desktop/MARL/config.json"
    main(config_file)

    