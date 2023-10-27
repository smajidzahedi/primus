from multiprocessing import Process, Queue
import numpy as np
import torch
from torch import nn, optim
import time
import os
from tqdm import tqdm
import json

import applications
import policies
import servers


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)


"""
An independent processor. Get information from each server, 
and send rack state and frac_sprinters back to each of them. 
Determining system trip or not.
"""
class Coordinator:
    def __init__(self, coordinator_config,
                 w2c_queues, c2w_queues,
                 path,
                 num_workers,
                 num_servers, app_type, policy_type, threshold):

        # Sprinters parameters
        self.num_sprinters = 0  # Initialize num_sprinting
        self.frac_sprinters = 0
        self.avg_frac_sprinters = 0 # initial value for exponential moving average of num_sprinting
        self.sprinters_decay_factor = coordinator_config["sprinters_decay_factor"]  #   for fictitious play

        # Iteration parameters
        self.total_active_iterations = coordinator_config["total_active_iterations"]
        self.num_active_iterations = 0 # store total number of active rounds
        self.num_all_active_iterations = 0

        # Worker parameters
        self.num_workers = num_workers
        self.w2c_queues = w2c_queues
        self.c2w_queues = c2w_queues

        # Server parameters
        self.num_servers = num_servers
        self.app_type = app_type
        self.policy_type = policy_type
        self.threshold = threshold  # Given threshold value, when policy_type == Thr_Policy

        # Recovery parameters
        self.in_recovery = False    # flag for recovery
        self.recovery_prob = coordinator_config["recovery_prob"]    # prob. for staying recovery state
        self.min_frac = coordinator_config["min_frac"]  # lower bound of sprinters of system trip
        self.max_frac = coordinator_config["max_frac"]  # higher bound of sprinters of system trip
        self.release_frac = coordinator_config["release_frac"]  #   prob. of servers that can leave recovery state
        self.rack_states = np.zeros(self.num_servers)

        # Loging parameters
        self.path = path

        # Privacy parameters
        self.c_epsilon = coordinator_config["c_epsilon"]
        self.c_delta = coordinator_config["c_delta"]
        self.epsilon = self.c_epsilon / np.log10(self.num_servers)
        self.epsilon_prime = self.epsilon / self.total_active_iterations
        self.delta = self.c_delta / self.num_servers
        self.alpha = 1 + 2 * np.log10(1 / self.delta) / self.epsilon
        self.var = self.alpha / (2 * self.num_servers ** 2 * self.epsilon_prime)
        self.sigma = np.sqrt(self.var)
        self.add_noise = coordinator_config["add_noise"]

    #   Whether system trips or not
    def is_tripped(self):
        prob = min(max((np.mean(self.frac_sprinters.tolist()) - self.min_frac) / (self.max_frac - self.min_frac), 0), 1)
        return np.random.rand() < prob

    # Calculate number of sprinters in this round, determining whether system trip or not.
    # Calculate the fractional number of sprinters by Bias-Corrected Exponential Weighted Moving Average
    # Add noise on the fraction number of sprinters in this round (# of sprinters / total # of servers)
    def aggregate_actions(self, actions):
        self.num_sprinters = self.num_servers - actions.sum()

        if self.in_recovery:
            assert self.num_sprinters == 0
            if np.random.rand() > self.recovery_prob:   #   whether servers are allowed start to leave recovery state
                self.in_recovery = False
        else:
            self.num_active_iterations += 1
            self.in_recovery = self.is_tripped()

            self.avg_frac_sprinters *= self.sprinters_decay_factor
            self.avg_frac_sprinters += (1 - self.sprinters_decay_factor) * (self.num_sprinters / self.num_servers)

            if self.add_noise == 1:
                self.avg_frac_sprinters += (1 - self.sprinters_decay_factor) * np.random.normal(loc=0, scale=self.sigma)

            self.frac_sprinters = self.avg_frac_sprinters / (1 - self.sprinters_decay_factor ** self.num_active_iterations)

        # release servers from recovery state randomly
        if self.in_recovery:
            self.rack_states = np.ones(self.num_servers)
        elif np.count_nonzero(self.rack_states) >= self.num_servers * self.release_frac: # randomly select servers leave recovery state
            recovers = np.random.choice(np.where(self.rack_states == 1)[0], size=int(self.num_servers * self.release_frac), replace=False)
            self.rack_states[recovers] = 0
        else:   #   all servers leave recovery state
            self.rack_states = np.zeros(self.num_servers)
            self.num_all_active_iterations += 1

    # Main function for coordinator
    def run_coordinator(self):
        actions_array = np.zeros(self.num_servers)
        frac_sprinters_list = []
        iteration = 0

        while self.num_all_active_iterations < self.total_active_iterations:
            # Split the array into self.num_workers parts
            reshaped_states = np.array_split(self.rack_states, self.num_workers)

            # Now, iterate over the queues and reshaped states
            for q, states in zip(self.c2w_queues, reshaped_states):
                q.put((self.frac_sprinters, states, iteration))

            # get information from workers
            for q in self.w2c_queues:
                info = q.get()
                server_ids, actions = info
                for j, index in enumerate(server_ids):
                    actions_array[index] = actions[j]
            self.aggregate_actions(actions_array)
            frac_sprinters_list.append(self.frac_sprinters)
            iteration += 1
        self.num_recovery_rounds = num_recovery_iterations
        self.num_active_iterations = num_active_iterations
        print(f"total active round: {self.num_active_iterations}")
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
                file.write(f"{self.policy_type}_{str(self.threshold)}_{self.app_type}\t{self.num_recovery_rounds}\n")
            else:
                file.write(f"{self.policy_type}_{self.game_type}_{self.app_type}\t{self.num_recovery_rounds}\n")

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
    c_epsilon = config["c_epsilon"]
    c_delta = config["c_delta"]
    path = f"{folder_name}/{num_servers}_server/{app_type}"
    if not os.path.exists(path):
        os.makedirs(path)
    w2c_queues = [Queue() for _ in range(num_workers)]
    c2w_queues = [Queue() for _ in range(num_workers)]
    servers = []
    worker_processors = []
    servers_per_worker = num_servers // num_workers
    coordinator = Coordinator(coordinator_config, w2c_queues, c2w_queues, path, num_workers, num_servers, game_type, app_type, policy_type, threshold, c_epsilon, c_delta)
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

    