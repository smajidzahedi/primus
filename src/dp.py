import json
import sys

import numpy as np
from scipy.stats import skellam


server_state = np.array([0, 1])
server_state_len = 2
error = 0.0001


def cost(fs, min_frac, max_frac):
    return min(max((fs - min_frac) / (max_frac - min_frac), 0), 1)


class App:

    def get_app_state_len(self) -> int:
        pass

    def get_sprinting_utility(self, index) -> float:
        pass

    def get_nominal_utility(self, index) -> float:
        pass

    def get_tran_prob(self) -> np.ndarray:
        pass

    def calculate_app_state_probs(self, action, prob_cooling):
        dim = (server_state_len, self.get_app_state_len())
        p = np.ones(dim) / (server_state_len * self.get_app_state_len())
        difference = 1
        trans = self.get_tran_prob()
        while difference > error:
            new_p = np.zeros(dim)

            for s2 in range(self.get_app_state_len()):
                for s1 in range(self.get_app_state_len()):
                    new_p[0][s2] += p[0][s1] * action[s1] * trans[s2][s1][int(action[s1])]
                    new_p[0][s2] += p[1][s1] * (1 - prob_cooling) * trans[s2][s1][1]

                    new_p[1][s2] += p[0][s1] * (1 - action[s1]) * trans[s2][s1][int(action[s1])]
                    new_p[1][s2] += p[1][s1] * prob_cooling * trans[s2][s1][1]

            difference = np.sqrt(((p - new_p) ** 2).sum())
            p = new_p.copy()

        return p


class Uniform(App):
    def __init__(self, app_utilities):
        self.app_state = app_utilities
        self.app_state_len = len(app_utilities)
        dim = (self.app_state_len, self.app_state_len, 2)
        self.tran_prob = np.ones(dim) * (1 / self.app_state_len)

    def get_app_state_len(self):
        return self.app_state_len

    def get_sprinting_utility(self, index):
        return self.app_state[index]

    def get_nominal_utility(self, index):
        return 0

    def get_tran_prob(self):
        return self.tran_prob


class Markov(App):
    def __init__(self, app_utilities, transition_matrix, utility_normalization_factor):
        self.app_state = app_utilities
        self.app_state_len = len(app_utilities)
        self.utility_normalization_factor = utility_normalization_factor
        dim = (self.app_state_len, self.app_state_len, 2)
        self.tran_prob = np.zeros(dim)
        for i in range(self.app_state_len):
            for j in range(self.app_state_len):
                self.tran_prob[j][i][0] = transition_matrix[i][j]
                self.tran_prob[j][i][1] = transition_matrix[i][j]

    def get_app_state_len(self):
        return self.app_state_len

    def get_sprinting_utility(self, index):
        return self.utility_normalization_factor * self.app_state[index]

    def get_nominal_utility(self, index):
        return 0

    def get_tran_prob(self):
        return self.tran_prob


class Queue(App):
    def __init__(self, arrival_tps, sprinting_tps, nominal_tps, max_queue_length, utility_normalization_factor):
        self.app_state = np.arange(max_queue_length)
        self.app_state_len = max_queue_length
        self.arrival_tps = arrival_tps
        self.nominal_tps = nominal_tps
        self.sprinting_tps = sprinting_tps
        self.utility_normalization_factor = utility_normalization_factor
        dim = (self.app_state_len, self.app_state_len, 2)
        self.tran_prob = np.zeros(dim)
        for i in range(self.app_state_len):
            for j in range(self.app_state_len):
                probability_s = skellam.pmf(j - i, arrival_tps, sprinting_tps)
                probability_ns = skellam.pmf(j - i, arrival_tps, nominal_tps)
                self.tran_prob[j][i][0] += probability_s
                self.tran_prob[j][i][1] += probability_ns

            probability_e_s = skellam.cdf(- i - 1, arrival_tps, sprinting_tps)
            probability_e_ns = skellam.cdf(- i - 1, arrival_tps, nominal_tps)
            self.tran_prob[0][i][0] += probability_e_s
            self.tran_prob[0][i][1] += probability_e_ns

            probability_f_s = skellam.sf(self.app_state_len - i - 1, arrival_tps, sprinting_tps)
            probability_f_ns = skellam.sf(self.app_state_len - i - 1, arrival_tps, nominal_tps)
            self.tran_prob[-1][i][0] += probability_f_s
            self.tran_prob[-1][i][1] += probability_f_ns

        # print(self.tran_prob.sum(axis=0))

    def get_app_state_len(self):
        return self.app_state_len

    def get_sprinting_utility(self, index):
        new_queue_length = max(0, self.app_state[index] + self.arrival_tps - self.sprinting_tps)
        return - self.utility_normalization_factor * min(new_queue_length, self.app_state_len - 1)

    def get_nominal_utility(self, index):
        new_queue_length = max(0, self.app_state[index] + self.arrival_tps - self.nominal_tps)
        return - self.utility_normalization_factor * min(new_queue_length, self.app_state_len - 1)

    def get_tran_prob(self):
        return self.tran_prob


def run_dp(config_file_name, app_type_id, app_sub_type_id):
    with open(config_file_name, 'r') as f:
        config = json.load(f)

    min_frac = config["coordinator_config"]["min_frac"]
    max_frac = config["coordinator_config"]["max_frac"]
    app_type = config["app_types"][app_type_id]
    app_sub_type = config["app_sub_types"][app_type][app_sub_type_id]
    discount_factor = config["ac_discount_factor"][app_type][app_sub_type]
    prob_cooling = config["servers_config"]["cooling_prob"]
    app_type = config["app_types"][app_type_id]
    app_sub_type = config["app_sub_types"][app_type][app_sub_type_id]
    add_change = config["servers_config"]["change"]
    if add_change == 1:
        error_1 = config["dp_error_change"][app_type][app_sub_type]
    else:
        error_1 = config["dp_error"][app_type][app_sub_type]
    print(error_1)

    app_utilities = config["app_utilities"]

    if app_type == "uniform":
        app = Uniform(app_utilities)
    elif app_type == "markov":
        utility_normalization_factor = config["utility_normalization_factor"][app_type][app_sub_type]
        transition_matrix = config["markov_app_transition_matrices"][app_sub_type]
        app = Markov(app_utilities, transition_matrix, utility_normalization_factor)
    elif app_type == "queue":
        if add_change == 1:
            arrival_tps = config["queue_app_arrival_tps_change"][app_sub_type]
        else:
            arrival_tps = config["queue_app_arrival_tps"][app_sub_type]
        sprinting_tps = config["queue_app_sprinting_tps"][app_sub_type]
        nominal_tps = config["queue_app_nominal_tps"][app_sub_type]
        utility_normalization_factor = config["utility_normalization_factor"][app_type][app_sub_type]
        app = Queue(arrival_tps, sprinting_tps, nominal_tps, 20, utility_normalization_factor)
        # sys.exit()
    else:
        sys.exit("App model is not supported")

    trans = app.get_tran_prob()
    app_state_len = app.get_app_state_len()
    dim = (server_state_len, app_state_len)
    #v = np.random.rand(server_state_len, app_state_len)
    v = np.zeros(dim)
    new_v = np.zeros(dim)
    actions = np.ones(app_state_len)
    q_s = np.zeros(app_state_len)
    q_ns = np.zeros(app_state_len)
    frac_sprinters = 0
    avg_reward = 0

    itr = 0
    diff = 10
    while diff > error_1:
        itr += 1
        total_cost = cost(frac_sprinters, min_frac, max_frac)

        for s1 in range(app_state_len):

            next_active_ns = 0
            next_inactive_s = 0
            next_inactive_ns = 0

            for s2 in range(app_state_len):
                next_active_ns += trans[s2][s1][1] * v[0][s2]
                next_inactive_s += trans[s2][s1][0] * v[1][s2]
                next_inactive_ns += trans[s2][s1][1] * v[1][s2]

            q_s[s1] = app.get_sprinting_utility(s1) - total_cost + discount_factor * next_inactive_s
            q_ns[s1] = app.get_nominal_utility(s1) - total_cost + discount_factor * next_active_ns

            new_v[1][s1] = app.get_nominal_utility(s1) - total_cost
            new_v[1][s1] += discount_factor * (prob_cooling * next_inactive_ns + (1 - prob_cooling) * next_active_ns)

            if q_s[s1] > q_ns[s1]:
                new_v[0][s1] = q_s[s1]
                actions[s1] = 0
            else:
                new_v[0][s1] = q_ns[s1]
                actions[s1] = 1

        probs = app.calculate_app_state_probs(actions, prob_cooling)
        frac_sprinters = 0
        avg_reward = - total_cost
        for s1 in range(app_state_len):
            frac_sprinters += probs[0][s1] * (1 - actions[s1])
            avg_reward += probs[0][s1] * (1 - actions[s1]) * app.get_sprinting_utility(s1)
            avg_reward += probs[0][s1] * actions[s1] * app.get_nominal_utility(s1)
            avg_reward += probs[1][s1] * app.get_nominal_utility(s1)

        diff = np.sqrt(((new_v - v) ** 2).sum())
        if itr % 500 == 0:
            print(itr)
            # print(probs.sum(0))
            # print(probs.sum(1))
            # print(actions)
            print("total costs", total_cost)
            print("avg rewards", avg_reward)
            print("frac_sprinters", frac_sprinters)
            print("main loop diff", diff)
        v = new_v.copy()

    print(v)
    print(itr)
    print(frac_sprinters)
    print(actions)
    print(avg_reward)


if __name__ == "__main__":
    # config_file = "/Users/smzahedi/Documents/Papers/MARL/configs/config.json"
    config_file = "/Users/jingyiwu/Documents/Project/MARL/configs/config.json"
    run_dp(config_file, 1, 5)

