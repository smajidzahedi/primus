import os
import numpy as np


class App:
    def __init__(self):
        self.app_state_history = []
        self.current_state = None

    def apply_change(self, change_type):
        pass

    def get_state_space_len(self):
        pass

    def get_current_state_index(self):
        pass

    def get_sprinting_utility(self):
        raise NotImplementedError("This method should be overridden.")

    def get_nominal_utility(self):
        raise NotImplementedError("This method should be overridden.")

    def get_current_state(self):
        return float(self.current_state)

    def update_state(self, action):
        self.app_state_history.append(self.get_current_state())

    def print_state(self, server_id, path):
        file_path = os.path.join(path, f"server_{server_id}_app_states.txt")
        with open(file_path, 'w') as file:
            for h in self.app_state_history:
                file.write(f"{str(h)}\n")


"""
State transition follows Markov process
"""


class MarkovApp(App):
    def __init__(self, transition_matrix, utilities, initial_state):
        super().__init__()
        self.transition_matrix = transition_matrix
        self.utilities = utilities
        self.current_state = initial_state

    def get_state_space_len(self):
        return len(self.utilities)

    def get_current_state_index(self):
        return self.utilities.index(self.current_state)

    def get_sprinting_utility(self):
        return self.current_state

    def get_nominal_utility(self):
        return 0

    def update_state(self, action):
        super().update_state(action)
        probabilities = self.transition_matrix[self.utilities.index(self.current_state)]
        self.current_state = np.random.choice(self.utilities, p=probabilities)


"""
State transition follows Uniform distribution
"""


class UniformApp(App):
    def __init__(self, utilities):
        super().__init__()
        self.utilities = utilities
        self.current_state = np.random.choice(self.utilities)

    def get_state_space_len(self):
        return len(self.utilities)

    def get_current_state_index(self):
        return self.utilities.index(self.current_state)

    def get_sprinting_utility(self):
        return self.current_state

    def get_nominal_utility(self):
        return 0

    def update_state(self, action):
        super().update_state(action)
        self.current_state = np.random.choice(self.utilities)


"""
Queue application uses Poisson distribution to model events that occur randomly and independently.
"""


class QueueApp(App):
    def __init__(self, arrival_tps, sprinting_tps, nominal_tps, max_queue_length=1000):
        super().__init__()
        self.current_state = 0
        self.current_queue_length = 0
        self.arrival_tps = arrival_tps
        self.sprinting_tps = sprinting_tps
        self.nominal_tps = nominal_tps
        self.max_queue_length = max_queue_length
        self.next_departure_not_sprinting = nominal_tps
        self.next_departure_sprinting = sprinting_tps
        self.next_arrival = arrival_tps

    def get_state_space_len(self):
        return self.max_queue_length + 1

    def get_current_state_index(self):
        return self.current_state

    def get_sprinting_utility(self):
        new_queue_length = max(0, self.current_queue_length + self.next_arrival - self.next_departure_sprinting)
        # return -min(new_queue_length, self.max_queue_length)
        return - new_queue_length

    def get_nominal_utility(self):
        new_queue_length = max(0, self.current_queue_length + self.next_arrival - self.next_departure_not_sprinting)
        # return -min(new_queue_length, self.max_queue_length)
        return - new_queue_length

    def update_state(self, action):
        super().update_state(action)
        arrived_tasks = self.next_arrival
        departed_tasks = self.next_departure_not_sprinting
        if action == 0:
            departed_tasks = self.next_departure_sprinting
        self.current_queue_length = max(0, self.current_queue_length + arrived_tasks - departed_tasks)
        self.current_state = min(self.current_queue_length, self.max_queue_length)
        self.next_arrival = np.random.poisson(self.arrival_tps)
        self.next_departure_not_sprinting = np.random.poisson(self.nominal_tps)
        self.next_departure_sprinting = np.random.poisson(self.sprinting_tps)

    def apply_change(self, change_type):
        if change_type == 0:
            self.arrival_tps *= 1.3
        elif change_type == 1:
            self.nominal_tps /= 1.5
            self.sprinting_tps /= 1.5


class SparkApp(App):
    def __init__(self, gains, initial_index):
        super().__init__()
        self.gains = gains
        self.max_gain = np.array(gains).max()
        self.current_state = self.gains[initial_index]
        self.current_index = initial_index
        self.state_space_len = len(self.gains)

    def get_state_space_len(self):
        return self.state_space_len

    def get_current_state_index(self):
        return self.current_index

    def get_sprinting_utility(self):
        return self.current_state / self.max_gain

    def get_nominal_utility(self):
        return 0

    def update_state(self, action):
        super().update_state(action)

        if self.current_index == self.state_space_len - 1:
            self.current_index = 0
        elif self.current_index < self.state_space_len - 1:
            self.current_index += 1

        self.current_state = self.gains[self.current_index]
