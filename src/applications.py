import os
import numpy as np


class App:
    def __init__(self):
        self.app_state_history = []
        self.current_state = None

    def get_sprinting_utility(self):
        raise NotImplementedError("This method should be overridden.")

    def get_cooling_utility(self):
        return 0

    def get_recovery_utility(self):
        return 0

    def get_current_state(self):
        return float(self.current_state)

    def update_state(self, action):
        self.app_state_history.append(self.current_state)

    def print_state(self, server_id, path):
        file_path = os.path.join(path, f"server_{server_id}_app_states.txt")
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
        super().update_state(action)
        index = self.app_states.index(self.current_state)
        probabilities = self.transition_matrix[index]
        self.current_state = np.random.choice(self.app_states, p=probabilities)

"""
State transition follows Uniform distribution
"""


class UniformApp(App):
    def __init__(self, app_states, utilities):
        super().__init__()
        self.app_states = app_states
        self.utilities = utilities
        self.current_state = np.random.choice(self.app_states)

    def get_sprinting_utility(self):
        index = self.app_states.index(self.current_state)
        return self.utilities[index]

    def update_state(self, action):
        super().update_state(action)
        self.current_state = np.random.choice(self.app_states)


"""
Queue application uses Poisson distribution to model events that occur randomly and independently.
"""


class QueueApp(App):
    def __init__(self, arrival_tps, sprinting_tps, nominal_tps, max_queue_length=1000):
        super().__init__()
        self.current_state = 0
        self.current_state_avg = 0
        self.arrival_tps = arrival_tps
        self.sprinting_tps = sprinting_tps
        self.nominal_tps = nominal_tps
        self.max_queue_length = max_queue_length

    #   current state for queue app is the current queue length in range 0 to 1
    def get_current_state(self):
        return min(self.current_state / self.max_queue_length, 1.0)

    def get_sprinting_utility(self):
        return -min(self.current_state + self.arrival_tps - min(self.current_state, self.sprinting_tps),
                    self.max_queue_length) / self.max_queue_length

    def get_cooling_utility(self):
        return -min(self.current_state + self.arrival_tps - min(self.current_state, self.nominal_tps),
                    self.max_queue_length) / self.max_queue_length

    def get_recovery_utility(self):
        return self.get_cooling_utility()

    def update_state(self, action):
        super().update_state(action)
        arrived_tasks = np.random.poisson(self.arrival_tps)
        departed_tasks = min(self.current_state, np.random.poisson(self.nominal_tps))
        if action == 0:
            departed_tasks = min(self.current_state, np.random.poisson(self.sprinting_tps))
        self.current_state = self.current_state + arrived_tasks - departed_tasks
