import numpy as np
import torch

from src import applications, policies, multiprocessing_MARL

multiprocessing_MARL.set_seed(42)


def calculate_app_state_probs(server_state_len, trans):
    p = np.ones(server_state_len) / server_state_len
    difference = 1
    while difference > 0.0001:
        new_p = np.zeros(server_state_len)
        for s2 in range(server_state_len):
            for s1 in range(server_state_len):
                new_p[s2] += p[s1] * trans[s1][s2]

        difference = ((p - new_p) ** 2).sum()
        p = new_p.copy()

    return p


def calculate_app_state_values(server_state_len, trans, df):
    p = np.zeros(server_state_len)
    difference = 10
    while difference > 0.001:
        new_p = np.zeros(server_state_len)
        for s1 in range(server_state_len):
            new_p[s1] = (s1 + 1)/server_state_len
            for s2 in range(server_state_len):
                new_p[s1] += df * p[s2] * trans[s1][s2]
        difference = ((p - new_p) ** 2).sum()
        # print(difference)
        p = new_p.copy()

    return p


def compute_returns(df, rs, next_state_v):
    r = next_state_v
    rts = []
    for step in reversed(range(len(rs))):
        rts.insert(0, rs[step] + df * r)
    return rts


if __name__ == "__main__":
    state_space_len = 10
    m1 = np.zeros((state_space_len, state_space_len))
    m2 = np.zeros((state_space_len, state_space_len))
    m3 = np.zeros((state_space_len, state_space_len))
    m4 = np.zeros((state_space_len, state_space_len))
    for i in range(state_space_len):
        for j in range(state_space_len):
            m1[i][j] = (np.abs(j - i) + 1)
            m2[i][j] = 1 / (np.abs(j - i) + 1)
            m3[i][j] = 1 / (j + 1)
            m4[i][j] = (j + 1) ** 2

    m1 = (m1.transpose() / m1.sum(1)).transpose()
    m2 = (m2.transpose() / m2.sum(1)).transpose()
    m3 = (m3.transpose() / m3.sum(1)).transpose()
    m4 = (m4.transpose() / m4.sum(1)).transpose()

    # print(calculate_app_state_probs(state_space_len, m2))
    # print(calculate_app_state_probs(state_space_len, m3))
    # print(calculate_app_state_probs(state_space_len, m4))

    # print("============")

    # print(repr(m1))
    # print(repr(m2))
    # print(repr(m3))
    # print(repr(m4))
    tran_matrix = m1
    normalization_factor = 0.00001
    app = applications.MarkovApp(tran_matrix, (np.arange(0, 1, 1/state_space_len) + 1/state_space_len).tolist(),
                                 1/state_space_len)
    critic = policies.Critic(1, 128, 0.01)
    discount_factor = 0.9999
    state = normalization_factor * np.array([app.get_current_state()])
    state_t = torch.tensor(state, dtype=torch.float32)
    state_value = critic(state_t)
    values = []
    rewards = []
    iteration = 0

    for i in range(1, 5000):
        reward = app.get_sprinting_utility()
        app.update_state(0)
        # print(reward, app.current_state)
        next_state = normalization_factor * np.array([app.get_current_state()])
        values.append(state_value)
        rewards.append(reward)

        next_state_t = torch.tensor(next_state, dtype=torch.float32)
        iteration += 1

        if iteration == 5:
            next_state_value = critic(next_state_t)
            returns = compute_returns(discount_factor, rewards, next_state_value)

            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            advantage = returns - values

            loss = advantage.pow(2).mean()

            critic.optimizer.zero_grad()
            loss.backward()
            critic.optimizer.step()

            rewards = []
            values = []
            iteration = 0

        state_value = critic(next_state_t)

    v = calculate_app_state_values(state_space_len, tran_matrix, discount_factor)
    print(v)

    v = []
    for i in np.arange(0, 1, 1/state_space_len):
        state = torch.tensor([normalization_factor * (i + 1/state_space_len)], dtype=torch.float32)
        v.append(critic(state).item())
    v = np.array(v)

    # print(calculate_app_state_probs(state_space_len, tran_matrix))
    # print(calculate_app_state_values(state_space_len, tran_matrix, discount_factor))
    # print(v)

    for i in range(0, state_space_len):
        print(v[i])
        print((i + 1)/state_space_len + discount_factor * (v * tran_matrix[i]).sum())
