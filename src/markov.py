import numpy as np
import torch

from src import applications, policies, multiprocessing_MARL, servers

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
        difference = np.sqrt(((p - new_p) ** 2).sum())
        # print(difference)
        p = new_p.copy()

    return p


def compute_returns(df, rs, next_state_v):
    r = next_state_v
    rts = []
    for step in reversed(range(len(rs))):
        r = rs[step] + df * r
        rts.insert(0, r)
    return rts


if __name__ == "__main__":
    state_space_len = 10
    m1 = np.zeros((state_space_len, state_space_len))
    m2 = np.zeros((state_space_len, state_space_len))
    m3 = np.zeros((state_space_len, state_space_len))
    m4 = np.zeros((state_space_len, state_space_len))
    m5 = np.zeros((state_space_len, state_space_len))
    for i in range(state_space_len):
        for j in range(state_space_len):
            m1[i][j] = (np.abs(j - i) + 1)
            m2[i][j] = 1 / (np.abs(j - i) + 1)
            m3[i][j] = 1 / (j + 1)
            m4[i][j] = (j + 1) ** 2
            m5[i][j] = 1/state_space_len

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
    normalization_factor = 0.01
    app = applications.MarkovApp(tran_matrix, (np.arange(0, 1, 1/state_space_len) + 1/state_space_len).tolist(),
                                 1/state_space_len)
    a_h1_size = 256
    c_h1_size = 64
    std_max = 0.01
    df = 0.99
    mini_batch_size = 5
    a_lr = 0.001
    c_lr = 0.003
    state_normalization_factor = 0.05
    utility_normalization_factor = 1
    policy = policies.ACPolicy(1, 3, a_h1_size, c_h1_size, a_lr, c_lr, df, std_max, mini_batch_size)
    servers_config = {
        "cooling_prob": 0.5,
        "change": 0,
        "change_iteration": 2000,
        "change_type": 0
    }
    server = servers.ACServer(0, policy, app, servers_config,
                              state_normalization_factor, utility_normalization_factor)

    frac_sprinters = 0.28666103977608953
    cost = min(max((frac_sprinters - 0.25) / 0.5, 0), 1)

    for i in range(1, 5000):
        server.run_server(cost, frac_sprinters, i)

    state = state_normalization_factor * np.array([frac_sprinters])
    mean, std = server.policy.get_mean_std(state)
    print(mean, std)
    rewards = np.array(server.reward_history[-1000:])
    print(rewards.mean())

    # vv = calculate_app_state_values(state_space_len, tran_matrix, discount_factor)
    # print(vv)
    #
    # v = []
    # for i in np.arange(0, 1, 1/state_space_len):
    #     state = torch.tensor([normalization_factor * (i + 1/state_space_len)], dtype=torch.float32)
    #     v.append(critic(state).item())
    # v = np.array(v)
    # print(v)
    # print(v - vv)
    # print(calculate_app_state_probs(state_space_len, tran_matrix))
    # print(calculate_app_state_values(state_space_len, tran_matrix, discount_factor))
    # print(v)

    # for i in range(0, state_space_len):
    #     print(v[i])
    #     print((i + 1)/state_space_len + discount_factor * (v * tran_matrix[i]).sum())
