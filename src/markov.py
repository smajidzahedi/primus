import numpy as np


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

    print(calculate_app_state_probs(state_space_len, m1))
    print(calculate_app_state_probs(state_space_len, m2))
    print(calculate_app_state_probs(state_space_len, m3))
    print(calculate_app_state_probs(state_space_len, m4))

    print("============")

    print(repr(m1))
    print(repr(m2))
    print(repr(m3))
    print(repr(m4))

