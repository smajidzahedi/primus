import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

config_file_name = "/Users/jingyiwu/Documents/Project/MARL/configs/config.json"
with open(config_file_name, 'r') as f:
    config = json.load(f)

decay_factor = 0.999
app_type_id = 2
app_type_sub_id = 1
policy_id = 0
num_servers = config["num_servers"]
folder_name = config["folder_name"]
num_servers = config["num_servers"]
app_type = config["app_types"][app_type_id]
app_sub_type = config["app_sub_types"][app_type][app_type_sub_id]
policy_type = config["policy_types"][policy_id]
path = f"{folder_name}/{num_servers}_server/{policy_type}/{app_type}_{app_sub_type}"
data_server_0 = []
data_server_1 = []
data_server_2 = []

for server_id in [2, 56, 75]:
    itr = 1
    avg_length = 0
    y = []  # Initialize the y-axis
    file_path = os.path.join(path, f"server_{server_id}_app_states.txt")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        avg_length = decay_factor * avg_length + (1 - decay_factor) * float(line.strip())
        if (itr) % 10 == 0:
            y.append(avg_length / (1 - decay_factor ** itr))
        itr += 1
    if server_id == 2:
        data_server_0 = y
    elif server_id == 56:
        data_server_1 = y
    elif server_id == 75:
        data_server_2 = y

with open(os.path.join("/Users/jingyiwu/Documents/Project/MARL_PAPER/asplos24/data", f"q{app_type_sub_id+1}_length_change.txt"), "w") as file:
    for i in range(len(data_server_0)):  # Assuming all lists have the same length
        file.write(f"{data_server_0[i]}\t{data_server_1[i]}\t{data_server_2[i]}\n")
