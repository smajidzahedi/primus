import matplotlib.pyplot as plt
import os
import numpy as np
import json
import argparse

def main(config_file_name, app_type_id, app_type_sub_id, policy_id):
    with open(config_file_name, 'r') as f:
        config = json.load(f)

    num_servers = config["num_servers"]
    folder_name = config["folder_name"]
    num_servers = config["num_servers"]
    app_type = config["app_types"][app_type_id]
    app_sub_type = config["app_sub_types"][app_type][app_type_sub_id]
    policy_type = config["policy_types"][policy_id]
    path = f"{folder_name}/{num_servers}_server/{policy_type}/{app_type}_{app_sub_type}"
    if not os.path.exists(path):
        os.mkdir(path)
    frac_sprinters_path = os.path.join(path, "frac_sprinters.txt")
    frac_sprinters = np.loadtxt(frac_sprinters_path)
    total_iter = frac_sprinters.shape[0]
    plt.figure(figsize=(25, 10))
    plt.plot(frac_sprinters)
    plt.xlabel("Iterations")
    plt.ylabel("Fractional Sprinters")
    plt.title("Fractional Sprinters Over Iterations")
    plt.grid(True)
    plt.savefig(os.path.join(path, "frac_sprinter.png"))
    plt.close()

    rewards_from_servers = np.empty([num_servers, total_iter])
    for i in range(num_servers):
        reward_file_path = os.path.join(path, f"server_{i}_rewards.txt")
        rewards = np.loadtxt(reward_file_path)
        rewards_from_servers[i] = rewards
    mean_rewards = rewards_from_servers.mean(axis=0)

    plt.figure(figsize=(25, 10))
    plt.plot(mean_rewards)
    plt.xlabel('Iterations')
    plt.ylabel('Mean reward')
    plt.title('Mean Reward Over Time Across All Servers')
    plt.grid(True)
    plt.savefig(os.path.join(path, "mean_rewards.png"))
    plt.close()

    #   calculate average reward over rounds for different policy 
    if policy_id == 0:
        avg_reward = mean_rewards[len(mean_rewards)-1000:].mean()
    else:
        avg_reward = mean_rewards.mean()

    file_path = os.path.join(path, "different_policy_avg_rewards.txt")
    with open(file_path, 'w+') as file:
        file.write(f"{avg_reward}\n")

    if app_type_id == 2:
        # Iterate over all 10 servers
        x = list(range(1, total_iter+1))
        plt.figure(figsize=(40, 20)) 
        for server_id in range(num_servers):
            y = []  # Initialize the y-axis
            file_path = os.path.join(path, f"server_{server_id}_app_states.txt")
            with open(file_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                y.append(float(line.strip()))  # Convert each line to a float and append to the y-axis
            plt.plot(x, y, label=f'Server {server_id}')
        x_ticks = list(range(0, total_iter, 10000))
        x_tick_labels = [str(tick) for tick in x_ticks]  # Convert tick values to strings
        plt.xticks(x_ticks, x_tick_labels, rotation=45)
        plt.legend(ncol=2, bbox_to_anchor=(1, 1))
        plt.xlabel('Number of Iterations')
        plt.ylabel('Queue Length')
        plt.title('Queue Length Over Iterations for Each Server')
        plt.grid(True)
        plt.savefig(os.path.join(path, f"queue_length.png"))
        plt.close()
        
    return avg_reward

if __name__ == "__main__":
    """parser = argparse.ArgumentParser(description="Run MARL with specified parameters.")
    parser.add_argument("app_type_id", type=int, help="App type ID.")
    parser.add_argument("app_type_sub_id", type=int, help="App type sub ID.")
    parser.add_argument("policy_id", type=int, help="Policy ID.")
    args = parser.parse_args()"""

    config_file = "/Users/jingyiwu/Desktop/MARL/configs/config.json"
    
    avg_reward = main(config_file, 2, 0, 1)
    print(f"Average reward: {avg_reward}")
