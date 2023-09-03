import matplotlib.pyplot as plt
import os
import numpy as np
import json

def main():
    with open('/Users/jingyiwu/Desktop/MARL/config.json', 'r') as f:
        config = json.load(f)

    coordinator_config = config["coordinator_config"]
    num_servers = config["num_servers"]
    num_iterations = coordinator_config["num_iterations"]
    folder_name = config["folder_name"]
    game_type = config["game_type"] 
    utilities = config["utilities"]
    num_servers = config["num_servers"]
    app_type = config["app_type"]
    location = f"{folder_name}/{num_servers}_server/{app_type}"   
    policy = ""
    threshold = config["threshold"]
    policy = config["policy_type"]
    frac_sprinters_path = os.path.join(location, "frac_sprinters.txt")
    frac_sprinters = np.loadtxt(frac_sprinters_path)
    plt.figure(figsize=(25, 10))
    plt.plot(frac_sprinters)
    plt.xlabel("Iterations")
    plt.ylabel("Fractional Sprinters")
    plt.title("Fractional Sprinters Over Iterations")
    plt.grid(True)
    if policy == "thr_policy":
        plt.savefig(os.path.join(location, f"{app_type}_{policy}_frac_sprinter.png"))
    else:
        plt.savefig(os.path.join(location, f"{game_type}_{app_type}_{policy}_frac_sprinter.png"))
    plt.close()

    rewards_from_servers = np.empty([num_servers, num_iterations])
    policies_from_servers = np.empty([num_servers, len(utilities)])
    for i in range(num_servers):
        reward_file_path = os.path.join(location, f"server_{i}_reward.txt")
        rewards = np.loadtxt(reward_file_path)
        rewards_from_servers[i] = rewards
        if policy == "ac_policy":
            policy_file_path = os.path.join(location, f"server_{i}_policy.txt")
            policies = np.loadtxt(policy_file_path)[:,1]
            policies_from_servers[i] = policies
    mean_rewards = rewards_from_servers.mean(axis=0)
    mean_policies = policies_from_servers.mean(axis=0)

    plt.figure(figsize=(25, 10))
    plt.plot(mean_rewards)
    plt.xlabel('Iterations')
    plt.ylabel('Mean reward')
    plt.title('Mean Reward Over Time Across All Servers')
    plt.grid(True)
    if policy == "thr_policy":
        plt.savefig(os.path.join(location, f'{app_type}_{policy}_mean_rewards.png'))
    else:
        plt.savefig(os.path.join(location, f'{game_type}_{app_type}_{policy}_mean_rewards.png'))
    plt.close()

    plt.figure(figsize=(25, 10))
    plt.scatter(utilities, mean_policies)
    for i, txt in enumerate(mean_policies):
        plt.annotate(f"{txt:.4f}", (utilities[i], mean_policies[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.xlabel('Utility')
    plt.ylabel('Threshold')
    plt.title('Selected Theshold for Different Utilities')
    plt.grid(True)
    if policy == "ac_policy":
        plt.savefig(os.path.join(location, f'{game_type}_{app_type}_{policy}_policy.png'))
    plt.close()

    #   calculate average reward over rounds for different policy 
    if policy == "ac_policy" or policy == "q_learning":
        avg_reward = mean_rewards[len(mean_rewards)-1000:].mean()
    else:
        avg_reward = mean_rewards.mean()

    file_path = os.path.join(location, f"{app_type}_different_policy_avg_rewards.txt")
    with open(file_path, 'a+') as file:
        if policy == "ac_policy" or policy == "q_learning":
            file.write(f"{policy}_{game_type}_{app_type}\t{avg_reward}\n")
        else:
            file.write(f"{policy}_{str(threshold)}_{app_type}\t{avg_reward}\n")

    if app_type == "queue_app":
        # Iterate over all 10 servers
        x = list(range(1, num_iterations+1))
        plt.figure(figsize=(40, 20)) 
        for server_id in range(num_servers):
            y = []  # Initialize the y-axis
            file_path = os.path.join(location, f"server_{server_id}_app_state.txt")
            with open(file_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                y.append(float(line.strip()))  # Convert each line to a float and append to the y-axis
            plt.plot(x, y, label=f'Server {server_id}')
        x_ticks = list(range(0, num_iterations, 10000))
        x_tick_labels = [str(tick) for tick in x_ticks]  # Convert tick values to strings
        plt.xticks(x_ticks, x_tick_labels, rotation=45)
        plt.legend(ncol=2, bbox_to_anchor=(1, 1))
        plt.xlabel('Number of Iterations')
        plt.ylabel('Queue Length')
        plt.title('Queue Length Over Iterations for Each Server')
        plt.grid(True)
        if policy == "thr_policy":
            plt.savefig(os.path.join(location, f'{app_type}_{policy}_queue_length.png'))
        else:
            plt.savefig(os.path.join(location, f'{game_type}_{app_type}_{policy}_queue_length.png'))
        plt.close()
    return avg_reward

if __name__ == "__main__":
    avg_reward = main()
    print(f"Average reward: {avg_reward}")