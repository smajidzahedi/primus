import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the configuration file to get game_type and app_type
with open('/Users/jingyiwu/Desktop/MARL/config.json', 'r') as f:
    config = json.load(f)

game_type = config['game_type']
app_type = config['app_type']
folder_name = config["folder_name"]
num_servers = config["num_servers"]
game_type = config["game_type"]
app_type = config["app_type"]
location = f"{folder_name}/{num_servers}_server/{app_type}"

# Load data from the files
avg_rewards = pd.read_csv(f"{location}/{app_type}_different_policy_avg_rewards.txt", sep='\t', header=None, names=['policy', 'average_reward'])
num_recovery = pd.read_csv(f"{location}/{app_type}_num_recovery.txt", sep='\t', header=None, names=['policy', 'num_recovery'])

# Merge the dataframes on policy
data = pd.merge(avg_rewards, num_recovery, on='policy', how='outer')

# Set policy as the index
data.set_index('policy', inplace=True)

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(25,22))

labels = data.index.values
width = 0.35

x = np.arange(len(labels))

# Create the bars for 'Average Reward' and add the values at the top
rects1 = ax1.bar(x, data['average_reward'], width, label='Average Reward', color='b')
for i, v in enumerate(data['average_reward']):
    if v >= 0:
        ax1.text(i, v + 0.01, str(round(v, 6)), ha='center')
    else:
        ax1.text(i, v - 0.2, str(round(v, 6)), ha='center')
ax1.set_ylabel('Average Reward')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.legend(loc='lower right')

rects2 = ax2.bar(x, data['num_recovery'], width, label='Number of Recovery', color='r')
for i, v in enumerate(data['num_recovery']):
    ax2.text(i, v + 1, str(round(v, 6)), ha='center')
ax2.set_ylabel('Number of Recovery')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right')
ax2.legend(loc='upper right')
plt.suptitle('Average Reward and Number of Recovery for Different Policies')
plt.savefig(os.path.join(location, f'{app_type}_policy_compare.png'))
