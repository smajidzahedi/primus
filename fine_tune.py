import optuna
import json
import subprocess 

def objective(trial):
    
    # Load the configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)
    alpha = trial.suggest_categorical("alpha", [2, 4, 6, 8, 10])
    if config["app_type"] == "queue_app":
        config["max_queue_length"] = trial.suggest_categorical("max_queue_length", [500, 1000, 2000, 3000, 4000, 5000])
    if config["policy_type"] == "ac_policy":  
        config['lr_actor'] = trial.suggest_categorical("lr_actor", [1e-4, 2e-4, 3e-4, 
                                                                    4e-4, 5e-4, 6e-4, 
                                                                    7e-4, 8e-4, 9e-4])  
        beta = trial.suggest_categorical("beta", [1.5, 2.0, 2.5, 3.0])
        config['lr_critic'] = config['lr_actor'] * beta
        config['l1_out_l2_in_actor'] = trial.suggest_categorical("l1_out_l2_in_actor", [4, 8, 16, 32, 64, 128, 256, 512])
        config['l1_out_l2_in_critic'] = trial.suggest_categorical("l1_out_l2_in_critic", [8, 16, 32, 64, 128, 256, 512])
        decay_factor = 1 - alpha * config['lr_actor']
    elif config["policy_type"] == "q_learning":
        config["l1_out_l2_in_q"] = trial.suggest_categorical("l1_out_l2_in_q", [32, 64, 128, 256, 512])
        config["lr_q"] = trial.suggest_categorical("lr_q", [1e-4, 2e-4, 3e-4, 
                                                            4e-4, 5e-4, 6e-4, 
                                                            7e-4, 8e-4, 9e-4,
                                                            1e-5, 2e-5, 3e-5, 
                                                            4e-5, 5e-5, 6e-5, 
                                                            7e-5, 8e-5, 9e-5])
        decay_factor = 1 - alpha * config["lr_q"]
    # Update the parameters in the config
    config['servers_config']['recovery_cost'] = trial.suggest_int("recovery_cost", 10, 30, 1)
    config['coordinator_config']['decay_factor'] = decay_factor

    # Save the updated configuration file
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    
    subprocess.run(['python3', '/Users/jingyiwu/Desktop/MARL/multiprocessing_MARL.py'], check=True)
    result = subprocess.run(['python3', '/Users/jingyiwu/Desktop/MARL/plot_images.py'], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    # Extract the average reward from the output
    for line in result.stdout.decode().split('\n'):
        if line.startswith('Average reward: '):
            avg_reward = float(line[len('Average reward: '):])
            break
    else:
        raise RuntimeError('Average reward not found in output')
     
    return avg_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(study.best_params)
# Get the best parameters
best_params = study.best_params

# Load the existing configuration file
with open('config.json', 'r') as f:
    config = json.load(f)

# Update the parameters in the config
if config["policy_type"] == "ac_policy":  
    config['lr_actor'] = best_params["lr_actor"]
    config['lr_critic'] = best_params["lr_actor"] * best_params["beta"]
    config['l1_out_l2_in_actor'] = best_params["l1_out_l2_in_actor"]
    config['l1_out_l2_in_critic'] = best_params["l1_out_l2_in_critic"]
    config['coordinator_config']['decay_factor'] = 1 - best_params["alpha"] * best_params["lr_actor"]
elif config["policy_type"] == "q_learning":
    config["l1_out_l2_in_q"] = best_params["l1_out_l2_in_q"]
    config["lr_q"] = best_params["lr_q"]
    config['coordinator_config']['decay_factor'] = 1 - best_params["alpha"] * best_params["lr_q"]
config['servers_config']['recovery_cost'] = best_params["recovery_cost"]
if config["app_type"] == "queue_app":
    config["max_queue_length"] = best_params["max_queue_length"]

# Save the updated configuration file
with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)
