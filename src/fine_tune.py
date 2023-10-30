import optuna
import json
import subprocess 


def format(obj, indent_level=0):
    INDENT_SIZE = 4
    
    # Function to determine if an object is a matrix
    def is_matrix(array):
        return isinstance(array, list) and all(isinstance(row, list) for row in array)
    
    # Format matrix
    def format_matrix(matrix, indent_level):
        rows = []
        indent = ' ' * INDENT_SIZE * indent_level
        for row in matrix:
            formatted_row = ', '.join(map(str, row))
            rows.append(indent + ' ' * INDENT_SIZE + '[' + formatted_row + ']')
        return '[\n' + ',\n'.join(rows) + '\n' + indent + ']'
    
    if is_matrix(obj):
        return format_matrix(obj, indent_level)
    
    elif isinstance(obj, list):
        return '[' + ', '.join(json.dumps(item) for item in obj) + ']'
    
    elif isinstance(obj, dict):
        items = []
        indent = ' ' * INDENT_SIZE * indent_level
        for key, value in obj.items():
            formatted_value = format(value, indent_level + 1)
            items.append(f'"{key}": {formatted_value}')
        return '{\n' + ',\n'.join(indent + ' ' * INDENT_SIZE + item for item in items) + '\n' + indent + '}'
    
    else:
        return json.dumps(obj)


def objective(trial, app_type_id, app_type_sub_id, policy_id):
    
    # Load the configuration file
    with open('/Users/jingyiwu/Desktop/MARL/configs/config.json', 'r') as f:
        config = json.load(f)
    a = trial.suggest_categorical("a", [2, 4, 6, 8, 10])
    config["a_lr"] = trial.suggest_categorical("a_lr", [1e-4, 2e-4, 3e-4, 
                                                        4e-4, 5e-4, 6e-4])  
    b = trial.suggest_categorical("b", [1.5, 2.0, 2.5, 3.0])
    config['c_lr'] = config['a_lr'] * b
    decay_factor = 1 - a * config['a_lr']
    # Update the parameters in the config
    config['coordinator_config']['sprinters_decay_factor'] = decay_factor

    with open("/Users/jingyiwu/Desktop/MARL/configs/config.json", 'w') as f:
        f.write(format(config))

    subprocess.run(['python3', '/Users/jingyiwu/Desktop/MARL/src/multiprocessing_MARL.py', str(app_type_id), str(app_type_sub_id), str(policy_id)], check=True)
    result = subprocess.run(['python3', '/Users/jingyiwu/Desktop/MARL/src/plot_images.py', str(app_type_id), str(app_type_sub_id), str(policy_id)], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    # Extract the average reward from the output
    for line in result.stdout.decode().split('\n'):
        if line.startswith('Average reward: '):
            avg_reward = float(line[len('Average reward: '):])
            break
    else:
        raise RuntimeError('Average reward not found in output')
     
    return avg_reward

policy_id = 0
app_type_sub_ids = [1, 3, 1]
for app_type_id in range(0, 3):
    for app_type_sub_id in range(0, app_type_sub_ids[app_type_id]):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, app_type_id, app_type_sub_id, policy_id), n_trials=15)
        print(study.best_params)
        # Get the best parameters
        best_params = study.best_params

        # Load the existing configuration file
        with open('/Users/jingyiwu/Desktop/MARL/configs/config.json', 'r') as f:
            config = json.load(f)

        # Update the parameters in the config
        config['a_lr'] = best_params["a_lr"]
        config['c_lr'] = best_params["a_lr"] * best_params["b"]
        config['coordinator_config']['sprinters_decay_factor'] = 1 - best_params["a"] * best_params["a_lr"]

        # Save the updated configuration file
        folder = f"/Users/jingyiwu/Desktop/MARL/configs"
        num_servers = config["num_servers"]
        app_type = config["app_types"][app_type_id]
        app_sub_type = config["app_sub_types"][app_type][app_type_sub_id]
        policy_type = config["policy_types"][policy_id]
        path = f"{folder}/{num_servers}_{policy_type}_{app_type}_{app_sub_type}"
        add_noise = config['coordinator_config']["add_noise"]
        file_name = f"{path}_noise_config.json" if add_noise == 1 else f"{path}_no_noise_config.json"
        with open(file_name, 'w') as f:
            f.write(format(config))
        with open("/Users/jingyiwu/Desktop/MARL/configs/config.json", 'w') as f:
            f.write(format(config))
