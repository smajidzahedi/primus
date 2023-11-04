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


def objective(trial, config_file_name, app_type_id, app_type_sub_id, policy_id):
    
    # Load the configuration file
    with open(config_file_name, 'r') as f:
        config = json.load(f)
    app_type = config["app_types"][app_type_id]
    app_sub_type = config["app_sub_types"][app_type][app_type_sub_id]
    config["a_lr"][app_type][app_sub_type] = trial.suggest_categorical("a_lr", [1e-4, 2e-4, 3e-4, 
                                                        4e-4, 5e-4, 6e-4,
                                                        7e-4, 8e-4, 9e-4])  
    b = trial.suggest_categorical("b", [1.5, 2.0, 2.5, 3.0])
    config['c_lr'][app_type][app_sub_type] = config["a_lr"][app_type][app_sub_type] * b
    # Update the parameters in the config

    with open(config_file_name, 'w') as f:
        f.write(format(config))

    subprocess.run(['python3', '/Users/jingyiwu/Desktop/Project/MARL/src/multiprocessing_MARL.py', str(app_type_id), str(app_type_sub_id), str(policy_id)], check=True)
    result = subprocess.run(['python3', '/Users/jingyiwu/Desktop/Project/MARL/src/plot_images.py', str(app_type_id), str(app_type_sub_id), str(policy_id)], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    # Extract the average reward from the output
    for line in result.stdout.decode().split('\n'):
        if line.startswith('Average reward: '):
            avg_reward = float(line[len('Average reward: '):])
            break
    else:
        raise RuntimeError('Average reward not found in output')
     
    return avg_reward

# Load the existing configuration file

config_file_name = '/Users/jingyiwu/Desktop/Project/MARL/configs/config.json'
with open(config_file_name, 'r') as f:
    config = json.load(f)

policy_id = 0
app_type_sub_ids = [1, 3, 2]
for app_type_id in range(0, 3):
    for app_type_sub_id in range(0, app_type_sub_ids[app_type_id]):
        app_type = config["app_types"][app_type_id]
        app_sub_type = config["app_sub_types"][app_type][app_type_sub_id]
        policy_type = config["policy_types"][policy_id]
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, config_file_name, app_type_id, app_type_sub_id, policy_id), n_trials=25)
        print(study.best_params)
        # Get the best parameters
        best_params = study.best_params
        # Update the parameters in the config
        config["a_lr"][app_type][app_sub_type] = best_params["a_lr"]
        config['c_lr'][app_type][app_sub_type] = best_params["a_lr"] * best_params["b"]

        with open("/Users/jingyiwu/Desktop/Project/MARL/configs/config.json", 'w') as f:
            f.write(format(config))
