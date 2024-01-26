import subprocess

# Number of iterations
num_iterations = 10
app_type_sub_ids = [1]
policy_ids = [1]
record = {}
for i in range(num_iterations):
    for app_type_sub_id in app_type_sub_ids:
        for policy_id in policy_ids:
            print(f"Running iteration {i+1} spark {app_type_sub_id+1} policy {policy_id}")
            # Run the first script
            print("Running multiprocessing_MARL.py")
            subprocess.run(["python3", "/Users/jingyiwu/Documents/Project/MARL/src/multiprocessing_MARL.py", str(3), str(app_type_sub_id), str(policy_id)], check=True)

            # Run the second script
            print("Running plot_images.py")
            result = subprocess.run(['python3', '/Users/jingyiwu/Documents/Project/MARL/src/plot_images.py', str(3), str(app_type_sub_id), str(policy_id)], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            for line in result.stdout.decode().split('\n'):
                avg_reward = float(line)
                break
            else:
                raise RuntimeError('Average reward not found in output')
            record[f"iter{i+1} spark {app_type_sub_id+1}"] = avg_reward
            print(avg_reward)
print("All iterations completed.")
print(record)
