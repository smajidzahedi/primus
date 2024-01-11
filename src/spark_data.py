import glob
import re
from datetime import datetime
import pandas as pd
import numpy as np


def convert_to_relative_time(log_lines):
    reference_time = datetime.strptime(log_lines[0][:17], '%y/%m/%d %H:%M:%S')
    relative_times = []
    for line in log_lines:
        current_time = datetime.strptime(line[:17], '%y/%m/%d %H:%M:%S')
        relative_seconds = int((current_time - reference_time).total_seconds())
        relative_time_entry = f"{relative_seconds}:{line[18:]}"
        relative_times.append(relative_time_entry)
    return relative_times


def interpolation(list, target_length):
    # Original indices (0 to 5 for 6 elements)
    original_indices = np.linspace(0, len(list) - 1, num=len(list))

    # New indices for target list
    new_indices = np.linspace(0, len(list) - 1, num=target_length)

    # Interpolating to get the new list
    new_list = np.interp(new_indices, original_indices, list)

    return new_list

executor_log_dir_path = 'data/executorLog'

def process_executor_logs(path):
    als_nominal_list = []
    als_sprinting_list = []
    kmeans_nominal_list = []
    kmeans_sprinting_list = []
    lr_nominal_list = []
    lr_sprinting_list = []
    pr_nominal_list = []
    pr_sprinting_list = []
    svm_nominal_list = []
    svm_sprinting_list = []
    for in_path in glob.glob(path + '/*.txt'):
        file_name = in_path.split("/")[-1]
        file_name_list = file_name.split("_")
        algorithm_name = file_name_list[0]
        mode_type = file_name_list[1]
        log_lines = []
        with open(in_path, 'r') as file:
            for line in file:
                if "Executor: !!!!Finished task" in line.strip():
                    log_lines.append(line.strip())
        count_sec = {}
        relative_time_lines = convert_to_relative_time(log_lines)

        for line in relative_time_lines:
            info = line.split(":", 1)
            if info[0] in count_sec:
                count_sec[info[0]] += 1
            else:
                count_sec[info[0]] = 1

        max_key = max(int(key) for key in count_sec)
        df = pd.DataFrame(index=range(max_key + 1))
        df[f'{algorithm_name} {mode_type} Count'] = 0

        for key, value in count_sec.items():
            df.at[int(key), f'{algorithm_name} {mode_type} Count'] = value
        
        if algorithm_name == "als":
            if mode_type == "nominal":
                als_nominal_list = df[f'{algorithm_name} {mode_type} Count'].tolist()
            else:
                als_sprinting_list = df[f'{algorithm_name} {mode_type} Count'].tolist()
        elif algorithm_name == "kmeans":
            if mode_type == "nominal":
                kmeans_nominal_list = df[f'{algorithm_name} {mode_type} Count'].tolist()
            else:
                kmeans_sprinting_list = df[f'{algorithm_name} {mode_type} Count'].tolist()
        elif algorithm_name == "lr":
            if mode_type == "nominal":
                lr_nominal_list = df[f'{algorithm_name} {mode_type} Count'].tolist()
            else:
                lr_sprinting_list = df[f'{algorithm_name} {mode_type} Count'].tolist()
        elif algorithm_name == "pr":
            if mode_type == "nominal":
                pr_nominal_list = df[f'{algorithm_name} {mode_type} Count'].tolist()
            else:
                pr_sprinting_list = df[f'{algorithm_name} {mode_type} Count'].tolist()
        elif algorithm_name == "svm":
            if mode_type == "nominal":
                svm_nominal_list = df[f'{algorithm_name} {mode_type} Count'].tolist()
            else:
                svm_sprinting_list = df[f'{algorithm_name} {mode_type} Count'].tolist()

    algorithm_list = [als_nominal_list, 
                      als_sprinting_list, 
                      kmeans_nominal_list, 
                      kmeans_sprinting_list, 
                      lr_nominal_list, 
                      lr_sprinting_list, 
                      pr_nominal_list, 
                      pr_sprinting_list, 
                      svm_nominal_list, 
                      svm_sprinting_list]
    return algorithm_list

if __name__ == '__main__':
    algorithm_list = process_executor_logs(executor_log_dir_path)
    als_nominal_list = algorithm_list[0]
    als_sprinting_list = algorithm_list[1]
    kmeans_nominal_list = algorithm_list[2]
    kmeans_sprinting_list = algorithm_list[3]
    lr_nominal_list = algorithm_list[4]
    lr_sprinting_list = algorithm_list[5]
    pr_nominal_list = algorithm_list[6]
    pr_sprinting_list = algorithm_list[7]
    svm_nominal_list = algorithm_list[8]
    svm_sprinting_list = algorithm_list[9]

als_sprinting_list = interpolation(als_sprinting_list, len(als_nominal_list))
kmeans_sprinting_list = interpolation(kmeans_sprinting_list, len(kmeans_nominal_list))
lr_sprinting_list = interpolation(lr_sprinting_list, len(lr_nominal_list))
pr_sprinting_list = interpolation(pr_sprinting_list, len(pr_nominal_list))
svm_sprinting_list = interpolation(svm_sprinting_list, len(svm_nominal_list))

als_nominal_list = np.array(als_nominal_list)
kmeans_nominal_list = np.array(kmeans_nominal_list)
lr_nominal_list = np.array(lr_nominal_list)
pr_nominal_list = np.array(pr_nominal_list)
svm_nominal_list = np.array(svm_nominal_list)

als_gain = als_sprinting_list - als_nominal_list
kmeans_gain = kmeans_sprinting_list - kmeans_nominal_list
lr_gain = lr_sprinting_list - lr_nominal_list
pr_gain = pr_sprinting_list - pr_nominal_list
svm_gain = svm_sprinting_list - svm_nominal_list

als_gain[als_gain < 0] = 0
kmeans_gain[kmeans_gain < 0] = 0
lr_gain[lr_gain < 0] = 0
pr_gain[pr_gain < 0] = 0
svm_gain[svm_gain < 0] = 0

gain_dic = {
    "als_gain": als_gain,
    "kmeans_gain": kmeans_gain,
    "lr_gain": lr_gain,
    "pr_gain": pr_gain,
    "svm_gain": svm_gain
    }

with open('data/gain.txt', 'w+') as file:
    for key, value_list in gain_dic.items():
        file.write(f"{key}:")
        for value in value_list:
            file.write(f"{value}\t")
        file.write("\n")

als_prob = np.histogram(als_gain, np.arange(0, als_gain.max()+1, 1))[0] / np.histogram(als_gain, np.arange(0, als_gain.max()+1, 1))[0].sum()
kmeans_prob = np.histogram(kmeans_gain, np.arange(0, kmeans_gain.max()+1, 1))[0] / np.histogram(kmeans_gain, np.arange(0, kmeans_gain.max()+1, 1))[0].sum()
lr_prob = np.histogram(lr_gain, np.arange(0, lr_gain.max()+1, 1))[0] / np.histogram(lr_gain, np.arange(0, lr_gain.max()+1, 1))[0].sum()
pr_prob = np.histogram(pr_gain, np.arange(0, pr_gain.max()+1, 1))[0] / np.histogram(pr_gain, np.arange(0, pr_gain.max()+1, 1))[0].sum()
svm_prob = np.histogram(svm_gain, np.arange(0, svm_gain.max()+1, 1))[0] / np.histogram(svm_gain, np.arange(0, svm_gain.max()+1, 1))[0].sum()

prob_dic = {
    "als_prob": als_prob,
    "kmeans_prob": kmeans_prob,
    "lr_prob": lr_prob,
    "pr_prob": pr_prob,
    "svm_prob": svm_prob
}

with open('data/gain.txt', 'a+') as file:
    for key, value_list in prob_dic.items():
        file.write(f"{key}:")
        for value in value_list:
            file.write(f"{value}\t")
        file.write("\n")

utility_dic = {
    "als_utilities": np.arange(0, als_gain.max(), 1) / als_gain.max(),
    "kmeans_utilities": np.arange(0, kmeans_gain.max(), 1) / kmeans_gain.max(),
    "lr_utilities": np.arange(0, lr_gain.max(), 1) / lr_gain.max(),
    "pr_utilities": np.arange(0, pr_gain.max(), 1) / pr_gain.max(),
    "svm_utilities": np.arange(0, svm_gain.max(), 1) / svm_gain.max()
}

with open('data/gain.txt', 'a+') as file:
    for key, value_list in utility_dic.items():
        file.write(f"{key}:")
        for value in value_list:
            file.write(f"{value}\t")
        file.write("\n")