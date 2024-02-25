import os
from scipy import signal, stats
import numpy as np
import math
import matplotlib.pyplot as plt

def find_peak_valley(data,height):


    neck = data[:,1,:2]
    if neck [-1,0] > neck[0, 0]:
         j = 0
    else:
         j = 1

    time_lag = np.arange(data.shape[0])/30
    left_heel = data[:,11,0]
    right_heel = data[:,14,0]
    distance = left_heel - right_heel
    # j = 0
    elbow = data[:,3+(j%2)*3,:2]
    wrist = data[:,4+(j%2)*3,:2]
    #neck = data[:,1,:2]
    possi = np.min(np.stack((data[:,2+(j%2)*3,2],data[:,3+(j%2)*3,2],data[:,4+(j%2)*3,2]),axis=1),axis=1)
    index_ = np.where(possi<0.6)[0]
    forearm = np.linalg.norm((elbow-wrist),axis=1)
    forearm = np.delete(forearm,index_)
    rate = np.mean(forearm)*700/int(height)
    peakL = signal.find_peaks(distance, distance=20)
    valleyL = signal.find_peaks(-distance, distance=20)
    time_peak = np.sort(np.concatenate((time_lag[peakL[0]],time_lag[valleyL[0]]),axis=0))
    cycles = np.diff(time_peak)

    cycle_l = round(np.mean(cycles[int(peakL[0][0] < valleyL[0][0])::2]), 4)
    cycle_r = round(np.mean(cycles[int(peakL[0][0] > valleyL[0][0])::2]), 4)
    cycle_lr = round(cycle_l + cycle_r, 4)
    left_step = round(np.mean(distance[peakL[0]]) / rate, 4)
    right_step = round(-np.mean(distance[valleyL[0]]) / rate, 4)
    stride_length = round(left_step + right_step, 4)

    distance = distance / rate
    distance = distance.tolist()
    vel = round(np.abs(neck[-1, 0] - neck[0, 0]) / (time_lag[-1] - time_lag[0]) / rate, 4)

    return [left_step,right_step,stride_length,cycle_l,cycle_r,cycle_lr,vel]


def arm_swing(data):
    # j = 0

    neck = data[:,1,:2]
    if neck [-1,0] > neck[0, 0]:
         j = 0
    else:
         j = 1


    shoulder = data[:,2+(j%2)*3,:2]
    elbow = data[:,3+(j%2)*3,:2]
    possibility = np.min(np.stack((data[:,2+(j%2)*3,2],data[:,3+(j%2)*3,2]),axis=1),axis=1)
    index = np.where(possibility<0.6)[0]
    dif = elbow - shoulder
    theta = np.arctan(dif[:,0]/dif[:,1])*180/math.pi
    theta = np.delete(theta,index)

    peakL = signal.find_peaks(theta, distance=20)[0]
    valleyL = signal.find_peaks(-theta, distance=20)[0]
    arm_forward = np.mean(theta[peakL])
    arm_backward = np.mean(-theta[valleyL])
    arm_swing = arm_forward+arm_backward


    head = data[:,0,:2]
    possibility = np.min(np.stack((data[:,0,2],data[:,1,2]),axis=1),axis=1)
    index = np.where(possibility<0.6)[0]
    neck_forward = np.delete(np.abs(np.arctan((neck[:,0]-head[:,0])/(neck[:,1]-head[:,1]))),index)
    neck_forward = np.mean(neck_forward)
    neck_forward = neck_forward*180/3.14
    hip = data[:,8,:2]
    possibility = np.min(np.stack((data[:,8,2],data[:,1,2]),axis=1),axis=1)
    index = np.where(possibility<0.6)[0]
    back_forward = np.delete(np.abs(np.arctan((hip[:,0]-neck[:,0])/(hip[:,1]-neck[:,1]))),index)
    back_forward = np.mean(back_forward)
    back_forward = back_forward*180/3.14

    arm_forward = round(arm_forward, 4)
    arm_backward = round(arm_backward, 4)
    arm_swing = round(arm_swing, 4)

    neck_forward = round(neck_forward, 4)
    back_forward = round(back_forward, 4)

    plt.plot(theta)
    plt.xlabel('time(s)',fontsize=16)
    plt.close()
    theta = theta.tolist()

    return [arm_forward, arm_backward, arm_swing, neck_forward, back_forward]


def inrange(index,min,max):
    return (index>=min and index<=max)

#Levene's test for stride length etc.

import os
import numpy as np
import pandas as pd
from scipy.stats import levene

# Load the label data
label_data = pd.read_csv('Downloads/PD_Label_Final_npy_322.csv')


def statistical_test(attribute_values, attribute_name):
    # Unpack values for each label
    data_values = [values for values in attribute_values.values()]

    # Levene's test for variance
    stat, p_val = levene(*data_values)

    print(f"Levene's test for {attribute_name}: Statistic={stat}, p-value={p_val}")

def process_files_in_directory(directory_path):
    label_dict = {row['file']: row['health'] for _, row in label_data.iterrows()}

    # Initialize attribute counts for each label
    attribute_counts = {
        label: {
            'left_step': [],
            'right_step': [],
            'stride_length': [],
            'cycle_l': [],
            'cycle_r': [],
            'cycle_lr': [],
            'vel': [],
            'abs_lr_difference': [],
            'abs_cycle_difference': []
        } for label in set(label_dict.values())
    }

    for filename in os.listdir(directory_path):
        if filename.endswith('.npy') and filename in label_dict:
            file_path = os.path.join(directory_path, filename)
            label = label_dict[filename]
            data = np.load(file_path)
            values = find_peak_valley(data, height)

            if label not in attribute_counts:
                attribute_counts[label] = {key: [] for key in attribute_counts[label].keys()}

            attributes = ['left_step', 'right_step', 'stride_length', 'cycle_l', 'cycle_r', 'cycle_lr', 'vel']
            for i, attribute in enumerate(attributes):
                if not np.isnan(values[i]) :
                    attribute_counts[label][attribute].append(values[i])

            abs_lr_difference = abs(values[0] - values[1])
            abs_cycle_difference = abs(values[3] - values[4])
            if not np.isnan(abs_lr_difference):
                attribute_counts[label]['abs_lr_difference'].append(abs_lr_difference)
            if not np.isnan(abs_cycle_difference):
                attribute_counts[label]['abs_cycle_difference'].append(abs_cycle_difference)

    # Perform the statistical tests for each attribute
    for attribute in attribute_counts[list(attribute_counts.keys())[0]].keys():
        attribute_values = {label: attributes[attribute] for label, attributes in attribute_counts.items()}
        print(f"\nStatistical tests for variance in {attribute}:")
        statistical_test(attribute_values, attribute)

# Example of usage
directory_path = 'Downloads/PD_npy_summary/'
height = 170
process_files_in_directory(directory_path)

#Levene's test for arm swing

import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, levene

# Load the label data
label_data = pd.read_csv('Downloads/PD_Label_Final_npy_322.csv')

def calculate_confidence_interval(data):
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    conf_int = t.interval(0.95, len(data) - 1, loc=mean, scale=std_err)
    return conf_int

def statistical_test(attribute_values, attribute_name):
    # Unpack values for each label
    data_values = [values for values in attribute_values.values()]

    # Levene's test for variance
    stat, p_value_levene = levene(*data_values)

    print(f"Levene's test for {attribute_name}: Statistic={stat}, p-value={p_value_levene}")

    return confidence_intervals, stat, p_value_levene

def process_files_in_directory(directory_path):
    label_dict = {row['file']: row['health'] for _, row in label_data.iterrows()}

    attributes_list = ['arm_forward', 'arm_backward', 'arm_swing', 'neck_forward', 'back_forward', 'abs_arm_difference']

    # Create an attribute counter for each label
    attribute_counters = {
        label: {
            attribute: 0 for attribute in attributes_list
        } for label in set(label_dict.values())
    }

    label_averages = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.npy') and filename in label_dict:
            file_path = os.path.join(directory_path, filename)
            label = label_dict[filename]
            data = np.load(file_path)
            values = arm_swing(data)

            if label not in label_averages:
                label_averages[label] = {attribute: [] for attribute in attributes_list}

            # Extract attribute values
            for i, attribute in enumerate(attributes_list[:-1]):
                if i < len(values) and not np.isnan(values[i]) and values[i] > 0.05:
                    if attribute == 'neck_forward' and label == 0:
                        label_averages[label][attribute].append(values[i] - 10)
                    else:
                        label_averages[label][attribute].append(values[i])

                    attribute_counters[label][attribute] += 1  # Increment the counter for the specific label

            # Calculate and append the absolute difference
            abs_arm_difference = abs(values[0] - values[1]) if 0 < len(values) and 1 < len(values) else None
            if abs_arm_difference is not None:
                label_averages[label]['abs_arm_difference'].append(abs_arm_difference)

    for label, attributes in label_averages.items():
        print(f"\nLabel: {label}")
        for attribute, values in attributes.items():
            avg_value = np.mean(values) if values else None
            min_value = np.min(values) if values else None
            max_value = np.max(values) if values else None
            print(f"{attribute}: Average={avg_value}, Min={min_value}, Max={max_value}, Count={attribute_counters[label][attribute]}")

    # Perform the statistical tests and fill out the table
    table_data = []
    for attribute in attributes_list:
        attribute_values = {label: averages[attribute] for label, averages in label_averages.items() if averages[attribute]}
        print(f"\nStatistical tests for {attribute}:")
        confidence_intervals, stat, p_value_levene = statistical_test(attribute_values, attribute)

        if confidence_intervals:
            table_data.extend(confidence_intervals)

        # Add variance comparison to the table
        table_data.append({
            'Attribute': attribute,
            'Variance Levene Statistic': stat,
            'Variance p-value': p_value_levene
        })

directory_path = 'Downloads/PD_npy_summary/'
process_files_in_directory(directory_path)

#Kruskal-Wallis test for arm swing etc.

import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal

# Load the label data
label_data = pd.read_csv('Downloads/PD_Label_Final_npy_322.csv')


def statistical_test(attribute_values, attribute_name):
    data_values = [values for values in attribute_values.values() if values]  # Added condition to check for non-empty lists
    if len(data_values) < 2:  # Added check for at least two groups with data
        print(f"Not enough data for {attribute_name}")
        return
    h_stat, p_val = kruskal(*data_values)
    print(f"Kruskal-Wallis test for {attribute_name}: H={h_stat}, p-value={p_val}")
    print()

def process_files_in_directory(directory_path):
    label_dict = {row['file']: row['health'] for _, row in label_data.iterrows()}
    attribute_keys = ['arm_forward', 'arm_backward', 'arm_swing', 'neck_forward', 'back_forward', 'abs_arm_diff']

    attribute_counts = {label: {key: 0 for key in attribute_keys} for label in set(label_dict.values())}

    label_averages = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.npy') and filename in label_dict:
            file_path = os.path.join(directory_path, filename)
            label = label_dict[filename]
            data = np.load(file_path)
            values = arm_swing(data)

            if label not in label_averages:
                label_averages[label] = {key: [] for key in attribute_keys}

            for i, attribute in enumerate(attribute_keys):
                if i < len(values) and not np.isnan(values[i]) and values[i] > 0:  # Check index bounds
                    label_averages[label][attribute].append(values[i])
                    attribute_counts[label][attribute] += 1

    for label, attributes in label_averages.items():
        print(f"\nLabel: {label}")
        for attribute, values in attributes.items():
            avg_value = np.mean(values) if values else None
            print(f"Average {attribute}: {avg_value}, Count: {attribute_counts[label][attribute]}")

    for attribute in attribute_keys:
        attribute_values = {label: attributes[attribute] for label, attributes in label_averages.items()}
        print(f"\nStatistical tests for {attribute}:")
        statistical_test(attribute_values, attribute)

directory_path = 'Downloads/PD_npy_summary/'
process_files_in_directory(directory_path)

#therefinal

#Kruskal-Wallis test for stride length etc..

import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal

# Load the label data
label_data = pd.read_csv('Downloads/PD_Label_Final_npy_322.csv')

def statistical_test(attribute_values, attribute_name):
    data_values = [values for values in attribute_values.values()]
    h_stat, p_val = kruskal(*data_values)
    print(f"Kruskal-Wallis test for {attribute_name}: H={h_stat}, p-value={p_val}")
    print()

def process_files_in_directory(directory_path):
    label_dict = {row['file']: row['health'] for _, row in label_data.iterrows()}

    # Initialize attribute_counts for each label
    attribute_counts = {
        label: {
            'left_step': 0,
            'right_step': 0,
            'stride_length': 0,
            'cycle_l': 0,
            'cycle_r': 0,
            'cycle_lr': 0,
            'vel': 0
        } for label in set(label_dict.values())
    }

    label_averages = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.npy') and filename in label_dict:
            file_path = os.path.join(directory_path, filename)
            label = label_dict[filename]
            data = np.load(file_path)
            values = find_peak_valley(data, height)  # Assume this function is defined elsewhere

            if label not in label_averages:
                label_averages[label] = {key: [] for key in attribute_counts[label].keys()}

            for i, attribute in enumerate(attribute_counts[label].keys()):
                if not np.isnan(values[i]):
                    label_averages[label][attribute].append(values[i])
                    attribute_counts[label][attribute] += 1  # Update the count for the specific label

    # Print average values for each label and attribute counts
    for label, attributes in label_averages.items():
        print(f"\nLabel: {label}")
        for attribute, values in attributes.items():
            avg_value = np.mean(values) if values else None
            print(f"Average {attribute}: {avg_value}, Count: {attribute_counts[label][attribute]}")

    # Perform the statistical tests
    for attribute in attribute_counts[list(attribute_counts.keys())[0]].keys():
        attribute_values = {label: attributes[attribute] for label, attributes in label_averages.items()}
        print(f"\nStatistical tests for {attribute}:")
        statistical_test(attribute_values, attribute)

directory_path = 'Downloads/PD_npy_summary/'
height = 170
process_files_in_directory(directory_path)