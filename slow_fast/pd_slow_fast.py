import copy
import json
import math
import os
import pickle
import random
import joblib
import numpy as np
import pandas as pd
import glob
import os
import re
import subprocess

from utils import *

ArgumentParser = '/home/studio-lab-user/sagemaker-studiolab-notebooks/PD/makeup/data/all_gait_clips/' # location of 3d joints
joints_path = '/home/studio-lab-user/sagemaker-studiolab-notebooks/PD/makeup/data/all_gait_clips/'
exam_df = pd.read_csv('/home/studio-lab-user/sagemaker-studiolab-notebooks/PD/makeup/data/pd_label_dd_net_5.csv') # label file with example format

print(exam_df.head(10))

def dataloader(data_path, label_path):
    first_path = glob.glob(data_path + os.sep + '*')
    pkl_path = [glob.glob(i + os.sep + '*.pkl')[0] for i in first_path]
    person_uuid = [re.split('[-_]', i.split(os.sep)[-1])[0] for i in first_path]
    data = [joblib.load(i)['joints3d'] for i in pkl_path]
    df_label = pd.read_csv(label_path)
    label = [df_label.loc[df_label['Video Name'] == i.split(os.sep)[-1], 'Label'].values[0] if '_' in i.split(os.sep)[-1] else df_label.loc[df_label['Video Name'] == i.split(os.sep)[-1].replace('-', '_'), 'Label'].values[0] for i in first_path ]
    return data, person_uuid, label

def split_data_kfold(person_id, data_list, label_list, k, save_dir):
    unique_values = list(set(person_id))  # Get unique values in the data.
    unique_values = sorted([int(i) for i in unique_values])
    random.shuffle(unique_values)  # Randomly shuffle unique values
    unique_values = [str(i) for i in unique_values]
    fold_size = len(unique_values) // k  # Calculate the size of each fold
    folds = [unique_values[i * fold_size:(i + 1) * fold_size] if i != (k-1) else unique_values[i * fold_size:] for i in range(k)] 

    for i in range(k):
        train_data = []
        test_data = []
        train_person_id = []
        train_label = []
        test_label = []
        test_person_id = []
        test_ids = folds[i]  # The current fold serves as the test set
        train_ids = [id for id in unique_values if id not in test_ids]  # The remaining values serve as the training set.

        # Extract the corresponding file_path based on the person_id in the training set.
        for id, df, lb in zip(person_id, data_list, label_list):
            if id in train_ids:
                train_data.append(df)
                train_label.append(lb)
                train_person_id.append(id)

        for id, df, lb in zip(person_id, data_list, label_list):
            if id in test_ids:
                test_data.append(df)
                test_label.append(lb)
                test_person_id.append(id)
        train = {}
        test = {}
        train['pose'] = train_data
        train['label'] = train_label
        test['pose'] = test_data
        test['label'] = test_label

        pickle.dump(train_person_id, open(save_dir+"train_list_"+str(i)+".pkl", "wb"))
        pickle.dump(test_person_id, open(save_dir+"test_list_"+str(i)+".pkl", "wb"))
        pickle.dump(train, open(save_dir+"train_"+str(i)+".pkl", "wb"))
        pickle.dump(test, open(save_dir+"test_"+str(i)+".pkl", "wb"))

k_fold = 5
data_list, person_id, label_list = dataloader(data_path='/home/studio-lab-user/sagemaker-studiolab-notebooks/PD/PKL_80_Dir', label_path='/home/studio-lab-user/sagemaker-studiolab-notebooks/PD/makeup/data/pd_label.csv')
split_data_kfold(person_id, data_list, label_list, k=k_fold, save_dir='/home/studio-lab-user/sagemaker-studiolab-notebooks/PD/PKL_80_data/')


ArgumentParser = '/home/studio-lab-user/sagemaker-studiolab-notebooks/PD/makeup/data/all_gait_clips/'
joints_path = '/home/studio-lab-user/sagemaker-studiolab-notebooks/PD/makeup/data/all_gait_clips/'
exam_df = pd.read_csv('/home/studio-lab-user/sagemaker-studiolab-notebooks/PD/makeup/data/pd_label.csv') 
save_dir = '/home/studio-lab-user/sagemaker-studiolab-notebooks/PD/makeup/splits/PD_PKL/'
print(exam_df.head(10))

import warnings
warnings.filterwarnings('ignore')

get_ipython().system('python train.py')

