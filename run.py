import torch
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from functions.running import one_run, plot_results, plot_running_time
import random


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def run(X, y, dataname):
    # for init_type in ['he', 'xavier', 'orthogonal']:
    for init_type in ['he']:
        epochs = 200
        batch_size = 64
        n_layer = 5
        dataname_ = dataname + str(n_layer) + init_type
        nruns = 1
        res = []
        training_time = []
        for i in range(nruns):
            seed_everything(seed = i)
            res_one_run, training_time_one_run = one_run(init_type, X, y, epochs, batch_size, n_layer, n_frozen_epochs=200, dataname=dataname)
            res.append(res_one_run)
            training_time.append(training_time_one_run)
        res = np.array(res)
        training_time = np.array(training_time)
        mean_all_run = np.mean(res, axis = 0)
        training_time_mean_all_run = np.mean(training_time, axis = 0)
        plot_results(dataname_, mean_all_run, output_dir=f"output/img")
        if not os.path.exists(f"output/res"):
            os.makedirs(f"output/res")
        np.save(f"output/res/"+dataname_, res)

        plot_running_time(dataname_+"_time", training_time_mean_all_run, output_dir=f"output/img")
        if not os.path.exists(f"output/time"):
            os.makedirs(f"output/time")
        np.save(f"output/time/"+dataname_, training_time)

X = None
y = None

run(X, y, 'mnist')