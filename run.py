import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
from loader.load_data import *
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from optimizers.network import *
from optimizers.constrained import *
from algorithms.driver2 import run_algorithms

import sys
import hashlib

BUF_SIZE = 65536

sha256 = hashlib.sha256()

def env_hash():
    with open("./.env", 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

load_dotenv()

Q = int(os.getenv("Q_INIT"))
past = int(os.getenv("PAST"))
V_0 = int(os.getenv("V_0"))
future = int(os.getenv("FUTURE"))
alpha = float(os.getenv("ALPHA"))
NumSeq = int(os.getenv("NUM_SEQ"))
threshold = int(os.getenv("THRESHOLD"))
train_memory = int(os.getenv("TRAIN_MEMORY"))
use_saved = os.getenv("USE_SAVED")=="True"
run_others = os.getenv("RUN_OTHERS")=="True"
cost_constraint = int(os.getenv("COST_CONSTRAINT"))
time_limit = float('inf') if os.getenv("TIME_LIMIT")=='inf' else int(os.getenv("TIME_LIMIT"))
path_to_input = os.getenv("PATH_TO_INPUT")
tag = env_hash()[:10]
print("Experiment Tag:", tag)

cache_constraint = int(alpha*threshold)

path = f"./experiments/csv_{NumSeq}/"
try:
    os.makedirs(path)
except FileExistsError:
    pass



data = pd.read_csv(path_to_input, sep = ' ')
data.columns = ['Timestamp', 'File_ID', "File_Size"]
DataLength = len(data)


if run_others: run_algorithms(path_to_input, path, NumSeq, time_limit, threshold, alpha)

gamma = np.random.normal(0, 1, (threshold,))



queue = []
err = []
objective = []
fetching_cost = []
cache_hit = []
prev_demands = []
best_maximum = []
hit_rate = []
download_rate = []


hit_rate_ftpl = []
download_rate_ftpl = []


X_t_1 = np.zeros((threshold,))
init_indices = random.sample(range(threshold), cache_constraint)
X_t_1[init_indices] = 1

X_t_1_ftpl = np.zeros((threshold,))
X_t_1_ftpl[init_indices] = 1

for i in tqdm(range(NumSeq)):
    V = V_0
    if os.getenv("USE_ROOT_V")=="True": V *= (i+1)**0.5
    next_dem, time = get_demands(i, time_limit, data, DataLength, NumSeq, threshold)
    X_t = np.zeros((threshold,))
    init_indices = random.sample(range(threshold), cache_constraint)
    X_t[init_indices] = 1
    
    X_t_ftpl = np.zeros((threshold,))
    X_t_ftpl[init_indices] = 1
    if i==past+future:
        model = get_model(prev_demands, past, future, threshold, use_saved)
        print(model.summary())
    elif i>past+future:
        to_train = prev_demands[max(0, i-train_memory):]
        update_weight(model, to_train, past, future)
        pred = predict_demand(model, prev_demands[i-past:])
        pred = np.maximum(pred, np.zeros((pred.size,)))
        pred = np.round(pred)
        np.array(prev_demands).mean(axis=0)
        
        delta_t = get_delta()
        X_t, obj = constrained_solve(pred, cache_constraint, cost_constraint, X_t_1, delta_t, Q, V, threshold)
        objective.append(obj)
        Delta = delta_t*np.linalg.norm(X_t-X_t_1, ord=1)/2
        fetching_cost.append(Delta)
        
        X_t_ftpl, obj_ftpl = constrained_solve_ftpl(np.array(prev_demands).sum(axis=0), X_t_1_ftpl, cache_constraint, gamma, threshold, i)
        
        
        e = np.linalg.norm(next_dem-pred, ord=2)/len(pred)
        err.append(e)
        actual_cache_hit = np.dot(next_dem, X_t)
        cache_hit.append(actual_cache_hit)
        
        indices = np.argsort(next_dem)[::-1][:cache_constraint]
        final = np.zeros((threshold,))
        final[indices] = 1
        
        
        best = np.dot(next_dem, final)
        best_maximum.append(best)
        
        
        
        Q = max(Q + Delta - cost_constraint, 0)
        queue.append(Q)
    
    hit_rate.append(np.dot(X_t, next_dem)/time)
    download_rate.append(np.sum(np.logical_and(X_t==1, X_t_1==0))/time)
    
    hit_rate_ftpl.append(np.dot(X_t_ftpl, next_dem)/time)
    download_rate_ftpl.append(np.sum(np.logical_and(X_t_ftpl==1, X_t_1_ftpl==0))/time)
        
    X_t_1 = X_t
    X_t_1_ftpl = X_t_ftpl
    
    prev_demands.append(next_dem)

our_path = f"./experiments/{tag}/"
try:
    os.makedirs(our_path)
except FileExistsError:
    pass
copyfile("./.env", our_path+"/.env")


pd.DataFrame(hit_rate).to_csv(our_path+'hit_rate.csv',index=False)
pd.DataFrame(download_rate).to_csv(our_path+'download_rate.csv',index=False)
pd.DataFrame(hit_rate_ftpl).to_csv(our_path+'hit_rate_ftpl.csv',index=False)
pd.DataFrame(download_rate_ftpl).to_csv(our_path+'download_rate_ftpl.csv',index=False)

path = our_path


plt.plot(ma(err))
plt.title("Mean Squared Test Error in Demand Prediction vs Timeslot")
plt.xlabel("Timeslot")
plt.ylabel("MSE")
plt.savefig(path+"NN-MSE.jpg")
plt.show()


plt.plot(ma(queue))
plt.title("Q vs Timeslot")
plt.xlabel("Timeslot")
plt.ylabel("Q")
plt.savefig(path+"Q.jpg")
plt.show()


plt.plot(ma(objective))
plt.title("Constrained Objective Function vs Timeslot")
plt.xlabel("Timeslot")
plt.ylabel("Objective Function")
plt.savefig(path+"Obj.jpg")
plt.show()


plt.plot(ma(fetching_cost))
plt.title("Fetching Cost vs Timeslot")
plt.axhline(y=cost_constraint, linewidth=2, label= 'Cost Constraint')
plt.xlabel("Timeslot")
plt.ylabel("Cost")
plt.legend(loc = 'upper left')
plt.savefig(path+"Cost.jpg")
plt.show()


plt.plot(ma(cache_hit))
plt.title("Cache Hit vs Timeslot")
plt.xlabel("Timeslot")
plt.ylabel("Cache Hit")
plt.savefig(path+"Cache_Hit.jpg")
plt.show()