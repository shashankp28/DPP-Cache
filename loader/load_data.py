import numpy as np
import pandas as pd

def get_delta():
    return 1


def get_demands(seq_no, time_limit, data, DataLength, NumSeq, threshold):
    df = pd.DataFrame(data[int(seq_no*DataLength/NumSeq) : int((seq_no+1)*DataLength/NumSeq)])
    df.sort_values("Timestamp")
    old_id = df.File_ID.unique()
    old_id.sort()
    new_id = dict(zip(old_id, range(len(old_id))))
    df = df.replace({"File_ID": new_id})
    df.sort_values("Timestamp")
    df = df[df.File_ID < threshold]
    df = df.reset_index(drop=True)
    v = df['File_ID']
    RawSeq = np.array(v)
    demand_count = np.zeros((threshold,))
    m_time = int(np.floor(min(time_limit, len(v))))-1
    for i in range(m_time):
        demand_count[RawSeq[i]]+=1
    return demand_count, m_time

def ma(X, w=None):
    if(not w): w = len(X)
    avgs = []
    for i in range(len(X)):
        numer = np.sum(X[max(0, i-w+1):i+1])
        den = min(w, i+1)
        avgs.append(numer/den)
    return np.array(avgs)