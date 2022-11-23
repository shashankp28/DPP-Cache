import math
import random
import numpy as np
import pandas as pd
from loader.load_data import *
from algorithms.offline_opt import MIN 
from algorithms.LFU import Bipartite_LFU
from algorithms.LRU import Bipartite_LRU
from algorithms.Lead_cache import Lead_cache
from optimizers.constrained import constrained_solve_ftpl
from algorithms.Perturbed_LFU import Perturbed_Bipartite_LFU
from algorithms.Generate_network import generate_network_graph


def run_algorithms(path_to_input, path, NumSeq, time_limit, threshold, alpha, cache_constraint):
    
    
    hit_rate_ftpl = []
    download_rate_ftpl = []
    gamma = np.random.normal(0, 1, (threshold,))
    prev_demands = [[0 for i in range(threshold)]]
    X_t_1_ftpl = np.zeros((threshold,))
    init_indices = random.sample(range(threshold), cache_constraint)
    X_t_1_ftpl[init_indices] = 1
    
        
    
    users = 1
    caches = 1
    d = 1
    # Dropping all file requests with id larger than the threshold to reduce the library size
    print("Users=", users, "caches=", caches, "Library_Size=", threshold, "time=", time_limit, "NumSeq=", NumSeq, file=open("parameters.log","w"))

    # generates a random network
    Adj = generate_network_graph(users, caches, d)

    # saves the network 
    print(Adj, file=open("network_adjacency_matrix.log", "w"))

    # Setting up the arrays to store hits and downloads over multiple runs
    LFU_Hits = []
    perturbed_LFU_Hits = []
    perturbed_LFU_Downloads = []
    LRU_Hits = []
    LeadCache_Hits = []
    LFU_Downloads = []
    LRU_Downloads = []
    LeadCache_Downloads = []
    OPT_Hits=[]
    OPT_Downloads = []
    LeadCache_Hits_Madow = []
    LeadCache_Downloads_Madow = []

    # Generating the request sequence
    #df=pd.read_csv('ratings.csv', sep=',',header=None)
    data = pd.read_csv(path_to_input, sep = ' ')
    #data = pd.read_csv("sigmetrics_truncated_data.txt", sep = ' ')
    #data = pd.read_csv("ratings1m.dat", sep = '::')
    #data.columns = ['User_ID', 'File_ID', 'Ratings', 'Timestamp']
    data.columns = ['Timestamp', 'File_ID', 'File_Size']
    DataLength = len(data)
    # splitting up the entire time axis into non-overlapping parts
    for i in range(NumSeq):
        
        df = pd.DataFrame(data[int(i*DataLength/NumSeq) : int((i+1)*DataLength/NumSeq)])
        df.sort_values("Timestamp")

        #df= pd.DataFrame(data)

        # The Data is already sorted according to the Req_ID, so no need to sort it again
        # Renaming the annoynimized FileID's
        old_id = df.File_ID.unique()
        old_id.sort()
        new_id = dict(zip(old_id, range(len(old_id))))
        df = df.replace({"File_ID": new_id})
        df.sort_values("Timestamp")

        # Reducing the library size

        df = df[df.File_ID < threshold]
        df = df.reset_index(drop=True)

        library_size = df['File_ID'].max()+2
        C = math.floor(alpha*library_size)
        v = df['File_ID']
        RawSeq = np.array(v)
        print(np.max(v), len(v))
        time = int(np.floor(min(time_limit, len(v)/users)))-1
        print(time)
        # RawSeq contains an array of requests
        df = np.array_split(RawSeq, users)

        # Running the algorithms

        print("-"*50)
        print("Iter:", i, "/", NumSeq)
        print()
        hit_rates_OPT, download_rate_OPT = MIN(df, Adj, time, library_size, C)
        hit_rates_OPT = pd.DataFrame(hit_rates_OPT)
        download_rate_OPT = pd.DataFrame(download_rate_OPT)
        print("Min Completed")
        print()

        OPT_Hits.append(np.sum(hit_rates_OPT)/(time*users))
        OPT_Downloads.append(np.sum(download_rate_OPT)/(time*caches))

        hit_rates_LFU, download_rate_LFU = Bipartite_LFU(
            df, Adj, time, library_size, C)
        hit_rates_LFU = pd.DataFrame(hit_rates_LFU)
        download_rate_LFU = pd.DataFrame(download_rate_LFU)
        print("BLFU Completed")
        print()


        LFU_Hits.append(np.sum(hit_rates_LFU)/(time*users))
        LFU_Downloads.append(np.sum(download_rate_LFU)/(time*caches))

        hit_rates_LRU, download_rate_LRU = Bipartite_LRU(
            df, Adj, time, library_size, C)
        hit_rates_LRU = pd.DataFrame(hit_rates_LRU)
        download_rate_LRU = pd.DataFrame(download_rate_LRU)
        print("BLRU Completed")
        print()

        LRU_Hits.append(np.sum(hit_rates_LRU)/(time*users))
        LRU_Downloads.append(np.sum(download_rate_LRU)/(time*caches))

        hit_rates_Perturbed_LFU, download_rate_Perturbed_LFU = Perturbed_Bipartite_LFU(
            df, Adj, time, library_size, C, d)
        hit_rates_Perturbed_LFU = pd.DataFrame(hit_rates_Perturbed_LFU)
        download_rate_Perturbed_LFU = pd.DataFrame(download_rate_Perturbed_LFU)

        perturbed_LFU_Hits.append(np.sum(hit_rates_Perturbed_LFU)/(time*users))
        perturbed_LFU_Downloads.append(np.sum(download_rate_Perturbed_LFU)/(time*caches))
        print("PBLRU Completed")
        print()



        #print("Running LeadCache")

        hit_rates_Lead_cache, download_rate_Lead_cache, hit_rates_Madow, download_rates_Madow = Lead_cache(
            df, Adj, time, library_size, C, d)
        hit_rates_Lead_cache = pd.DataFrame(hit_rates_Lead_cache)
        download_rate_Lead_cache = pd.DataFrame(download_rate_Lead_cache)

        LeadCache_Hits.append(np.sum(hit_rates_Lead_cache)/(time*users))
        LeadCache_Downloads.append(np.sum(download_rate_Lead_cache)/(time*caches))

        hit_rates_Madow = pd.DataFrame(hit_rates_Madow)
        download_rates_Madow = pd.DataFrame(download_rates_Madow)

        LeadCache_Hits_Madow.append(np.sum(hit_rates_Madow)/(time*users))
        LeadCache_Downloads_Madow.append(np.sum(download_rates_Madow)/(time*caches))
        print("LC Completed")
        print()
        
        
        next_dem, time = get_demands(i, time_limit, data, DataLength, NumSeq, threshold)
        
        X_t_ftpl = np.zeros((threshold,))
        X_t_ftpl[init_indices] = 1
        
        
        X_t_ftpl, obj_ftpl = constrained_solve_ftpl(np.array(prev_demands).sum(axis=0), X_t_1_ftpl, cache_constraint, gamma, threshold, i)
        
        hit_rate_ftpl.append(np.dot(X_t_ftpl, next_dem)/time)
        download_rate_ftpl.append(np.sum(np.logical_and(X_t_ftpl==1, X_t_1_ftpl==0))/time)
        
        X_t_1_ftpl = X_t_ftpl
        prev_demands.append(next_dem)
        print("FTPL Completed")
        print()
        
        


        # #Outputting the result to stdout
        #print("LFU Hits=", LFU_Hits, "LRU Hits=", LRU_Hits, "OPT_Hits=", OPT_Hits, "Perturbed_LFU_Hits=", perturbed_LFU_Hits)
        #print("LFU Downloads=", LFU_Downloads, "LRU Downloads=", LRU_Downloads, "OPT_Downloads=", OPT_Downloads, "Perturbed_LFU_Downloads=", perturbed_LFU_Downloads)


    # Saving the output files

    pd.DataFrame(LFU_Hits).to_csv(path+'LFU_Hits.csv',index=False)
    pd.DataFrame(LFU_Downloads).to_csv(path+'LFU_Downloads.csv',index=False)
    pd.DataFrame(LRU_Hits).to_csv(path+'LRU_Hits.csv',index=False)
    pd.DataFrame(LRU_Downloads).to_csv(path+'LRU_Downloads.csv',index=False)
    pd.DataFrame(perturbed_LFU_Hits).to_csv(path+'Perturbed_LFU_Hits.csv',index=False)
    pd.DataFrame(perturbed_LFU_Downloads).to_csv(path+'Perturbed_LFU_Downloads.csv',index=False) 
    pd.DataFrame(LeadCache_Hits).to_csv(path+'LeadCache_Hits.csv',index=False)
    pd.DataFrame(LeadCache_Downloads).to_csv(path+'LeadCache_Downloads.csv',index=False) 
    pd.DataFrame(LeadCache_Hits_Madow).to_csv(path+'LeadCache_Hits_Madow.csv',index=False)
    pd.DataFrame(LeadCache_Downloads_Madow).to_csv(path+'LeadCache_Downloads_Madow.csv',index=False) 
    pd.DataFrame(OPT_Hits).to_csv(path+'OPT_Hits.csv',index=False)
    pd.DataFrame(OPT_Downloads).to_csv(path+'OPT_Downloads.csv',index=False)
        
    # Saving the dynamic hit-rate and download logs

    pd.DataFrame(hit_rates_OPT).to_csv(path+'OPT_Hits_Seq.csv',index=False)
    pd.DataFrame(download_rate_OPT).to_csv(path+'OPT_Downloads_Seq.csv',index=False)
    pd.DataFrame(hit_rates_LRU).to_csv(path+'LRU_Hits_Seq.csv',index=False)
    pd.DataFrame(download_rate_LRU).to_csv(path+'LRU_Downloads_Seq.csv',index=False)
    pd.DataFrame(hit_rates_LFU).to_csv(path+'LFU_Hits_Seq.csv',index=False)
    pd.DataFrame(download_rate_LFU).to_csv(path+'LFU_Downloads_Seq.csv',index=False)
    pd.DataFrame(hit_rates_Lead_cache).to_csv(path+'LeadCache_Hits_Seq.csv',index=False)
    pd.DataFrame(download_rate_Lead_cache).to_csv(path+'LeadCache_Downloads_Seq.csv',index=False)
    pd.DataFrame(perturbed_LFU_Hits).to_csv(path+'perturbed_LFU_Hits_Seq.csv',index=False)
    pd.DataFrame(perturbed_LFU_Downloads).to_csv(path+'perturbed_LFU_Downloads_Seq.csv',index=False)

    pd.DataFrame(hit_rate_ftpl).to_csv(path+'hit_rate_ftpl.csv',index=False)
    pd.DataFrame(download_rate_ftpl).to_csv(path+'download_rate_ftpl.csv',index=False)