from multiprocessing import Pool
from utils_run_models import generate_filenames, prefix_to_NT
import tqdm
import time
from models import run_model
from heuristic import simple_heuristic
from heuristicv2 import run_heuristic_v2
import pandas as pd
import random

start = time.time()
args = []

# fixed intervals 

for N in ['50']:#, '75', '100'] :
    for T in ['15']:#, '50', '60'] :
        for s in ['DD_DF']:#, 'DD_SF', 'SD_SF', 'SD_DF'] : 
            i = 'N' + N + 'T' + T + s 
            args.append(i)
        
def write_measures(path) :
    with open(path + '.csv', 'w') as f: 
            l = ['N', 'T', 'suffix', 'periods', 'method', 'obj', 'gap_obj', 'time', 'number_periods']
            for i in l : 
                f.write(i + ',')
            f.write("\n")

def write_info(path, N,T, filename, str_periods, obj, obj_ref, time, name, n):
    with open(path + '.csv', 'a') as f :
            for info in [ N,T, filename[-7:] , str_periods, name, obj, round(obj/ obj_ref * 100, 0), time, n ] : 
                f.write('%s,' %info)
            f.write("\n")

def is_solved(filename, path, n) :
    df = pd.read_csv(path + '.csv')
    df_dict = df.to_dict('index')
    prefix, suffix = filename[:-7], filename[-7:]
    N,T = prefix_to_NT(prefix)
    for row in df_dict.values():
        if (row['N'], row['T'], row['suffix'], row['number_periods']) == (N,T, suffix, n) :
            return(True)
    return(False)

def generate_random_periods(n, T):
    ordering_periods = [1] + random.sample(range(2, T+1), n-1) # n : number of periods in the interval
    return(ordering_periods)

def run(instance):
    """ runs a batch of ten instances
    input : 
    - prefix of instance (N50T30DD_DF)
    - models 'es'
    - pickperiods : same as in run_models
    - n same as in run model
    - method same as in run model
    """
    path = './Heuristic/results/random_periods' 
    print(instance)
    N,T = prefix_to_NT(instance[:-5])
    instances = generate_filenames(instance)  
    
    for k in [4, 6]:
        n = T//k
        for filename in instances : 
            if  is_solved(filename, path, n ):
                return(0)
            else : 
                ordering_periods  = generate_random_periods(n, T)
                str_periods = (str(ordering_periods).replace(',', '/'))
                obj_DP, time_DP = simple_heuristic(filename, ordering_periods)
                obj_MC, time_MC = run_model(filename, 'mc', 'fix_y', ordering_periods)
                obj_DP2, time_DP2 = run_heuristic_v2(filename, ordering_periods)
    

                for (obj, time, name) in [ (obj_DP, time_DP, 'simple_heuristic'), 
                                        (obj_DP2, time_DP2, 'heuristic_v2'), 
                                        (obj_MC, time_MC, 'mc')] :
                    write_info(path, N,T, filename, str_periods, obj, obj_MC, time, name, n)


                print(filename+  'finished')



def pool_handler():
    path = './Heuristic/results/random_periods'
    write_measures(path) 
    with Pool(4) as p:
      r = list(tqdm.tqdm(p.imap(run, args)))
    p.close()
    p.join()

if __name__ == '__main__':
    pool_handler()
    print(time.time() - start)

