from multiprocessing import Pool
from utils_run_models import *
import time
from tqdm.contrib.concurrent import process_map
import tqdm

start = time.time()
args = ()

# runs the models and save them in csv files

# list of instances we want to solve
for instance in ['N50T15'] : # , 'N75T15', 'N100T15', 'N50T20' , 'N75T20', 'N100T20','N50T25' , 'N75T25', 'N100T25'] : 
    for s in ['DD_DF']: #, 'DD_SF', 'SD_SF', 'SD_DF'] : 
        for model in   ['mc'] : #, 'sp', 'spc', 'tp', 'tpc', 'owmr', 'es', 'ses']  :
                i = instance + s 
                args = args + ((i, model, 'all', 0, 'not_sequential'),) 

def run(arg):
    instance, model, pick_periods, n, method = arg[0], arg[1], arg[2], arg[3], arg[4]
    """ runs a batch of ten instances
    input : 
    - prefix of instance (N50T30DD_DF)
    - models 'es'
    - pickperiods : same as in run_models
    - n same as in run model
    - method same as in run model

    output : saves data in two files 
    './results/' + pick_periods  + "mean.csv" : mean of the 10 instances with same caracteristics
    './results/' + pick_periods  + ".csv" : for each individual instance
    """
    path = './Run_multiprocessing/results/' + pick_periods 
    
    if is_solved(path + '_mean', instance[:-5], instance[-5:], model, method, pick_periods, n) : # test if 10 instance already solved
        print(colored('%s%s with model %s and method %s with %s pick order %s already solved' %(instance[:-5], instance[-5:], model, method, n, pick_periods), 'magenta'))
        return(0) # 10 instances already solved

    instances = generate_filenames(instance)  
    for filename in instances :  
        prefix, suffix = filename[:-7], filename[-7:]
        if  is_solved(path, prefix, suffix, model, method, pick_periods, n) : # filename already solved
            print(colored('%s%s with model %s and method %s with %s pick order %s already solved' %(prefix, suffix, model, method, n, pick_periods), 'green'))
        else : # solve instance and save info
            m, time_to_generate_model,gap  = run_model(filename, model, pick_periods, n, method)
            data = data_to_lists(model, method, pick_periods,n, m, time_to_generate_model,gap)
            write_model_info(data,path)
            print(colored('%s%s with model %s and method %s with %s pick order %s solved' %(prefix, suffix, model, method, n, pick_periods), 'red'))
    
    # calculate mean of the 10 instances
    if not is_solved(path + '_mean', instance[:-5], instance[-5:], model, method, pick_periods, n) :
        filenames_to_instance(instance, model, pick_periods, n, method, path)
        print(colored('%s%s with model %s and method %s with %s pick order %s finished' %(prefix, suffix, model, method, n, pick_periods), 'cyan')) 


def pool_handler():
    # attention write measure efface donnees
    write_measure_names('./Run_multiprocessing/results/all') 
    write_measure_names('./Run_multiprocessing/results/all_mean') 

    with Pool(4) as p: # split by processor / core
      r = list(tqdm.tqdm(p.imap(run, args)))
    p.close()
    p.join()

    
    path = './Run_multiprocessing/results/all' 
            
            # csv to excel
            #csv_to_excel(path)
            #csv_to_excel(path + '_mean')

    print('----beging testing----')
    # test if some of the methods do not yield the same solution
    models = []
    df = pd.read_csv(path + '.csv')
    for i in df.index:
        N = df['N'][i]  
        T = df['T'][i]
        suffix = df['suffix'][i]        
        model = df['model'][i]
        method = df['method'][i]
        n = df['n'][i]
        indices = df[(df['T'] ==  T) & (df['N'] ==  N) &(df['suffix'] == suffix)].index
        x = indices.to_numpy()
        for j in x :
            if abs(df['objective'][i] - df['objective'][j]) / df['objective'][i] > 1e-4 :
                print(N,T, suffix, model, method, n)


if __name__ == '__main__':
    pool_handler()
    print(time.time() - start)

