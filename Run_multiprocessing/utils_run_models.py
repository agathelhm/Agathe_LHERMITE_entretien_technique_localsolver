import numpy as np
from models import run_model
import pandas as pd
from tqdm.notebook import tqdm
from termcolor import colored

# path are always without the extension ('.csv') when called in a function argument

def prefix_to_NT(prefix : str):
    """ takes format N100T25 and returns (100,25)"""
    T = int(prefix[-2:])
    N = int(prefix[1:-3])
    return(N,T)

def generate_filenames(instance : str) -> list:
    """ 
    input : 'N100T25DD_DF' instance
    genereate the ten filenames corresponding to one instance
    output : list of filename [instance01, instance02, ..., instance10]
    """
    l = []
    numbers = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    for i in numbers :
        l.append(instance+ i)
    return(l)

def recover_n(filename : str) -> int:
    """
    input :  an  filename "N50T15DD_DF01"
    output : n the number of set up periods at the warehouse for this filename 
    """
    df = pd.read_csv('/home/lheragat/Documents/STAGE/solved_files/01/results/solved.csv')
    x = df[(df['instance'] == filename)]
    i = np.array(x.index)[0]
    n  = df['n'][i]
    return(n)

def is_solved(path : str, prefix, suffix, model, method, pick_periods, n) -> bool:
    """
    checks files path.csv to see if instance is solved"""
    df = pd.read_csv(path + '.csv')
    df_dict = df.to_dict('index')
    N,T = prefix_to_NT(prefix)
    for row in df_dict.values():
        if (row['N'], row['T'], row['suffix'], row['model'], row['method'], row['pick_periods'], row['n']) == (N,T, suffix, model, method, pick_periods, n) :
            return(True)
    return(False)

def write_measure_names(path : str) -> None:
    """
    /!\ deletes previsous data from path.csv
    writes in path + '.csv' the list of measures
    """
    with open(path + '.csv', 'w') as f:
        list = [ 'N', 'T', 'suffix', 'model', 'method','pick_periods', 'n', 'objective', 'root_time', 'solving_time','MIP_gap', 'relaxation_gap','nodes',
                    
                    'proportion_of_set_up_costs', 'proportion_of_holding_costs',
                    'set_up_costs_proportion_at_warehouse', 'set_up_costs_proportion_at_retailers', 'nb_set_ups_at_warehouse', 
                    'nb_set_ups_at_retailers', 'holding_costs_proportion_warehouse', 'holding_costs_proportion_retailers' ,

                    'nb_variables', 'nb_binary_variables', 'nb_constraints' ] 
        for i in list : 
            f.write(i + ',')
        f.write("\n")

def write_model_info(a : list, path : str) :
    """
    write all the elements of a into the file path + '.csv'
    """
    with open(path + '.csv', 'a') as f :
        for i in range(len(a)) : 
            f.write('%s,' %a[i])
        f.write("\n")

def csv_to_excel(path : str):
    """
    transforms the file at path + '.csv' into an excel"""
    read_file = pd.read_csv(path + '.csv')
    read_file.to_excel(r'' + path + '.xlsx', index = None, header=True) 

def data_to_lists(model : str, method : str, pick_periods : str, n : int, m, root_time : float, relaxation_gap : float) -> list:
    """
    writes all of the data into a list, in the good order
    """
    stats = m.statistics
    details = m.solve_details
    objective = m.objective_value
    kpi = m.kpis_as_dict()
    l = [] 
    prefix, suffix = m.name[:-7], m.name[-7:]
    N,T = prefix_to_NT(prefix)

    l.append(str(N)) # filename
    l.append(str(T))
    l.append(suffix) # filename
    l.append(model) # model
    l.append(method) # method
    l.append(pick_periods)
    l.append(n)
    l.append(objective)

    l.append(root_time) # root time
    l.append(details.time) # solving time
    l.append(details.gap) # end gap
    l.append(relaxation_gap) # relaxation gap
    l.append(details.nb_nodes_processed) # nodes

    l.append(kpi['set_up_costs_for_facilities']/objective) # proportion of set up costs
    l.append(kpi['holding_costs']/objective) # proportion of holding costs

    l.append(kpi['set_up_costs_for_warehouse']/kpi['set_up_costs_for_facilities']) # proportion of set up @ warehouse
    l.append(kpi['set_up_costs_for_retailers']/kpi['set_up_costs_for_facilities']) # proportion of set up @ retailers

    l.append(kpi['nb_set_up_warehouse']) 
    l.append(kpi['nb_set_up_retailers']) 

    l.append(kpi['holding_costs_warehouse']/kpi['holding_costs']) # proportion of holding costs @ warehouse
    l.append(kpi['holding_costs_retailers']/kpi['holding_costs']) # proportion of holding costs @ retailers

    l.append(stats.number_of_variables)
    l.append(stats.number_of_binary_variables)
    l.append(stats.number_of_linear_constraints)

    return(l)

def filenames_to_instance(instance : str, model : str, pick_periods : str, n : int, method : str, path : str) -> None:
    """
    input : 
    - instance of the form N50T15DD_DF
    gets the data in path + '.csv' and writes the agregated data in path + '_mean.csv'
    """
    prefix, suffix = instance[:-5], instance[-5:]
    N,T = prefix_to_NT(prefix)
    df = pd.read_csv(path + '.csv')

    filenames = generate_filenames(instance)
    suffixes = [f[-7:] for f in filenames]
    df['suffix'] = df['suffix'].replace(suffixes, suffix) # replace suffix DD_DF01 with DD_DF so we can agregate the data

    indices = df[(df['N'] ==  N) & (df['T'] ==  T) & (df['suffix'] ==  suffix) & (df['model'] == model) & (df['method'] == method) & (df['pick_periods'] == pick_periods) & (df['n'] == n)].index
    x = indices.to_numpy() # get corresponding indices to the instance solved

    model_info = [] # get list of data
    for column in df.columns[:-1] :
        if type(df[column][x[0]]) in [str] :
            model_info.append(df[column][x[0]]) # if cannot apply mean than take first value
        elif column in ['N', 'T'] :
             model_info.append(df[column][x[0]]) # get int, otherwise float value
        else :
            model_info.append(np.mean(np.array([df[column][i] for i in x])))
    write_model_info(model_info, path + '_mean')

