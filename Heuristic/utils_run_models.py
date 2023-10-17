import numpy as np
from models import run_model
import pandas as pd
from tqdm.notebook import tqdm
from termcolor import colored
import random
import collections

# path are always without the extension ('.csv') when called in a function argument

def prefix_to_NT(prefix : str):
    """ takes format N100T25 and returns (100,25)"""
    T = int(prefix[-2:])
    N = int(prefix[1:-3])
    return(N,T)

def generate_random_periods(n, T):
    ordering_periods = [1] + random.sample(range(2, T+1), n-1) # n : number of periods in the interval
    return(ordering_periods)
        
def generate_filenames(instance : str) -> list:
    """ 
    input : 'N100T25DD_DF'
    genereate the ten filenames corresponding to one instance
    output : list of filename [instance01, instance02, ..., instance10]
    """
    l = []
    numbers = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    for i in numbers :
        l.append(instance+ i)
    return(l)

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
