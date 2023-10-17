from docplex.mp.model import Model
import numpy as np
import time
from utils_models import *
from docplex.mp.relax_linear import LinearRelaxer
import random

def run_model(filename : str, model : str, pick_periods : str, n : int, method : str ) -> tuple:
    """
    input : 
    filename of dat file
    pick_periods : 'total_max', 'max_gap', 'min_gap', 'fixed_intervals', 'random', 'all' - method to decide on ordering periods 
    method in : 'fix_y_valid_inequalities' 'valid_inequalities', 'infinite_f', 'fix_y', 'finite_f', 'total_sum', 'consecutive_sum', 'infinite_f_new'
    n : int : parameter to decide ordering periods

    output : model, time on root node, relaxation gap
    """

    m = Model(name= filename, ignore_names=True)
    m.parameters.threads.set(1) # one processor
    m.set_time_limit(7200) # time limit of two hours

    # parameters 
    d, f, h, T, N, periods, facilities, retailers = load_parameters(filename) # recover parameters from dat file
    three_indices_retailers, three_indices_facilities,four_indices_retailers, four_indices_facilities = init_indices(retailers, facilities, periods, T) # sets of indices

    # ordering periods - are there constraints on the ordering periods ?
    ordering_periods_exists = False
    ordering_periods = []

    # create the authorized periods 
    if pick_periods == 'fixed_intervals' :
        ordering_periods = [1 + k * n for k in range(T//n+1)] # n : length of interval
    elif pick_periods == 'random' :
        ordering_periods = [1] + random.sample(range(2, T+1), n-1) # n : number of periods in the interval
    elif pick_periods == 'remove_random' :
        p = recover_periods(filename, 'owmr') # retrieve ordering periods at warehouse in optimal solution
        sample = p[1:]
        ordering_periods = [1] + random.sample(sample, len(sample) - n) # n : number of periods removed from optimal solution

    if pick_periods == 'random' or pick_periods== 'fixed_intervals' or  pick_periods == 'remove_random' : # there are constraints on the ordering periods
        non_ordering_periods = list(set(periods).difference(set(ordering_periods))) # calculate forbidden periods
        ordering_periods = list(set(ordering_periods) & set(periods))
        ordering_periods.sort()
        ordering_periods_exists = True

    # fixed periods - modify parameter f
    if ordering_periods_exists and method in ['infinite_f', 'infinite_f_new']  : # fix f to infinite value
        f_init = f.copy() 
        for t in non_ordering_periods :
            f[0,t] = m.infinity

    if ordering_periods_exists and method in ['finite_f']  :  # fix f to a finite value big M
        for k in non_ordering_periods :
            f[0,k] = 4500 + N * 100 + (n-1) * sum(d[i,t] for i in retailers for t in range(k,T+1))

    # Create models parameters and variables

    if model == 'owmr' :
        # variables
        x = m.continuous_var_matrix(facilities, periods, lb = 0, name = 'x')
        y = m.binary_var_matrix(facilities, periods, name = 'y')
        s = m.continuous_var_matrix(facilities, list(range(0,T+1)), lb = 0, name = 's')

        cons_owmr(m, s, x, y,d, retailers, periods, facilities)
        obj = obj_owmr(m, h, s, f, y, periods, retailers)

        kpi_owmr(m, h,s, periods, retailers)
 
    elif model == 'ses' :
        for t in periods :
            d[0,t] = sum(d[i,t] for i in retailers)
        H = init_H(T, N, periods, h, retailers)

        # variables
        X = m.continuous_var_dict(three_indices_facilities, lb = 0, name = 'X')
        y = m.binary_var_matrix(facilities, periods, name = 'y')

        cons_ses(m, X, d, y, T, periods, facilities, retailers, three_indices_facilities) 
        obj = obj_ses(m, y, X,H, f, periods, facilities, T)

        kpi_ses(m, model, f, h, y, X, retailers, facilities, periods, T, obj)
               
    elif model == 'sp':
        D = init_D(d,T,N,periods, facilities)
        G = init_G(T,N,periods, facilities,h,D)
        H_prime = init_H_prime(T,N,periods, facilities,h)
        
        a = np.where(D > 0, 1, 0)

        # variables
        Z = m.continuous_var_dict(three_indices_facilities, lb = 0, ub = 1, name = 'Z')
        U = m.continuous_var_dict(four_indices_facilities, lb = 0, name = 'U')
        y = m.continuous_var_matrix(retailers, periods, lb = 0, name = 'y')
        y_0 = m.binary_var_matrix([0],periods, name = 'y_0')

        cons_sp(m, U, a, y_0, Z, y, periods, T, retailers, three_indices_facilities)
        obj = obj_sp(m,y_0, y, Z, U, T, G, f, D, periods, retailers, H_prime)

        kpi_sp(m, G, Z, T, periods, retailers, H_prime, D, U)
         
    elif model == 'es':
        for t in periods :
            d[0,t] = sum(d[i,t] for i in retailers)

        D = init_D(d,T,N,periods, facilities)

        # variables
        Q = m.continuous_var_matrix(facilities, periods, lb = 0, name = 'Q')
        I = m.continuous_var_matrix(facilities, list(range(0,T+1)), lb = 0, name = 'I')
        y = m.binary_var_matrix(facilities, periods, name = 'y')

        cons_es(m, I, Q, d, y, T, periods, facilities, D, retailers)
        obj = obj_es(m, y, I, h, f, facilities, periods, retailers)

        kpi_es(m, h, I, periods, retailers)
   
    elif model == 'tp':
        H_prime = init_H_prime(T,N,periods, facilities,h)

        # variables
        W = m.continuous_var_dict(four_indices_facilities, lb = 0, name = 'W')
        X = m.continuous_var_dict(three_indices_facilities, lb = 0, name = 'X')
        y = m.continuous_var_matrix(retailers, periods, lb = 0, name = 'y')
        y_0 = m.binary_var_matrix([0],periods, name = 'y_0')    

        cons_tp(m, X, y_0, d, y, W, retailers, periods, three_indices_retailers) 
        obj = obj_tp(m, f, y, retailers, periods, y_0, T, H_prime, X, W)

        kpi_tp(m, H_prime, retailers, periods, W, X, T)            
    
    elif model == 'spc' :  
        H_prime = init_H_prime(T,N,periods, facilities,h)
        D = init_D(d,T,N,periods, facilities)
        G = init_G(T,N,periods, facilities,h,D)
        a = np.where(D > 0, 1, 0)
        
        # variables
        U = m.continuous_var_dict(four_indices_retailers, lb = 0, name = 'U')
        y = m.continuous_var_matrix(retailers, periods, lb = 0, name = 'y')
        y_0 = m.binary_var_matrix([0],periods, name = 'y_0')

        cons_spc(m, periods, retailers, U, a, y_0, y, T, three_indices_retailers )
        obj = obj_spc(m, f, y , y_0, T, D, U, G, H_prime, retailers, periods)

        kpi_spc(m, H_prime, D, U, T, G, periods, retailers)    

    elif model == 'tpc':
        H_prime = init_H_prime(T,N,periods, facilities,h)
        H_chapeau = init_H_chapeau(T,N,periods, facilities, H_prime)

        # variables    
        W = m.continuous_var_dict(four_indices_facilities, lb = 0, name = 'W')
        y = m.continuous_var_matrix(retailers, periods, lb = 0, name = 'y')
        y_0 = m.binary_var_matrix([0],periods, name = 'y_0')

        cons_tpc(m, W, d, y_0, y,retailers, periods, three_indices_retailers)
        obj = obj_tpc(m, y, y_0, periods, retailers, f, T, W, H_chapeau)

        kpi_tpc(m, H_prime, periods, retailers, T, W)

    elif model == 'mc':
        # variables
        y = m.continuous_var_matrix(retailers, periods, lb = 0, ub = 1, name = 'y')
        y_0 = m.binary_var_matrix([0],periods, name = 'y_0')
        w_0 = m.continuous_var_dict(three_indices_retailers, lb = 0, name = 'w_0')
        w_1 = m.continuous_var_dict(three_indices_retailers, lb = 0, name = 'w_1')
        sigma_0 = m.continuous_var_dict([(i,t,k) for i in retailers for t in range(T+1) for k in range(t, T+1)], lb = 0, name = 'sigma_0')
        sigma_1 = m.continuous_var_dict([(i,t,k) for i in retailers for t in range(T+1) for k in range(t, T+1)], lb = 0, name = 'sigma_1')
        x = m.continuous_var_matrix(facilities, periods, lb = 0, name = 'x')

        delta = np.eye(T+1)     

        cons_mc(m, sigma_0, sigma_1, w_0, w_1, x, y,y_0, three_indices_retailers, periods, retailers, T, d, delta)
        obj = obj_mc(m, h, sigma_0, retailers, T, f, y_0, periods, sigma_1, y)  

        kpi_mc(m, h,sigma_0, sigma_1, periods, retailers, T)
    
    else : 
        print('model %s not known' %model)

    m.set_objective('min', obj )

    # fixed periods - add constraints y = 0 when forbidden period
    if ordering_periods_exists and method in ['fix_y'] :
        if model in ['owmr', 'es', 'ses'] : # model with variable y :
            m.add_constraints((y[0,t] == 0 for t in non_ordering_periods), names = 'periods')
        else : # model with variable y_0
            m.add_constraints((y_0[0,t] == 0 for t in non_ordering_periods), names = 'periods')    

    if method in ['valid_inequalities'] and pick_periods == 'fixed_intervals': # method valid inequalities only works with fixed intervals
        if model == 'owmr':
            valid_inequalities_owmr(m, s, retailers, ordering_periods, n, T, d)
        elif model == 'es':
           valid_inequalities_es(m, I, n, d, retailers, ordering_periods, T)
        elif model == 'ses':
            valid_inequalities_ses(m, X, ordering_periods, d, T, retailers, n)
        elif model in ['tp', 'tpc']:
            valid_inequalities_tp_tpc(m, W, retailers, d, ordering_periods, T, n)
        elif model in ['sp', 'spc'] :
            valid_inequalities_sp_spc(m, U, d, retailers, ordering_periods, n, T)
        elif model == 'mc':
            valid_inequalities_mc(m, sigma_0, sigma_1, retailers, ordering_periods, T, n, d)

    if pick_periods == 'total_max' : # authorize only a maximum amount of production periods
        if model in ['owmr', 'es', 'ses'] :
            f_total_max(m, method, model, periods, n, T, y)
        else : 
            f_total_max(m, method, model, periods, n, T, y, y_0)
        
    if pick_periods == 'min_gap' :
        if model in ['owmr', 'es', 'ses'] :# model avec variable y :
            f_min_gap(m, method, model, periods, T, n, y)
        else :
            f_min_gap(m, method, model, periods, T, n, y, y_0)

    if pick_periods == 'max_gap' :
        if model in ['owmr', 'es', 'ses'] :# model avec variable y :
            f_max_gap(m, model, method, periods, T, n, y)
        else :
           f_max_gap(m, model, method, periods, T, n, y, y_0)
         
    # kpi
    if model in ['owmr', 'es', 'ses'] :# model avec variable y :
        get_kpis(m, obj, model, f,periods, facilities, retailers, y)
    else :
        get_kpis(m, obj, model, f,periods, facilities, retailers, y, y_0) 
            
    m.solve()

    # relax model and solve it
    gap, root_time = relaxed_model(m)
    
    return(m, root_time, gap)