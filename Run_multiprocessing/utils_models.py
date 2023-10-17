import numpy as np
from docplex.mp.relax_linear import LinearRelaxer
import pandas as pd

def load_parameters(filename : str):
    """
    filename under the form N15T30DD_DF01
    loads parameters from dat files and returns then under numpy arrays
    """
    # parameters 
    path = 'data/'
    d = np.load(path + filename + '/demand.npy') # demand
    f = np.load(path + filename + '/set_up.npy') # set up costs
    h = np.load(path + filename + '/holding.npy') # holding costs

    T, N = d.shape[1]-1, d.shape[0]-1

    periods = list(range(1,T+1))
    facilities = list(range(N+1))
    retailers = list(range(1, N+1))

    return(d, f, h, T, N, periods, facilities, retailers)

def relaxed_model(model_mip) :
    """
    input : model MIP
    output : relative gap between the relaxation and the MIP formulation, solving time for relaxaed model
    if model not solvable, infinity for the gap
    """
    lp = LinearRelaxer.make_relaxed_model(model_mip)
    lp.solve()
    try :
        gap = abs(lp.objective_value - model_mip.objective_value) / model_mip.objective_value
        return(gap, lp.solve_details.time)
    except :
        gap = lp.infinity
        return(lp.infinity, -1)

def recover_periods(filename : str, model : str) -> str :
    """
    takes an filename N15T30DD_DF01 and returns the list of the warehouse ordering periods in the objective solution
    need to have the ordering periods saved in a file 
    """
    df = pd.read_csv('/home/lheragat/Documents/STAGE/ns/solved.csv')
    x = df[(df['instance'] == filename) & (df['model'] == model)]
    i = np.array(x.index)[0]
    periods  = df['periods'][i]
    l = str(periods).replace('/', ',')
    lst = eval(l)
    return(lst)

# parameters

def init_indices(retailers, facilities, periods, T):
    # indices
    three_indices_retailers = [(i,t,k) for i in retailers for t in periods for k in range(t, T+1)]
    three_indices_facilities = [(i,t,k) for i in facilities for t in periods for k in range(t, T+1)]

    four_indices_retailers= [(i,q,t,k) for i in retailers for q in periods for t in range(q, T+1) for k in range(t, T+1)]
    four_indices_facilities = [(i,q,t,k) for i in facilities for q in periods for t in range(q, T+1) for k in range(t, T+1)]
    return(three_indices_retailers,three_indices_facilities,four_indices_retailers,four_indices_facilities )

# init parameters for models

def init_D(d,T,N,periods, facilities):
    D = np.zeros((N+1,T+1,T+1))
    for i in facilities :
        for t in periods :
            for k in range(t, T+1) :
                D[i,t,k] = sum(d[i,r] for r in range(t, k + 1))
    return(D)

def init_G(T,N,periods, facilities,h,D):
    G = np.zeros((N+1,T+1,T+1))
    for i in facilities :
        for t in periods :
            for k in range(t, T+1) :
                G[i,t,k] = sum(h[i] * D[i,l+1,k] for l in range(t, k))
    return(G)

def init_H_prime(T,N,periods, facilities,h):
    H_prime = np.zeros((N+1,T+1,T+1))
    for i in facilities :
        for t in periods :
            for k in range(t, T+1) :
                H_prime[i,t,k] = sum(h[i] for l in range(t, k))
    return(H_prime)

def init_H_chapeau(T,N,periods, facilities, H_prime):
    H_chapeau = np.zeros((N+1,T+1,T+1, T+1))
    for i in facilities :
        for q in periods: 
            for t in range(q, T+1) :
                for k in range(t, T+1) :
                    H_chapeau[i,q,t,k] = H_prime[0,q,t] + H_prime[i,t,k]
    return(H_chapeau)

def init_H(T, N, periods, h, retailers):
    H = np.zeros((N+1,T+1,T+1))
    for t in periods :
        for k in range(t, T+1) :
            H[0,t,k] = sum(h[0] for l in range(t, k))

    for i in retailers :
        for t in periods :
            for k in range(t, T+1) :
                H[i,t,k] = sum(h[i] - h[0] for l in range(t, k))
    return(H)

# owmr
def cons_owmr(m, s, x, y, d, retailers :list, periods : list, facilities : list):
    M = sum(sum(d))

    m.add_constraints((s[0, t-1] + x[0, t] == m.sum(x[c,t] for c in retailers) + s[0,t] 
                            for t in periods),
                            names = 'balance_prod')
    m.add_constraints((s[c, t-1] + x[c, t] == d[c,t] + s[c,t] 
                            for t in periods for c in retailers), 
                            names = 'balance_r')
    m.add_constraints((x[0,t] <= M * y[0,t] 
                            for t in periods), 
                            names = 'set_up_prod')
    m.add_constraints((x[c,t] <= M * y[c,t] 
                            for t in periods for c in retailers), 
                            names = 'set_up_r')
    m.add_constraints((s[c,0] == 0  for c in facilities), names = 'initial_stock')

def obj_owmr(m, h, s, f, y, periods, retailers):
    obj = m.sum(h[0] * s[0,t] + f[0, t] * y[0,t] for t in periods) + m.sum(h[c]* s[c,t] + f[c,t]* y[c,t] for t in periods for c in retailers)
    return(obj)

# es
def cons_es(m, I, Q, d, y, T, periods, facilities, D, retailers):
# constraints
    m.add_constraints((I[i, t-1] + Q[i,t] == d[i,t] + I[i,t] 
                            for t in periods for i in facilities), 
                            names = 'balance')

    m.add_constraints((Q[i,t] <= D[i,t,T] * y[i,t] 
                            for t in periods for i in facilities), 
                            names = 'set_up')

    m.add_constraints((m.sum(Q[0,r] for r in range(1,t+1)) >= m.sum(Q[i,r] for i in retailers for r in range(1,t+1)) 
                            for t in periods), 
                            names = 'cons4')

    m.add_constraints((I[c,0] == 0  for c in facilities), 
                        names = 'initial_stock') 

def obj_es(m, y, I, h, f, facilities, periods, retailers):
    obj = m.sum(f[i,t] * y[i,t] for i in facilities for t in periods) + m.sum( h[0] * I[0,t] for t in periods) + m.sum((h[i] - h[0])*I[i,t] for i in retailers for t in periods)
    return(obj)  

# ses
def cons_ses(m, X, d, y, T, periods, facilities, retailers, three_indices_facilities):
    m.add_constraints(( m.sum(X[i,t,k] for t in range(1, k+1)) 
                            == d[i,k] 
                            for k in periods for i in facilities), 
                            names = 'order_demand')

    m.add_constraints(( X[i,t,k] <= d[i,k] * y[i,t]
                            for (i,t,k) in three_indices_facilities), 
                            names = 'set_up')

    m.add_constraints((m.sum(X[0,r,k] for r in range(1, t+1) for k in range(r, T+1)) >= 
                            m.sum(X[i,r,k] for i in retailers for r in range(1,t+1) for k in range(r, T+1)) 
                            for t in periods), 
                        names = 'cons11')  

def obj_ses(m, y, X,H, f, periods, facilities, T):
    obj = m.sum(f[i,t] * y[i,t] for i in facilities for t in periods) + m.sum(m.sum(H[i,t,k] * X[i,t,k] for k in range(t,T+1)) for i in facilities for t in periods) 
    return(obj)

# tp
def cons_tp(m, X, y_0, d, y, W, retailers, periods, three_indices_retailers):
    m.add_constraints(( m.sum(W[i,q,t,k] for q in range(1,t+1)) == X[i,t,k] for (i,t,k) in three_indices_retailers ), 
                        names = 'def_W')

    m.add_constraints(( m.sum(W[i,q,t,k] for t in range(q, k+1))<= d[i,k] * y_0[0,q] for (i,q,k) in three_indices_retailers), 
                        names = 'set_up_w')

    m.add_constraints((m.sum(X[i,t,k] for t in range(1,k+1)) == d[i,k] for i in retailers for k in periods), 
                    names = 'order_everything')

    m.add_constraints(( X[i,t,k] <= d[i,k] * y[i,t]
                        for (i,t,k) in three_indices_retailers), 
                        names = 'set_up')

def obj_tp(m, f, y, retailers, periods, y_0, T, H_prime, X, W):
    obj = m.sum(f[i,t] * y[i,t] for i in retailers for t in periods) + m.sum(f[0,t] * y_0[0,t] for t in periods) + m.sum(
            m.sum(m.sum(H_prime[0,q,t] * W[i,q,t,k] for k in range(t,T+1)) for t in range(q,T+1)) 
            for i in retailers for q in periods) + m.sum(
            m.sum(H_prime[i,t,k] * X[i,t,k] for k in range(t, T+1)) for i in retailers for t in periods )
    return(obj)

# tpc
def obj_tpc(m, y, y_0, periods, retailers, f, T, W, H_chapeau):
    obj = m.sum(f[i,t] * y[i,t] for i in retailers for t in periods) + m.sum(f[0,t] * y_0[0,t] for t in periods) + m.sum(m.sum(
            m.sum(H_chapeau[i,q,t,k] * W[i,q,t,k] for k in range(t, T+1)) for t in range(q, T+1)) for q in periods for i in retailers)
    return(obj)
  
def cons_tpc(m, W, d, y_0, y,retailers, periods, three_indices_retailers):
    m.add_constraints(( m.sum(W[i,q,t,k] for t in range(q, k+1))<= d[i,k] * y_0[0,q] for (i,q,k) in three_indices_retailers), 
                            names = 'cons15')
    m.add_constraints(( m.sum(m.sum(W[i,q,t,k] for q in range(1,t+1))  for t in range(1,k+1)) == d[i,k] for i in retailers for k in periods), 
                        names = 'def_W') 
    m.add_constraints((m.sum(W[i,q,t,k] for q in range(1,t+1)) <= d[i,k] * y[i,t] for (i,t,k) in three_indices_retailers), 
                        names = 'cons24')

# sp
def obj_sp(m,y_0, y, Z, U, T, G, f, D, periods, retailers, H_prime):
    obj = m.sum(f[i,t] * y[i,t] for i in retailers for t in periods) + m.sum(f[0,t] * y_0[0,t] for t in periods) + m.sum(
            m.sum(H_prime[0,q,t] * D[i,t,k] * U[i,q,t,k] for q in range(1, t+1) for k in range(t, T+1)) for t in periods for i in retailers) + m.sum(
            m.sum(G[i,t,k]* Z[i,t,k] for k in range(t, T+1)) for t in periods for i in retailers)
    return(obj)

def cons_sp(m, U, a, y_0, Z, y, periods, T, retailers, three_indices_facilities):
    m.add_constraints(( m.sum(U[i,q,t,k] for q in range(1,t+1)) == Z[i,t,k] for (i,t,k) in three_indices_facilities), 
                            names = 'def_U')

    m.add_constraints(( m.sum(a[i,k,r] * U[i,q,k,r] for r in range(t,T+1) for k in range(q, t+1)) <=y_0[0,q] for (i,q,t) in three_indices_facilities), 
                            names = 'set_up')

    m.add_constraints((m.sum(Z[i,1,t] for t in periods) == 1 for i in retailers), 
                            names = 'fraction')

    m.add_constraints((m.sum(Z[i,t,k] for k in range(t, T+1)) - m.sum(Z[i,k,t-1] for k in range(1, t)) == 0 for i in retailers for t in range(2, T+1)), 
                            names = 'flux')
        
    m.add_constraints((m.sum(Z[i,t,k] * a[i,t,k] for k in range(t,T+1)) <= y[i,t] for i in retailers for t in periods), 
                            names = 'fraction2')

# spc
def cons_spc(m, periods, retailers, U, a, y_0, y, T, three_indices_retailers ):
    m.add_constraints((m.sum(U[i,1,1,t] for t in periods) == 1 for i in retailers))
        
    m.add_constraints((m.sum(m.sum(U[i,q,t,k] for q in range(1,t+1)) for k in range(t, T+1)) == m.sum(m.sum(U[i,q,k,t-1] for q in range(1,k+1)) for k in range(1,t)) for i in retailers for t in range(2, T+1)))

    m.add_constraints((m.sum(m.sum(a[i,t,k]* U[i,q,t,k] for q in range(1,t+1)) for k in range(t, T+1)) <= y[i,t] for i in retailers for t in periods))

    m.add_constraints((m.sum(a[i,k,r] * U[i,q,k,r] for r in range(t,T+1) for k in range(q, t+1)) <=y_0[0,q] for (i,q,t) in three_indices_retailers))
       
def obj_spc(m, f, y , y_0, T, D, U, G, H_prime, retailers, periods):
    obj = m.sum(f[i,t] * y[i,t] for i in retailers for t in periods) + m.sum(f[0,t] * y_0[0,t] for t in periods) + m.sum(m.sum(
            m.sum((H_prime[0,q,t]*D[i,t,k] + G[i,t,k]) * U[i,q,t,k] for q in range(1, t+1)) for k in range(t, T+1)) for t in periods for i in retailers)
    return( obj)
        
# mc 

def cons_mc(m, sigma_0, sigma_1, w_0, w_1, x, y,y_0, three_indices_retailers, periods, retailers, T, d, delta):
    m.add_constraints((sigma_0[i, k-1, t] + w_0[i,k,t] == w_1[i,k,t] + sigma_0[i,k,t] 
                   for (i,k,t) in three_indices_retailers), names = ' cons60')

    m.add_constraints((sigma_1[i, k-1, t] + w_1[i,k,t] == delta[k,t]*d[i,t] + (1-delta[k,t])* sigma_1[i,k,t] 
                    for (i,k,t) in three_indices_retailers), names = 'cons61')

    m.add_constraints((w_0[i,k,t] <= y_0[0,k] * d[i,t] 
                    for (i,k,t) in three_indices_retailers), names = 'cons62')

    m.add_constraints((w_1[i,k,t] <= y[i,k]*d[i,t] 
                    for (i,k,t) in three_indices_retailers), names = 'cons63')

    m.add_constraint((m.sum(sigma_0[c,0, t] for t in periods for c in retailers) == 0 ))

    m.add_constraints((m.sum(sigma_1[c,0, t] for t in periods) == 0  for c in retailers), names = 'initial_stock_retailers')

    m.add_constraints((m.sum(w_0[i,k,t] for t in range(k, T+1) for i in retailers) == x[0,k] for k in periods), names = 'cons64')

    m.add_constraints((m.sum(w_1[i,k,t] for t in range(k,T+1)) == x[i,k] for i in retailers for k in periods), names = 'cons65')

def obj_mc(m, h, sigma_0, retailers, T, f, y_0, periods, sigma_1, y):
    obj = m.sum(h[0] * m.sum(sigma_0[c,t,l] 
                        for c in retailers for l in range(t,T+1)) + f[0, t]*y_0[0,t] for t in periods) + m.sum(h[c]* m.sum(sigma_1[c,t,l] for l in range(t, T+1)) + f[c,t]* y[c,t] 
                                                                                                               for t in periods for c in retailers)
    return(obj)

# kpis

def get_kpis(m, obj, model, f,periods, facilities, retailers, y, y_0 = None):
    if model in ['owmr', 'es', 'ses'] : # model with variable y :
            m.add_kpi(m.sum(f[i,t] * y[i,t] for i in facilities for t in periods), 'set_up_costs_for_facilities')
            m.add_kpi((obj -m.sum(f[i,t] * y[i,t] for i in facilities for t in periods)), 'holding_costs' )
            m.add_kpi((m.sum(f[0,t] * y[0,t] for t in periods)), 'set_up_costs_for_warehouse')
            m.add_kpi(m.sum(y[0,t] for t in periods), 'nb_set_up_warehouse')

    else :
        m.add_kpi((m.sum(f[i,t] * y[i,t] for i in retailers for t in periods) + m.sum(f[0,t] * y_0[0,t] for t in periods)),'set_up_costs_for_facilities' )
        m.add_kpi((obj - (m.sum(f[i,t] * y[i,t] for i in retailers for t in periods) + m.sum(f[0,t] * y_0[0,t] for t in periods))), 'holding_costs' )
        m.add_kpi((m.sum(f[0,t] * y_0[0,t] for t in periods)), 'set_up_costs_for_warehouse')
        m.add_kpi(m.sum(y_0[0,t] for t in periods), 'nb_set_up_warehouse')
    
    m.add_kpi(m.sum(y[i,t] for i in retailers for t in periods), 'nb_set_up_retailers')
    m.add_kpi((m.sum(f[i,t] * y[i,t] for i in retailers for t in periods)), 'set_up_costs_for_retailers')

# kpi holding costs 

def kpi_owmr(m, h,s, periods, retailers):
    m.add_kpi(m.sum(h[0]* s[0,t] for t in periods), 'holding_costs_warehouse')
    m.add_kpi(m.sum(h[c]* s[c,t] for t in periods for c in retailers), 'holding_costs_retailers')

def kpi_es(m, h, I, periods, retailers ):
    m.add_kpi( m.sum(h[i] * I[i,t] for t in periods for i in retailers), 'holding_costs_retailers')
    m.add_kpi( m.sum(h[0]*I[0,t] for t in periods) - m.sum(h[0]* I[i,t] for t in periods for i in retailers), 'holding_costs_warehouse')

def kpi_ses(m, model, f, h, y, X, retailers, facilities, periods, T, obj) :  
    m.add_kpi(m.sum(m.sum(m.sum(h[i]* X[i,t,k] for l in range(t, k)) for k in range(t, T+1)) for t in periods for i in retailers), 'holding_costs_retailers')
    m.add_kpi((obj -m.sum(f[i,t] * y[i,t] for i in facilities for t in periods) - (m.sum(m.sum(m.sum(h[i]* X[i,t,k] for l in range(t, k)) for k in range(t, T+1)) for t in periods for i in retailers))), 'holding_costs_warehouse')
        
def kpi_tp(m, H_prime, retailers, periods, W, X, T):
    m.add_kpi( m.sum(m.sum(H_prime[i,t,k] * X[i,t,k] for k in range(t, T+1)) for i in retailers for t in periods ), 'holding_costs_retailers')
    m.add_kpi( m.sum(m.sum(m.sum(H_prime[0,q,t] * W[i,q,t,k] for k in range(t,T+1)) for t in range(q,T+1)) 
            for i in retailers for q in periods), 'holding_costs_warehouse')
        
def kpi_tpc(m, H_prime, periods, retailers, T, W):
    m.add_kpi( m.sum(m.sum(
        m.sum(H_prime[0,q,t] * W[i,q,t,k] for k in range(t, T+1)) for t in range(q, T+1)) for q in periods for i in retailers), 'holding_costs_warehouse')
    m.add_kpi( m.sum(m.sum(
        m.sum(H_prime[i,t,k] * W[i,q,t,k] for k in range(t, T+1)) for t in range(q, T+1)) for q in periods for i in retailers), 'holding_costs_retailers')

def kpi_sp(m, G, Z, T, periods, retailers, H_prime, D, U):
    m.add_kpi(m.sum(m.sum(G[i,t,k]* Z[i,t,k] for k in range(t, T+1)) for t in periods for i in retailers), 'holding_costs_retailers')
    m.add_kpi(m.sum(
        m.sum(H_prime[0,q,t] * D[i,t,k] * U[i,q,t,k] for q in range(1, t+1) for k in range(t, T+1)) for t in periods for i in retailers) , 'holding_costs_warehouse')
        
def kpi_spc(m, H_prime, D, U, T, G, periods, retailers) :   
    m.add_kpi( m.sum(m.sum(
        m.sum(H_prime[0,q,t]*D[i,t,k] * U[i,q,t,k] for q in range(1, t+1)) for k in range(t, T+1)) for t in periods for i in retailers), 'holding_costs_warehouse')
    m.add_kpi( m.sum(m.sum(
        m.sum(G[i,t,k] * U[i,q,t,k] for q in range(1, t+1)) for k in range(t, T+1)) for t in periods for i in retailers), 'holding_costs_retailers')       

def kpi_mc(m, h,sigma_0, sigma_1, periods, retailers, T):
    m.add_kpi(m.sum(h[0]* m.sum(sigma_0[c,t,l] 
                        for c in retailers for l in range(t,T+1)) for t in periods), 'holding_costs_warehouse')
    m.add_kpi(m.sum(h[c]* m.sum(sigma_1[c,t,l] for l in range(t, T+1)) for t in periods for c in retailers), 'holding_costs_retailers')
    
# total max

def f_total_max(m, method, model, periods, n, T, y, y_0 = None):
    if method == 'total_sum' :
        if model in ['owmr', 'es', 'ses'] : # model with variable y :
            m.add_constraint((m.sum(y[0,t] for t in periods) <= n))
        else :
            m.add_constraint((m.sum(y_0[0,t] for t in periods) <= n))

    elif method == 'consecutive_sum' :
        v = m.integer_var_matrix([0], periods, lb = 0, name = 'v')
        m.add_constraint(v[0,1] == 1)
        m.add_constraint(v[0,T] <= n)

        if model in ['owmr', 'es', 'ses'] : # model with variable y :
            m.add_constraints((v[0,t] - v[0, t-1] == y[0,t] for t in range(2,T+1))) # definition of v
        else :
            m.add_constraints((v[0,t] - v[0, t-1] == y_0[0,t] for t in range(2,T+1))) # definition of v

# min gap

def f_min_gap(m, method, model, periods, T, n, y, y_0 = None):
    if method == 'total_sum' :
        if model in ['owmr', 'es', 'ses'] :# model avec variable y :
            m.add_constraints((m.sum(y[0,k] for k in range(t, t+n) ) <= 1 for t in range(1, T-n + 2)), names = 'periods')
        else :
            m.add_constraints((m.sum(y_0[0,k] for k in range(t, t+ n) ) <= 1 for t in range(1, T-n + 2)), names = 'periods')
    elif method == 'consecutive_sum' :
        z = m.integer_var_matrix([0], periods,  name = 'z')
        v = m.integer_var_matrix([0], list(range(T+1)),lb = -T, name = 'v')
        m.add_constraints((z[0,t] <= v[0,t-1] for t in periods))
        m.add_constraint(v[0,0] == 0)
        m.add_constraints( (v[0,t-1] - z[0,t] <= T-n + 1 for t in periods))

        if model in ['owmr', 'es', 'ses'] :# model avec variable y :
            m.add_constraints((z[0,t] <= (1-y[0,t]) * T for t in periods))
            m.add_constraints((z[0,t] >= v[0,t-1] - y[0,t] * T for t in periods))
            m.add_constraints(( z[0,t] - v[0,t] == 1 - (T+1) * y[0,t] for t in periods))

        else :
            m.add_constraints((z[0,t] <= (1-y_0[0,t]) * T for t in periods))
            m.add_constraints((z[0,t] >= v[0,t-1] - y_0[0,t] * T for t in periods))
            m.add_constraints(( z[0,t] - v[0,t] == 1 - (T+1) * y_0[0,t] for t in periods))

# max gap

def f_max_gap(m, model, method, periods, T, n, y, y_0 = None):
    if method == 'total_sum' :
        if model in ['owmr', 'es', 'ses'] :# model avec variable y :
            m.add_constraints((m.sum(y[0,k] for k in range(t, t+n) ) >= 1 for t in range(1, T-n + 2)), names = 'periods')
        else :
            m.add_constraints((m.sum(y_0[0,k] for k in range(t, t+n) ) >= 1 for t in range(1, T-n + 2 )), names = 'periods')
    elif method == 'consecutive_sum' :
        z = m.integer_var_matrix([0], periods,  name = 'z')
        v = m.integer_var_matrix([0], list(range(T+1)),lb = 0, name = 'v')
        m.add_constraints((z[0,t] <= v[0,t-1] for t in periods))
        m.add_constraints((v[0,t] - z[0,t] == 1 for t in periods))
        m.add_constraint(v[0,0] == 0)
        m.add_constraints((v[0,t] <= n for t in periods))

        if model in ['owmr', 'es', 'ses'] :# model avec variable y :
            m.add_constraints((z[0,t] <= (1-y[0,t]) * T for t in periods))
            m.add_constraints((z[0,t] >= v[0,t-1] - y[0,t] * T for t in periods))
        else :
            m.add_constraints((z[0,t] <= (1-y_0[0,t]) * T for t in periods))
            m.add_constraints((z[0,t] >= v[0,t-1] - y_0[0,t] * T for t in periods))           
            
# valid inequalities

def valid_inequalities_owmr(m, s, retailers, ordering_periods, n, T, d):
    m.add_constraints((s[0,t] + m.sum(s[c,t] for c in retailers) - s[0,t+n-1] - m.sum(s[c,t+n-1] for c in retailers) >= m.sum(d[c,k] for c in retailers for k in range(t+1, t + n)) for t in ordering_periods[:-1]))
    t = ordering_periods[-1]
    m.add_constraint((s[0,t] + m.sum(s[c,t] for c in retailers) - s[0,T] - m.sum(s[c,T] for c in retailers) >= m.sum(d[c,k] for c in retailers for k in range(t+1, T+1))))
        
def valid_inequalities_es(m, I, n, d, retailers, ordering_periods, T):
    m.add_constraints((I[0,t] -I[0,t + n -1]  >=  m.sum(d[c,k] for c in retailers for k in range(t+1, t + n)) for t in ordering_periods[:-1]))
    t = ordering_periods[-1]
    m.add_constraint((I[0,t] - I[0,T] >=  m.sum(d[c,k] for c in retailers for k in range(t+1,T+1))))

def valid_inequalities_ses(m, X, ordering_periods, d, T, retailers, n)  :  
    m.add_constraints((m.sum(X[0,j,k] for j in range(1, t+1) for k in range(t+1, T+1)) - m.sum(X[0,j,k] for j in range(1, t+n) for k in range(t+n, T+1)) >= m.sum(d[c,k] for c in retailers for k in range(t+1, t + n)) for t in ordering_periods[:-1]))
    t = ordering_periods[-1]
    m.add_constraint((m.sum(X[0,j,k]  for j in range(1, t+1) for k in range(t+1, T+1)) >= m.sum(d[c,k] for c in retailers for k in range(t+1, T+1))))
       
def valid_inequalities_tp_tpc(m, W, retailers, d, ordering_periods, T, n):
    m.add_constraints((m.sum(m.sum(W[c,q,j,k] for j in range(q, k+1)) for k in range(t+1, T+1) for q in range(1, t+1) for c in retailers) 
                            - m.sum(m.sum(W[c,q,j,k] for j in range(q, k+1)) for k in range(t+n, T+1) for q in range(1, t+n) for c in retailers) >=  
                               m.sum(d[c,k] for c in retailers for k in range(t+1, t + n)) for t in ordering_periods[:-1]))
    t = ordering_periods[-1]
    m.add_constraint((m.sum(m.sum(W[c,q,j,k] for j in range(q, k+1)) for k in range(t+1, T+1) for q in range(1, t+1) for c in retailers) >=  m.sum(d[c,k] for c in retailers for k in range(t+1, T+1))))
        
def valid_inequalities_sp_spc(m, U, d, retailers, ordering_periods, n, T):
    m.add_constraints((m.sum((m.sum(U[c,q,j,k]* (m.sum(d[c,l] for l in range(t+1, k+1))) for j in range(q, t+1)) + m.sum(U[c,q,j,k]* (m.sum(d[c,l] for l in range(j, k+1))) for j in range(t+1, k+1))) for k in range(t+1, T+1)  for c in retailers for q in range(1,t+1)) 
                               - m.sum((m.sum(U[c,q,j,k]* (m.sum(d[c,l] for l in range(t+n, k+1))) for j in range(q, t+n)) + m.sum(U[c,q,j,k]* (m.sum(d[c,l] for l in range(j, k+1))) for j in range(t+n, k+1))) for k in range(t+n, T+1)  for c in retailers for q in range(1,t+n))
                               >= m.sum(d[c,k] for c in retailers for k in range(t+1, t + n)) for t in ordering_periods[:-1]))
    t =  ordering_periods[-1]
    m.add_constraint((m.sum((m.sum(U[c,q,j,k]* (m.sum(d[c,l] for l in range(t+1, k+1))) for j in range(q, t+1)) + m.sum(U[c,q,j,k]* (m.sum(d[c,l] for l in range(j, k+1))) for j in range(t+1, k+1))) for k in range(t+1, T+1)  for c in retailers for q in range(1,t+1)) 
                               >= m.sum(d[c,k] for c in retailers for k in range(t+1, T + 1))))

def valid_inequalities_mc(m, sigma_0, sigma_1, retailers, ordering_periods, T, n, d) :
    m.add_constraints((m.sum(sigma_0[c,t,j] + sigma_1[c,t,j] for c in retailers for j in range(t+1, T+1)) - m.sum(sigma_0[c,t+n-1,j] + sigma_1[c,t+n-1,j] for c in retailers for j in range(t+n, T+1)) >= m.sum(d[c,k] for c in retailers for k in range(t+1, t + n)) for t in ordering_periods[:-1]))
    t = ordering_periods[-1]
    m.add_constraint(( m.sum(sigma_0[c,t,j] + sigma_1[c,t,j] for c in retailers for j in range(t+1, T+1)) >= m.sum(d[c,k] for c in retailers for k in range(t+1, T+1)) ))

# takes variable y and returns the list of indices t where the warehouse orders         
def return_ordering_periods(y):
    """
    y variable
    """
    res = []
    i = 1
    while (0,i) in y.keys(): # get T value
        i += 1
    T = i -1
    for t in range(1,T+1):
        if y[0,t].solution_value > 0.5 :
            res.append(t)
    return(res)
        
# these functions give the echelon stock at period t
def stock_owmr(m, periods, s,retailers):
    print([round((s[0,t] + m.sum(s[c,t] for c in retailers)).solution_value) for t in periods])

def stock_es(m,periods, I):
    print([round(I[0,t].solution_value) for t in periods])

def stock_ses(m, periods, X, T):
    print([round((m.sum(X[0,j,k] for j in range(1, t+1) for k in range(t+1, T+1))).solution_value) for t in periods])

def stock_mc(m, periods, sigma_0, sigma_1, retailers, T):
    print([round((m.sum(sigma_0[c,t,j] + sigma_1[c,t,j] for c in retailers for j in range(t+1, T+1))).solution_value) for t in periods])

def stock_tp(m, periods, W, retailers, T):
    print([round((((m.sum(m.sum(W[c,q,j,k] for j in range(q, k+1)) for k in range(t+1, T+1) for q in range(1, t+1) for c in retailers)))).solution_value) for t in periods])

def stock_sp(m, periods, U, d, retailers,T):
    print([(m.sum( m.sum(U[c,q,j,k]* (m.sum(d[c,l] for l in range(t+1, k+1))) for j in range(q, t+1)) + m.sum(U[c,q,j,k]* (m.sum(d[c,l] for l in range(j, k+1))) for j in range(t+1, k+1))for k in range(t+1, T+1)  for c in retailers for q in range(1,t+1))).solution_value  for t in periods])