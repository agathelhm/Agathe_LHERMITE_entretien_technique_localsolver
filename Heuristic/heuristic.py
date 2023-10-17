import numpy as np
import cProfile
import time
import heapq

def load_parameters(filename : str):
    """
    filename under the form N15T30DD_DF01
    """
    # parameters 
    path = './data/'
    d = np.load(path + filename + '/demand.npy') # demand
    f = np.load(path + filename + '/set_up.npy') # set up costs
    h = np.load(path + filename + '/holding.npy') # holding costs

    T, N = d.shape[1]-1, d.shape[0]-1

    periods = list(range(1,T+1))
    facilities = list(range(N+1))
    retailers = list(range(1, N+1))

    return(d, f, h, T, N, periods, facilities, retailers)

def intersection_droites(a1, b1, a2, b2) :
    if abs(a1 - a2) > 1e-4 :
        return((b2-b1)/(a1-a2))
    else :
        return(None)

def recurrence_phi(phi : list, f : float, p : float, new_ind : int) :
    phi_j = []
    start, middle, end = True, False, False # vaut True si l'intersection ne s'est pas encore produite
    for i, (alpha, beta, gamma, ind) in enumerate(phi) : 
        if (f + p * alpha <= gamma) and (start): # premier croisement
            if alpha == 0 :
                phi_j.append((0, p, f, new_ind))
            else : # find equality point
                alpha_l, beta_l, gamma_l, ind_l = phi[i-1]
                new_alpha = intersection_droites(p, f, beta_l, gamma_l - beta_l * alpha_l)
                phi_j.append((new_alpha, p, f + p * new_alpha, new_ind))
            start = False
            middle = True
        elif (f + p * alpha >= gamma) and middle: # deuxième croisement
            alpha_l, beta_l, gamma_l, ind_l = phi[i-1]
            new_alpha = intersection_droites(p, f, beta_l, gamma_l - beta_l * alpha_l)
            phi_j.append((new_alpha, beta_l, new_alpha * beta_l + gamma_l - alpha_l * beta_l, ind_l))
            phi_j.append((alpha, beta, gamma, ind))
            middle = False
            end = True
        elif (f + p * alpha >= gamma) and (start or end) : # début et fin
            phi_j.append((alpha, beta, gamma, ind))
    alpha, beta, gamma, ind = phi[-1] 
    if (f + p * alpha >= gamma) and (p < beta) : 
        new_alpha = intersection_droites(p, f, beta, gamma - beta * alpha)
        phi_j.append((new_alpha, p, f + p * new_alpha, new_ind))
    elif (f + p * alpha <= gamma) and (p > beta) : 
        new_alpha = intersection_droites(p, f, beta, gamma - beta * alpha)
        phi_j.append((new_alpha, beta, new_alpha * beta + gamma - alpha * beta, ind))

    return(phi_j)

def init_phi(f, p, T, ordering_periods):
    """"
    calculate all phi j"""
    phis = [[]] * (T+1) 
    phis[1] = [(0, p[0, 1], f[0, 1], 1)]
    for j in range(2, T+1) :
        if j in ordering_periods :
            phis[j] = recurrence_phi(phis[j-1], f[0,j], p[0,j], j)
        else :
            phis[j] = phis[j-1]
    return(phis)

def init_p(h, T, periods, N):
    p = np.zeros((N+1, T+1))
    for c in range(1, N+1) :
        for t in periods :
            p[c,t] = sum((h[c] - h[0]) for k in range(t, T+1))
    for t in periods :
        p[0,t] = sum(h[0] for k in range(t, T+1))
    return(p)

def init_D_uT(d, T, N, u):
    D = np.zeros((N+1, T+1,T+1))
    for c in range(1, N+1) :
        for i in range(u,T+1) :
            for k in range(i, T+1) :
                #D[c,i,k] = sum(d[c,r] for r in range(i, k + 1))
                D[c,i,k] = D[c, i, k-1] + d[c, k]
    return(D)

def init_H(T,  d, f, h, N, p) :
    H = np.zeros((N+1, T+1,T+1))
    FH = np.zeros((N+1, T+1,T+1))
    HH = np.zeros((N+1, T+1,T+1))
    for u in range(1, T+1) :
        D_u_T = init_D_uT(d, T, N, u)
        for c in range(1, N+1) :
            for i in range(u, T+1) :
                l = [H[c, u, k-1] + f[c,k] + p[c,k] * D_u_T[c, k,i] for k in range(u,i+1)]
                H[c, u, i] = min(l)
                k = l.index(H[c, u, i]) + u 
                HH[c, u, i] = HH[c, u, k-1] + p[c,k] * D_u_T[c, k,i]
                FH[c, u, i] = FH[c, u, k-1] + f[c,k] 
    return(H, FH, HH)

def merge(phi_j, D, j, T) :
    """
    for each t, res[t] such that alpha[t] <= D[j,t] <= alpha[t+1
    """
    res = [0] * (T+1)
    i = 0
    for t in range(j, T+1):
        while ((i < len(phi_j)) and (phi_j[i][0] <= D[j, t])) : 
            i += 1
        res[t] = i-1
        if i == len(phi_j) :
            res[t] = len(phi_j)-1
    return(res)

def phi_j_djt(phi_j, T, j, D):
    """
    phi = phis[j]
    returns khi[j] s. t. khi[j][i] 
    """
    res = merge(phi_j, D, j ,T)
    khi = [0] * (T+1)
    X_khi = [0] * (T+1)
    hkhi = [0] * (T+1)
    fkhi = [0] * (T+1)
    for t in range(j,T+1) : 
        alpha, beta, gamma, i = phi_j[res[t]]
        khi[t] = beta * D[j, t] + gamma - beta * alpha
        X_khi[t] = i
        hkhi[t] = beta * D[j, t] 
        fkhi[t] = gamma - beta * alpha
    return(khi, X_khi, hkhi, fkhi)
    
def init_khi(T : int, d, f, p, D, c, ordering_periods):
    phi = init_phi(f, p, T, ordering_periods)
    khi = [[]] * (T+1)
    X_khi = [[]] * (T+1)
    hkhi = [[]] * (T+1)
    fkhi = [[]] * (T+1)
    for j in range(1, T+1) : 
        khi[j], X_khi[j], hkhi[j], fkhi[j]  = phi_j_djt(phi[j], T, j, D)
    return(khi, X_khi, hkhi, fkhi)

def init_G_t(c : int, ordering_periods : list, T, d, periods, f, h, H, p, FH, HH):
    """
    filename N50T15DD_DF01
    c retailer number
    X list of authorized periods
    """

    D = np.zeros((T+1,T+1))
    for t in range(1, T+1) :
        for k in range(t, T+1) :
            #D[t,k] = sum(d[c,r] for r in range(t, k + 1))
            D[t,k] = D[t, k-1] + d[c,k]

    khi, X_khi, hkhi, fkhi = init_khi(T, d, f, p, D, c, ordering_periods)

    X = [[]] * (T+1)
    G = np.zeros(T+1)
    holding_costs = np.zeros(T+1)
    setup_costs = np.zeros(T+1)
    for t in periods :
        #js = list(set(range(1,t+1)).intersection(set(ordering_periods)))
        l = [G[j-1] + khi[j][t] + H[j,t] for j in range(1,t+1)]
        G[t] = min(l)
        j = l.index(G[t]) + 1
        X[t] = set(X[j-1]).union([X_khi[j][t]])
        holding_costs[t] = HH[j,t] + holding_costs[j-1] + hkhi[j][t]
        setup_costs[t] = FH[j,t] + setup_costs[j-1 ] + fkhi[j][t]

    return(G[T] - sum(h[c] * D[1,t] for t in periods), X[T], holding_costs[T] - sum(h[c] * D[1,t] for t in periods), setup_costs[T])

def simple_heuristic(filename : str, ordering_periods : list) :
    d, f, h, T, N, periods, facilities, retailers = load_parameters(filename) 
    start_time = time.time()

    obj = np.zeros(N+1)
    holding_costs = np.zeros(N+1)
    setup_costs = np.zeros(N+1)
    omega = [[]] * (N+1)
    p = init_p(h, T, periods, N)
    H, FH, HH = init_H(T,d, f, h, N, p)

    for c in retailers :
        obj[c], omega[c], holding_costs[c], setup_costs[c] = init_G_t(c, ordering_periods, T, d, periods, f, h, H[c], p,  FH[c], HH[c])

    number_of_orders = np.zeros(T+1)
    for c in retailers :
        for i in omega[c] :
            number_of_orders[i] += 1 
   
    
    res = sum(obj) - sum(f[0,t] * max(0,(number_of_orders[t]-1)) for t in periods)
    res_holding_costs = sum(holding_costs)
    res_setup_costs = sum(setup_costs) - sum(f[0,t] * max(0,(number_of_orders[t]-1)) for t in periods)
    total_time = time.time() - start_time
    #print(number_of_orders)
    return(res, total_time) #, res_holding_costs, res_setup_costs)


