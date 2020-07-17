from math import sqrt
from scipy.stats import norm, kurtosis, skew
import numpy as np
from plotly.subplots import make_subplots
import os
import pickle
import time


rand = np.random.rand(2, 1, 1)

def update_seed(seed):
    global rand
    print('WTF')
    print(rand.shape)
    np.random.seed(seed)
    rand = np.random.rand(2, 1, 1)
    print(rand)


def mean_price(price, period, rvmean):
    if period >= rvmean:
        return np.mean(price[period - rvmean : period, :])

    # Assuming starting price is 0
    return np.mean(price[0 : period, :]) * period / rvmean

def calculate_returns(params):
    global rand
    #print(rand.shape)
    rvmean = params["rvmean"]
    nr = params["num_runs"]
    sim_L = params["periods"]
    mu = params["mu"]
    typ = params["prob_type"]

    alpha_w = params["alpha_w"]
    alpha_o = params["alpha_O"]
    alpha_p = params["alpha_p"]
    eta = params["eta"]
    beta = params["beta"]
    phi = params["phi"]
    chi = params["chi"]

    sigma_f = params["sigma_f"]
    sigma_c = params["sigma_c"]

    nagents, rand_nr, rand_sim_L = rand.shape
    if rand_nr < nr:
        r = norm.rvs(loc = 0, scale = 1, size=(2, nr - rand_nr, rand_sim_L))
        rand = np.append(rand, r, axis = 1)

    nagents, rand_nr, rand_sim_L = rand.shape
    if rand_sim_L < sim_L:
        r = norm.rvs(loc = 0, scale = 1, size=(2, rand_nr, sim_L - rand_sim_L))
        rand = np.append(rand, r, axis = 2)

    Df = np.zeros([sim_L, nr])  ##diagnostic state
    Dc = np.zeros([sim_L, nr])  ##diagnostic state
    Gf = np.zeros([sim_L, nr])  ##diagnostic state
    Gc = np.zeros([sim_L, nr])  ##diagnostic state
    Wf = np.zeros([sim_L, nr])  ##diagnostic state
    Wc = np.zeros([sim_L, nr])  ##diagnostic state
    A = np.zeros([sim_L,  nr])  ##diagnostic state

    # add for loop to create ndarray of many runs of exog signal
    Nc = np.zeros([sim_L, nr])  #Number of chartists
    Nf = np.zeros([sim_L, nr])  ##progostic state
    P = np.zeros([sim_L + 1, nr])  ##progostic state
    pstar = np.zeros([sim_L, nr])
    model_vol = np.zeros([sim_L, nr])

    for t in range(2, sim_L):  # generation of single signal over time
        # portfolio performance
        Gf[t] = (np.exp(P[t, :]) - np.exp(P[t - 1, :])) * Df[t - 2]
        Gc[t] = (np.exp(P[t, :]) - np.exp(P[t - 1, :])) * Dc[t - 2]

        # summarize performance over time
        Wf[t] = eta * Wf[t - 1] + (1 - eta) * Gf[t]
        Wc[t] = eta * Wc[t - 1] + (1 - eta) * Gc[t]

        if typ ==  'TPA':
            v = 0.05 # XXX take from interface
            # Determine transition probabilities
            Pi_cf = np.minimum(np.ones([1, nr]), v * np.exp( A[t-1, :]));
            Pi_fc = np.minimum(np.ones([1, nr]), v * np.exp(-A[t-1, :]));

            Nf[t, :] = Nf[t-1, :] + Nc[t-1, :] * Pi_cf - Nf[t-1, :] * Pi_fc
            Nc[t, :] = 1 - Nf[t, :]
            A[t, :] = alpha_w * (Wf[t, :] - Wc[t, :])
        elif typ == 'DCA':
            Nf[t, :] = 1 / (1 + np.exp(-beta * A[t - 1, :]))
            Nc[t, :] = 1 - Nf[t, :]
            # The A[t] dynamic is set up to handle several models
            A[t, :] = alpha_w * (Wf[t, :] - Wc[t, :])\
                    + alpha_o + alpha_p * (pstar[t-1, :] - P[t, :]) ** 2
        else:
            raise ValueError('Type ont supported. Choose DCA or TPA.')


        # demands
        Df[t, :] = phi * (pstar[t-1, :] - P[t, :]) + sigma_f * rand[0, : nr, t]
        Dc[t, :] = chi * (P[t, :] - P[t - 1, :])   + sigma_c * rand[1, : nr, t]

        # pricing
        P[t + 1, :] = P[t, :]\
                + mu * (Nf[t, :] * Df[t, :] + Nc[t, :] * Dc[t, :])

        if rvmean is not  None:
            pstar[t, :] = mean_price(P, t, rvmean)

        model_vol[t, :] = mu * (Nf[t, :] * sigma_f + Nc[t, :] * sigma_c)

    log_r = P[1 : sim_L + 1, :] - P[0:sim_L, :]
    return log_r, Nc, model_vol


def generate_constraint(params):
    t_start = time.time()
    log_r, Nc, model_vol = calculate_returns(params)

    # log -> simple returns
    simple_R = np.exp(log_r) - 1.0

    # H to be determined
    output = {"H": None,
              "exog_signal": simple_R,
              "prices": np.array(np.cumprod(simple_R + 1, 0)),
              "Nc": Nc,
              "model_vol": model_vol,
             }
    #print(time.time() - t_start)
    return output

def model_stat(swipe_type, model_out, returns_start, returns_stop):
    if swipe_type == 'Return':
        data = model_out['exog_signal'][returns_start : returns_stop, :].ravel()
    else:
        data = model_out['prices'][returns_start : returns_stop, :].ravel()
    return np.mean(data),\
           np.std(data),\
           np.mean(model_out['Nc'].ravel()),\
           skew(data),\
           kurtosis(data, fisher=False)

def swipe_params(params, swipe_params):
    price_mean = {}
    return_vol = {}
    for param in swipe_params:
        price_mean[param] = []
        return_vol[param] = []
        for p in swipe_params[param]:
            params[param] = p
            out = generate_constraint(params, params)
            p, v = model_stat(out)
            price_mean[param].append(p)
            return_vol[param].append(v)
            
    return price_mean, return_vol
