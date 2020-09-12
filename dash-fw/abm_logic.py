from math import sqrt
from scipy.stats import norm, kurtosis, skew
import numpy as np
from plotly.subplots import make_subplots
import os
import pickle
import time


rand = np.random.rand(2, 1, 1)

def mean_price(price, pstar, period, rvmean):
    if period >= rvmean:
        return np.mean(price[period - rvmean : period, :], axis = 0)

    M = np.mean(price[0 : period, :], axis = 0) * period \
            + pstar[period - 1, :] * (rvmean - period)

    return M / rvmean


def mean_ret(price, period, retmean):
    if period >= retmean:
        return (price[period, :] - price[period - retmean, :]) / retmean

    return (price[period, :] - price[0, :])  / retmean


def calculate_returns(params, start_params):
    np.random.seed(params['seed_val'])

    rvmean = params["rvmean"]
    retmean = params["retmean"]
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

    rand = norm.rvs(loc = 0, scale = 1, size=(2, nr, sim_L))

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
    cstar = np.zeros([sim_L, nr])
    model_vol = np.zeros([sim_L, nr])

    if start_params is not None:
        P[0:3, :] = np.matrix(start_params['P']).T
        A[0:3, :] = np.matrix(start_params['A']).T
        Nc[0:3, :] = np.matrix(start_params['Nc']).T
        Nf[0:3, :] = np.matrix(start_params['Nf']).T
        Dc[0:3, :] = np.matrix(start_params['Dc']).T
        Df[0:3, :] = np.matrix(start_params['Df']).T
        Wc[0:3, :] = np.matrix(start_params['Wc']).T
        Wf[0:3, :] = np.matrix(start_params['Wf']).T
        pstar[0:3, :] = np.matrix(start_params['pstar']).T
        cstar[0:3, :] = np.matrix(start_params['cstar']).T

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
        Dc[t, :] = chi * (cstar[t-1, :] + \
	                  P[t, :] - P[t - 1, :])   + sigma_c * rand[1, : nr, t]

        # pricing
        P[t + 1, :] = P[t, :]\
                + mu * (Nf[t, :] * Df[t, :] + Nc[t, :] * Dc[t, :])

        if rvmean is not  None:
            pstar[t, :] = mean_price(P, pstar, t, rvmean)

        if retmean is not  None:
            cstar[t, :] = mean_ret(P, t, retmean)

        model_vol[t, :] = mu * (Nf[t, :] * sigma_f + Nc[t, :] * sigma_c)

    log_r = P[1 : sim_L + 1, :] - P[0:sim_L, :]
    return log_r, Nc, model_vol, P, Nc, Nf, A, Dc, Df, Wc, Wf, pstar, cstar


def generate_constraint(params, start_params = None):
    t_start = time.time()
    log_r, Nc, model_vol, P, Nc, Nf,\
        A, Dc, Df, Wc, Wf, pstar, cstar = calculate_returns(params, start_params)

    # log -> simple returns
    simple_R = np.exp(log_r) - 1.0

    # H to be determined
    output = {"H": None,
              "exog_signal": simple_R,
              "prices"   : np.exp(P),
              "model_vol": model_vol,
	      'P'        : P,
	      'A'        : A,
	      'Nc'       : Nc,
	      'Nf'       : Nf,
	      'Df'       : Df,
	      'Dc'       : Dc,
	      'Wf'       : Wf,
	      'Wc'       : Wc,
	      'pstar'    : pstar,
	      'cstar'    : cstar,
             }
    return output

def model_stat(swipe_type, model_out, returns_start, returns_stop):
    nvals = model_out['prices'].shape[1] * (returns_stop - returns_start)

    if swipe_type == 'Return':
        data = model_out['exog_signal'][returns_start : returns_stop, :].ravel()
    else:
        data = model_out['prices'][returns_start : returns_stop, :].ravel()

    p = np.mean(data)
    v = np.std(data)
    plo = p - v / nvals**0.5
    phi = p + v / nvals**0.5
    return [p, plo, phi],\
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
