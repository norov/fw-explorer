from math import sqrt
from scipy.stats import norm, kurtosis, skew
import numpy as np
from plotly.subplots import make_subplots
import os
import pickle
import time


def set_randomness(x, y):
    randomness = norm.rvs(loc = 0, scale = 1, size=(2, x, y))
    with open('rnd.pickle', 'wb') as f:
        pickle.dump(randomness,f)


def get_randomness():
    with open('rnd.pickle', 'rb') as f:
        randomness = pickle.load(f)
    return randomness

rand = get_randomness()

def update_randomness(nr, sim_L):
    rand = get_randomness()
    nagents, rand_nr, rand_sim_L = rand.shape
    if rand_nr < nr:
        r = norm.rvs(loc = 0, scale = 1, size=(2, nr - rand_nr, rand_sim_L))
        rand = np.append(rand, r, axis = 1)

    if rand_sim_L < sim_L:
        r = norm.rvs(loc = 0, scale = 1, size=(2, rand_nr, sim_L - rand_sim_L))
        rand = np.append(rand, r, axis = 2)

    with open('rnd.pickle', 'wb') as f:
        pickle.dump(rand, f)

    return rand


def mean_price(price, period, rvmean):
    if period >= rvmean:
        return np.mean(price[period - rvmean : period, :])

    # Assuming starting price is 0
    return np.mean(price[0 : period, :]) * period / rvmean

def calculate_returns(given_params, calibrated_params):
    global rand
    rvmean = given_params["rvmean"]
    nr = given_params["num_runs"]
    sim_L = given_params["periods"]
    mu = given_params["mu"]
    typ = given_params["prob_type"]

    alpha_w = calibrated_params["alpha_w"]
    alpha_o = calibrated_params["alpha_O"]
    alpha_p = calibrated_params["alpha_p"]
    eta = calibrated_params["eta"]
    beta = given_params["beta"]
    phi = calibrated_params["phi"]
    chi = calibrated_params["chi"]

    sigma_f = calibrated_params["sigma_f"]
    sigma_c = calibrated_params["sigma_c"]

    if os.path.exists('rnd.pickle') == False :
        set_randomness(nr, sim_L)

    #rand = update_randomness(nr, sim_L)

    nagents, rand_nr, rand_sim_L = rand.shape
    if rand_nr < nr:
        r = norm.rvs(loc = 0, scale = 1, size=(2, nr - rand_nr, rand_sim_L))
        rand = np.append(rand, r, axis = 1)

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


def generate_constraint(given_params, calibrated_params):
    t_start = time.time()
    log_r, Nc, model_vol = calculate_returns(given_params, calibrated_params)

    # log -> simple returns
    simple_R = np.exp(log_r) - 1.0

    # H to be determined
    output = {"H": None,
              "exog_signal": simple_R,
              "prices": np.array(np.cumprod(simple_R + 1, 0)),
              "Nc": Nc,
              "model_vol": model_vol,
             }
    print(time.time() - t_start)
    return output

def model_stat(swipe_type, model_out, returns_start, returns_stop):
    if swipe_type == 'Return':
        data = model_out['exog_signal'][returns_start : returns_stop, :].ravel()
    else:
        data = model_out['prices'][-1,:]
    return np.mean(data),\
           np.std(data),\
           np.mean(model_out['Nc'].ravel()),\
           skew(data),\
           kurtosis(data, fisher=False)

def swipe_params(given_params, calibrated_params, swipe_params):
    price_mean = {}
    return_vol = {}
    for param in swipe_params:
        price_mean[param] = []
        return_vol[param] = []
        for p in swipe_params[param]:
            calibrated_params[param] = p
            out = generate_constraint(given_params, calibrated_params)
            p, v = model_stat(out)
            price_mean[param].append(p)
            return_vol[param].append(v)
            
    return price_mean, return_vol
