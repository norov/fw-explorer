from math import sqrt
from scipy.stats import norm
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


def update_randomness(nr, sim_L):
    rand = get_randomness()
    nagents, rand_nr, rand_sim_L = rand.shape
    if rand_nr < nr:
        r = norm.rvs(loc = 0, scale = 1, size=(2, nr - rand_nr, rand_sim_L))
        rand = np.append(rand, r, axis = 1)
    elif rand_sim_L < sim_L:
            r = norm.rvs(loc = 0, scale = 1, size=(2, rand_nr, sim_L - rand_sim_L))
            rand = np.append(rand, r, axis = 2)
    else:
        return rand

    with open('rnd.pickle', 'wb') as f:
        pickle.dump(rand, f)

    return rand


def mean_price(price, period, rvmean):
    if period >= rvmean:
        return np.mean(price[period - rvmean : period, :])

    # Assuming starting price is 0
    return np.mean(price[0 : period, :]) * period / rvmean

def calculate_returns(given_params, calibrated_params):
    print(given_params)
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

    rand = get_randomness()

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
    # for rrrr in range(nr):  ##AK: why not start at time t=1
    #     ## YN: to make all start from a single point?
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
        Dc[t, :] = chi * (P[t, :] - P[t - 1, :]) +   sigma_c * rand[1, : nr, t]

        # pricing
        P[t + 1, :] = P[t, :]\
                + mu * (Nf[t, :] * Df[t, :] + Nc[t, :] * Dc[t, :])

        if rvmean is not  None:
            pstar[t, :] = mean_price(P, t, rvmean)

    log_r = P[1 : sim_L + 1, :] - P[0:sim_L, :]
    return log_r, Nc


def generate_constraint(given_params, calibrated_params):
    t_start = time.time()
    log_r, Nc = calculate_returns(given_params, calibrated_params)

    # log -> simple returns
    simple_R = np.exp(log_r) - 1.0

    # H to be determined
    output = {"H": None,
              "exog_signal": simple_R,
              "Nc": Nc,
              
             }
    print(time.time() - t_start)
    return output

