from math import sqrt
from scipy.stats import norm
import numpy as np
from plotly.subplots import make_subplots
import os
import pickle
import time


def set_switching(runType, calibrated_params):
    if runType == "W":  # wealth only
        calibrated_params["alpha_w"] = 1580
        calibrated_params["alpha_O"] = 0
        calibrated_params["alpha_p"] = 0
    elif runType == "WP":  # wealth and predisposition
        calibrated_params["alpha_w"] = 2668
        calibrated_params["alpha_O"] = 2.1
        calibrated_params["alpha_p"] = 0
    elif runType == "WM":  # wealth misalignment
        # this case is not in the FW set: just testing the
        calibrated_params["alpha_w"] = 1580
        calibrated_params["alpha_O"] = 0
        calibrated_params["alpha_p"] = 500
    elif runType == "CN":  # common noise
        mean_sig = (calibrated_params["sigma_f"] + calibrated_params["sigma_c"]) / 2
        calibrated_params["sigma_f"] = mean_sig
        calibrated_params["sigma_c"] = mean_sig

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


def calculate_returns(given_params, calibrated_params):
    print(given_params)
    nr = given_params["num_runs"]
    sim_L = given_params["periods"]

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

    Nf = np.zeros([sim_L, 1])  ##progostic state
    Df = np.zeros([sim_L, 1])  ##diagnostic state
    Dc = np.zeros([sim_L, 1])  ##diagnostic state
    Gf = np.zeros([sim_L, 1])  ##diagnostic state
    Gc = np.zeros([sim_L, 1])  ##diagnostic state
    Wf = np.zeros([sim_L, 1])  ##diagnostic state
    Wc = np.zeros([sim_L, 1])  ##diagnostic state
    A = np.zeros([sim_L, 1])  ##diagnostic state

    # initial values
    Nf[0:2] = 0.5
    ##AK: what else should be i nitialized if we were doing forecasting?

    # add for loop to create ndarray of many runs of exog signal
    Nc = np.zeros([sim_L, nr])  #Number of chartists
    Nf = np.zeros([sim_L, nr])  ##progostic state
    P = np.zeros([sim_L + 1, nr])  ##progostic state
    pstar = 0
    for r in range(nr):  ##AK: why not start at time t=1
        ## YN: to make all start from a single point?
        for t in range(2, sim_L):  # generation of single signal over time
            # portfolio performance
            Gf[t] = (np.exp(P[t, r]) - np.exp(P[t - 1, r])) * Df[t - 2]
            Gc[t] = (np.exp(P[t, r]) - np.exp(P[t - 1, r])) * Dc[t - 2]

            # summarize performance over time
            Wf[t] = (
                calibrated_params["eta"] * Wf[t - 1]
                + (1 - calibrated_params["eta"]) * Gf[t]
            )
            Wc[t] = (
                calibrated_params["eta"] * Wc[t - 1]
                + (1 - calibrated_params["eta"]) * Gc[t]
            )

            # type fractions
            Nf[t, r] = 1 / (1 + np.exp(-given_params["beta"] * A[t - 1]))
            Nc[t, r] = 1 - Nf[t, r]

            # The A[t] dynamic is set up to handle several models
            A[t] = (
                calibrated_params["alpha_w"] * (Wf[t] - Wc[t])
                + calibrated_params["alpha_O"]
                + calibrated_params["alpha_p"] * (pstar - P[t, r]) ** 2
            )

            # demands
            Df[t] = calibrated_params["phi"] * (pstar - P[t, r]) + calibrated_params[
                "sigma_f"
            ] * rand[0, r, t]
            Dc[t] = calibrated_params["chi"] * (P[t, r] - P[t - 1, r]) + calibrated_params[
                "sigma_c"
            ] * rand[1, r, t]

            # pricing
            P[t + 1, r] = P[t, r] + given_params["mu"] * (Nf[t, r] * Df[t] + Nc[t, r] * Dc[t])
    log_r = P[1 : sim_L + 1, :] - P[0:sim_L, :]
    return log_r, Nc


def generate_constraint(given_params, calibrated_params, run_type="WP"):
    t_start = time.time()
    set_switching(run_type, calibrated_params)
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

