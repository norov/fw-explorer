from math import sqrt
from scipy.stats import norm
import numpy as np
from plotly.subplots import make_subplots
import os
import pickle


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

    P = np.zeros([sim_L + 1, 1])  ##progostic state
    Nf = np.zeros([sim_L, 1])  ##progostic state
    pstar = 0
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
    log_r = np.zeros(shape=(1, sim_L, nr))
    Nc = np.zeros([sim_L, nr])  #Number of chartists
    for r in range(nr):  ##AK: why not start at time t=1
        ## YN: to make all start from a single point?
        for t in range(2, sim_L):  # generation of single signal over time
            # portfolio performance
            Gf[t] = (np.exp(P[t]) - np.exp(P[t - 1])) * Df[t - 2]
            Gc[t] = (np.exp(P[t]) - np.exp(P[t - 1])) * Dc[t - 2]

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
            Nf[t] = 1 / (1 + np.exp(-given_params["beta"] * A[t - 1]))
            Nc[t, r] = 1 - Nf[t]

            # The A[t] dynamic is set up to handle several models
            A[t] = (
                calibrated_params["alpha_w"] * (Wf[t] - Wc[t])
                + calibrated_params["alpha_O"]
                + calibrated_params["alpha_p"] * (pstar - P[t]) ** 2
            )

            # demands
            Df[t] = calibrated_params["phi"] * (pstar - P[t]) + calibrated_params[
                "sigma_f"
            ] * rand[0, r, t]
            Dc[t] = calibrated_params["chi"] * (P[t] - P[t - 1]) + calibrated_params[
                "sigma_c"
            ] * rand[1, r, t]

            # pricing
            P[t + 1] = P[t] + given_params["mu"] * (Nf[t] * Df[t] + Nc[t, r] * Dc[t])
        log_r[:, :, r] = (P[1 : sim_L + 1] - P[0:sim_L]).T
    # returns
    return log_r, Nc

gparams = {"mu": 0.01, "beta": 1, "num_runs": 200, "periods": 500}

cparams = {
    "phi": 1.00,  ##AK: demand senstivity of the fundamental agents to price deviations.
    "chi": 1.20,  ##AK: demand senstivity of the chartest agents to price deviations.
    "eta": 0.991,  ##AK: performance memory (backward looking ewma)
    "alpha_w": 1580,  ## AK: importance of backward looking performance
    "alpha_O": 0,  ##a basic predisposition toward the fundmental strategy
    "alpha_p": 0,  ##misalignment; version to a fundamental strategy when price becomes too far from fundamental
    "sigma_f": 0.681,  ##noise in the fundamental agent demand
    "sigma_c": 1.724,  ##noise in the chartest agent demand
}


def generate_constraint(given_params, calibrated_params, run_type="WP"):
    log_r, Nc = calculate_returns(given_params, calibrated_params)

    # log -> simple returns
    simple_R = np.exp(log_r) - 1.0

    # H to be determined
    output = {"H": None,
              "exog_signal": simple_R,
              "Nc": Nc,
              
             }

    return output

