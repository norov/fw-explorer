from math import sqrt
from scipy.stats import norm
import numpy as np

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


def calculate_returns(given_params, calibrated_params):
    nr = given_params["num_runs"]
    sim_L = given_params["periods"]

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
            ] * np.random.randn(1)
            Dc[t] = calibrated_params["chi"] * (P[t] - P[t - 1]) + calibrated_params[
                "sigma_c"
            ] * np.random.randn(1)

            # pricing
            P[t + 1] = P[t] + given_params["mu"] * (Nf[t] * Df[t] + Nc[t, r] * Dc[t])
        log_r[:, :, r] = (P[1 : sim_L + 1] - P[0:sim_L]).T
    # returns
    return log_r, Nc


def generate_constraint(given_params, calibrated_params, run_type="WP"):

    set_switching(run_type, calibrated_params)
    log_r, Nc = calculate_returns(given_params, calibrated_params)

    # log -> simple returns
    simple_R = np.exp(log_r) - 1.0

    # H to be determined
    output = {"H": None,
              "exog_signal": simple_R,
              "Nc": Nc,
              
             }

    return output

