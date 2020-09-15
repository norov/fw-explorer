"""
Integrate the model return simulation with the dashboard team
"""

import numpy as np

def sz_init_parameters():
    """
    Parameter initialization for the model
    Returns:
        given_params, calibrated_params in dict
    """
    given_params = {
        "mu": 0.01,  ## market liquidity (exogenous parameter)
        "mu1": 0.0003782,  ## constant drift for log price
        "pstar": 0,  ## the price which fundamentalist thinks the price should reverse to
        "noise_pct": 0.50,  ## percentage of noise agent in the market (fixed)
        "leverage_pct": 0.10,  ## percentage of leverage agent in the market (fixed)
        "chartist_lookback": 20,  ## momentum lookback
        "leverage_loc": 0.02,  ## loc in inverse_sigmoid
        "leverage_scale": 300,  ## scale in inverse_sigmoid
        'sigma_j': 3.2,  ## std of jump
        "jump_probability": 0.0,  ## probability of jump
    }
    calibrated_params = {
        "nu": 0.1,  ## transition coefficient
        "phi": 1.2662585762718377,  ## demand sensitivity of the fundamental agents to price deviations.
        "chi": 1.1704990853964858,  ## demand sensitivity of the chartist agents to price deviations.
        "chi1": 5.242566077501879,  ## demand sensitivity of the chartist agents to price deviations.
        "alpha_o": -1.4292499690966622,  ## a basic predisposition toward the fundamental strategy
        "alpha_p": 9.961520185901511,  # # misalignment; version to a fundamental strategy when price becomes too far
        # from fundamental
        "sigma_f": 0.20001759086829005,  ## noise in the fundamental agent demand
        "sigma_c": 5.94842972483428,  ## noise in the chartist agent demand
        "sigma_n": 0.012486955824132978,  ## noise in the noise agent demand
    }
    return given_params, calibrated_params


def inverse_sigmoid(x, loc=0.02, scale=300):
    return -1 / (1 + np.exp(scale * (x + loc)))


def noise_generator(p_j, sigma_j, sigma_n):
    return np.random.randn(1) * sigma_n + \
           (np.random.uniform(size=1) < p_j) * np.random.randn(1) * sigma_j


def sz_calculate_returns(given_params, calibrated_params, num_runs=100, sim_L=500, seed=None, burnout=100):
    """
    Calculation of model simulated return in Python
    """
    sim_L += burnout
    P = np.zeros([num_runs, sim_L + 1])  ##progostic state
    R = np.zeros([num_runs, sim_L + 1])  ##progostic state
    Nf = np.zeros([num_runs, sim_L])  ##progostic state
    Nc = np.zeros([num_runs, sim_L])  ##progostic state
    Wf = np.zeros([num_runs, sim_L])  ##diagnostic state
    Wc = np.zeros([num_runs, sim_L])  ##diagnostic state
    Df = np.zeros([num_runs, sim_L])  ##diagnostic state
    Dc = np.zeros([num_runs, sim_L])  ##diagnostic state
    Dl = np.zeros([num_runs, sim_L])  ##diagnostic state
    Dn = np.zeros([num_runs, sim_L])  ##diagnostic state
    A = np.zeros([num_runs, sim_L])  ##diagnostic state
    pstars = np.zeros([num_runs, sim_L])  ##diagnostic state
    cstars = np.zeros([num_runs, sim_L])  ##diagnostic state
    model_vol = np.zeros([num_runs, sim_L])

    # given param setting
    mu = given_params["mu"]
    mu1 = given_params["mu1"]
    pstar = given_params["pstar"]
    noise_pct = given_params["noise_pct"]
    leverage_pct = given_params["leverage_pct"]
    fc_pct = 1. - noise_pct - leverage_pct
    window = int(given_params["chartist_lookback"])
    sigma_j = given_params["sigma_j"]
    sigma_j /= noise_pct
    jump_probability = given_params["jump_probability"]
    lev_loc = given_params["leverage_loc"]
    lev_scale = given_params["leverage_scale"]

    # calibrated param setting
    nu = calibrated_params["nu"]
    phi = calibrated_params["phi"]
    chi = calibrated_params["chi"]
    chi1 = calibrated_params["chi1"]
    alpha_o = calibrated_params["alpha_o"]
    alpha_p = calibrated_params["alpha_p"]
    sigma_f = calibrated_params["sigma_f"]
    sigma_c = calibrated_params["sigma_c"]
    sigma_n = calibrated_params["sigma_n"]

    # initial values
    Nf[:, :2] = fc_pct / 2.
    Nc[:, :2] = fc_pct / 2.
    pstars[:, :2] = pstar

    if seed is not None:
        np.random.seed(seed)

    # add for loop to create ndarray of many runs of exog signal
    for r in range(num_runs):
        for t in range(2, sim_L):  # generation of single signal over time
            # type fractions
            Nft = (1 / (1 + np.exp(-A[r, t - 1]))) * fc_pct
            Nct = fc_pct - Nft
            Nf[r, t] = Nf[r, t - 1] + nu * (Nft - Nf[r, t - 1])
            Nc[r, t] = Nc[r, t - 1] + nu * (Nct - Nc[r, t - 1])

            # The A[t] dynamic is set up to handle several models
            A[r, t] = alpha_o + alpha_p * np.abs(pstars[r, t - 1] - P[r, t])

            # momentum
            if t < window + 2:
                mom = 0.0
            else:
                mom = np.mean(R[r, t - window:t]) / np.std(R[r, t - window:t])

            # demands
            Df[r, t] = phi * (pstars[r,t - 1] - P[r, t]) + sigma_f * np.random.randn(1)
            Dc[r, t] = chi * mom + sigma_c * np.random.randn(1)
            Dl[r, t] = chi1 * inverse_sigmoid(R[r, t], loc=lev_loc, scale=lev_scale)
            Dn[r, t] = noise_generator(p_j=jump_probability, sigma_j=sigma_j, sigma_n=sigma_n)

            # pricing
            R[r, t + 1] = mu * (Nf[r, t] * Df[r, t] +
                                Nc[r, t] * Dc[r, t] +
                                noise_pct * Dn[r, t] +
                                leverage_pct * Dl[r, t]
                                )
            P[r, t + 1] = P[r, t] + R[r, t + 1]
            pstars[r, t] = pstars[r, t - 1] + mu1
            model_vol[r, t] = mu * np.sqrt(
                (Nf[r, t] * sigma_f)**2 + (Nc[r, t] * sigma_c)**2 + (noise_pct * sigma_n)**2)
    # returns
    R = R[:, burnout + 1:].T
    P = P[:, burnout + 1:].T
    Nc = Nc[:, burnout:].T
    Nf = Nf[:, burnout:].T
    Wf = Wf[:, burnout:].T
    Wc = Wc[:, burnout:].T
    Df = Df[:, burnout:].T
    Dc = Dc[:, burnout:].T
    Dl = Dl[:, burnout:].T
    Dn = Dn[:, burnout:].T
    A = A[:, burnout:].T
    model_vol = model_vol[:, burnout:].T
    pstars = pstars[:, burnout:].T
    cstars = cstars[:, burnout:].T
    output = {"H": None,
              "exog_signal": R,
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
	      'pstar'    : pstars,
	      'cstar'    : cstars,
             }
    return output


if __name__ == '__main__':
    given_params, calibrated_params = init_parameters()
    results = calculate_returns(given_params, calibrated_params, num_runs=1, sim_L=5000, seed=None, burnout=100)
