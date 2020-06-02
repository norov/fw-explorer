from math import sqrt
from scipy.stats import norm
import numpy as np

from plotly.tools import mpl_to_plotly
import matplotlib
import plotly.express as px

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output, State

models = ["W", "WP", "WM", "CN", "HPM"]


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
    Nc = np.zeros([sim_L, 1])  ##progostic state
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
    Nc[0:2] = 0.5
    ##AK: what else should be initialized if we were doing forecasting?

    # add for loop to create ndarray of many runs of exog signal
    log_r = np.zeros(shape=(1, sim_L, nr))
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
            Nc[t] = 1 - Nf[t]

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
            P[t + 1] = P[t] + given_params["mu"] * (Nf[t] * Df[t] + Nc[t] * Dc[t])
        log_r[:, :, r] = (P[1 : sim_L + 1] - P[0:sim_L]).T
    # returns
    return log_r


def plot_ret(simple_R):
    # plot price
    sim_length = simple_R.shape[1]
    fig_p, ax_p = plt.subplots()
    ax_p.plot(range(sim_length), np.cumprod(simple_R[0, :sim_length, :] + 1, 0))
    # fig_p, ax_p = plt.subplots()
    # ax_p.plot(range(sim_L), np.exp(simple_R[:, :, 1:]))
    plt.xlabel("Time")
    plt.ylabel("Price Ratio")
    ax_p.set_title("Discrete Choice Approach: Wealth")
    plt.show()

    # plot returns
    fig_r, ax_r = plt.subplots()
    ax_r.plot(range(sim_length), simple_R[0, :sim_length, :])
    plt.xlabel("Time")
    plt.ylabel("Returns")
    ax_r.set_title("Discrete Choice Approach:Wealth")  ##FIX to reflect actual choice.
    return fig


def generate_constraint(given_params, calibrated_params, run_type="WP"):

    set_switching(run_type, calibrated_params)
    log_r = calculate_returns(given_params, calibrated_params)

    # log -> simple returns
    simple_R = np.exp(log_r) - 1.0

    # H to be determined
    output = {"H": None, "exog_signal": simple_R}

    return output


def brownian(loc, scale, paths, n):
    rvs = norm.rvs(loc=0, scale=scale, size=(n, paths))
    bm = np.cumsum(rvs, axis=0)
    return np.vstack((np.zeros(paths), bm)) + loc


out = brownian(0, 1, 1000, 500)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.H1(
            children="""
        Talagent: Brownian Motion
    """
        ),
        html.Div(
            [
                html.Label("Market liquidity:"),
                dcc.Input(id="ml", type="number", value=0.01),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            [
                html.Label("Switching strength:"),
                dcc.Input(id="ss", type="number", value=1),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            [
                html.Label("Number of runs:"),
                dcc.Input(id="nr", type="number", value=200),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            [html.Label("Periods:"), dcc.Input(id="periods", type="number", value=500)],
            style={"display": "inline-block"},
        ),
        html.Button(id="update", n_clicks=0, children="Update"),
        dcc.Graph(id="matplotlib-graph"),
        dcc.Dropdown(
            id="test_model",
            options=[{"label": i, "value": i} for i in models],
            value="WP",
        ),
    ]
)


@app.callback(
    Output("matplotlib-graph", "figure"),
    [Input("test_model", "value"), Input("update", "n_clicks")],
    [
        State("ml", "value"),
        State("ss", "value"),
        State("nr", "value"),
        State("periods", "value"),
    ],
)
def update_figure(n_clicks, test_model, ml, ss, nr, periods):
    given_params = {"mu": ml, "beta": ss, "num_runs": nr, "periods": periods}

    # TODO: Take from file
    calibrated_params = {
        "phi": 1.00,  ##AK: demand senstivity of the fundamental agents to price deviations.
        "chi": 1.20,  ##AK: demand senstivity of the chartest agents to price deviations.
        "eta": 0.991,  ##AK: performance memory (backward looking ewma)
        "alpha_w": 1580,  ## AK: importance of backward looking performance
        "alpha_O": 0,  ##a basic predisposition toward the fundmental strategy
        "alpha_p": 0,  ##misalignment; version to a fundamental strategy when price becomes too far from fundamental
        "sigma_f": 0.681,  ##noise in the fundamental agent demand
        "sigma_c": 1.724,  ##noise in the chartest agent demand
    }

    ret = generate_constraint(given_params, calibrated_params, run_type=test_model)[
        "exog_signal"
    ]
    sim_length = ret.shape[1]

    data = np.cumprod(ret[0, :sim_length, :] + 1, 0)
    m = np.mean(data, axis=1)
    p = np.quantile(data, [0.05, 0.95], axis=1).T
    print(p.shape)
    print(m.shape)
    # plot price
    fig_p = plt.figure()
    ax_p = fig_p.add_subplot(111)
    ax_p.plot(data, "k", alpha=0.05)
    ax_p.plot(m, "k", alpha=0.4)
    ax_p.plot(p, "k", alpha=0.3)
    # ax_p.plot(range(sim_L), np.exp(simple_R[:, :, 1:]))
    plt.xlabel("Time")
    plt.ylabel("Price Ratio")
    ax_p.set_title(
        "Discrete Choice Approach: %s" % str(test_model)
    )  # todo this is not necessarily wealth
    plotly_fig = mpl_to_plotly(fig_p)

    return plotly_fig


if __name__ == "__main__":
    app.run_server(debug=True, port=8050, host="localhost")
