from math import sqrt
from scipy.stats import norm
import numpy as np
from abm_logic import *

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

        html.Div(
            [
            html.Div(
                [
                    dcc.Graph(id="return_dist", style = {'width': '40vh'}),
                ], className = "six columns"),
            html.Div(
                [
                    dcc.Graph(id="num_chartists", style = {'width': '40vh'}),
                ], className = "six columns"),
            ], className = "row"),
        dcc.Dropdown(
            id="test_model",
            options=[{"label": i, "value": i} for i in models],
            value="WP",
        ),
    ]
)

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


@app.callback(
    [
        Output("return_dist", "figure"),
        Output("num_chartists", "figure"),
    ],
    [
        Input("test_model", "value"),
        Input("update", "n_clicks")],
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

    ret = generate_constraint(given_params, calibrated_params, run_type=test_model)
    exog = ret["exog_signal"]
    Nc = ret["Nc"]
    print(exog.shape)
    print(Nc.shape)

    return paths_fig(exog, test_model), chartists_fig(Nc)

def paths_fig(exog, test_model):
    sim_length = exog.shape[1]
    data = np.cumprod(exog[0, :sim_length, :] + 1, 0)
    m = np.mean(data, axis=1)
    p = np.quantile(data, [0.05, 0.95], axis=1).T
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
    #ax_p.set_facecolor('white')
    plotly_fig = mpl_to_plotly(fig_p)

    return plotly_fig

def chartists_fig(Nc):
    m = np.mean(Nc, axis=1)
    fig_p = plt.figure()
    ax_p = fig_p.add_subplot(111)
    ax_p.plot(Nc, "k", alpha = 0.05)
    ax_p.plot(m, "k", alpha = 0.4)
    # ax_p.plot(range(sim_L), np.exp(simple_R[:, :, 1:]))
    plt.xlabel("Time")
    plt.ylabel("Chartists")
    ax_p.set_title("Chartists share")
    plotly_fig = mpl_to_plotly(fig_p)

    return plotly_fig


if __name__ == "__main__":
    app.run_server(debug=True, port=8050, host="localhost")
