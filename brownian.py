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
                html.Label("Start:"),
                dcc.Input(id="start", type="number", placeholder="Start", value=0),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            [
                html.Label("Volatility:"),
                dcc.Input(id="sigma", type="number", placeholder="Sigma", value=1),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            [
                html.Label("Periods:"),
                dcc.Input(
                    id="periods", type="number", placeholder="Periods", value=500
                ),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            [
                html.Label("Simulations:"),
                dcc.Input(id="paths", type="number", placeholder="Paths", value=1000),
            ],
            style={"display": "inline-block"},
        ),
        html.Button(id="update", n_clicks=0, children="Update"),
        dcc.Graph(id="matplotlib-graph"),
        dcc.Slider(
            id="slider",
            min=1,
            max=out.shape[1],
            value=800,
            marks={str(n): str(n) for n in range(1, out.shape[1], 50)},
        ),
    ]
)


@app.callback(Output("slider", "max"), [Input("paths", "value")])
def on_path(paths):
    return paths


@app.callback(
    Output("matplotlib-graph", "figure"),
    [Input("slider", "value"), Input("update", "n_clicks")],
    [
        State("start", "value"),
        State("sigma", "value"),
        State("periods", "value"),
        State("paths", "value"),
    ],
)
def update_figure(slider, n_clicks, start, sigma, periods, paths):
    global out
    m = np.mean(out, axis=1)
    s = np.std(out, axis=1)
    trigger = dash.callback_context.triggered[0]["prop_id"]
    if trigger == "update.n_clicks":
        out = brownian(start, sigma, paths, periods)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(out[:, 0:slider], "k", alpha=0.02)
    # ax.grid(True)
    ax.plot(m, "k", alpha=0.5)
    ax.plot(m - s, "k-.", alpha=0.2)
    ax.plot(m + s, "k", alpha=0.2)
    ax.plot(m - 2 * s, "k", alpha=0.1)
    ax.plot(m + 2 * s, "k", alpha=0.1)

    plotly_fig = mpl_to_plotly(fig, resize=True)
    return plotly_fig


if __name__ == "__main__":
    app.run_server(debug=True, port=8050, host="localhost")
