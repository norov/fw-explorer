import time
import importlib

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State

import utils.figures as figs

from interface import *
from abm_logic import *

import pandas as pd

fw_params = pd.read_csv('data.csv')

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server


def generate_data(n_samples, dataset, noise):
    if dataset == "moons":
        return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=0)

    elif dataset == "circles":
        return datasets.make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=1
        )

    elif dataset == "linear":
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1,
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            "Data type incorrectly specified. Please choose an existing dataset."
        )


app.layout = html.Div(
    children=[
        div_header("Franke-Westerhoff Explorer",
               "https://github.com/talagent",
               app.get_asset_url("logo_talagent_glyph.svg"),
               "https://talagentfinancial.com"),
        div_panel(),
    ]
)


@app.callback(
    [
        Output("Phi",	  "value"),
        Output("Chi",	  "value"),
        Output("Eta",	  "value"),
        Output("alpha_w", "value"),
        Output("alpha_o", "value"),
        Output("alpha_n", "value"),
        Output("alpha_p", "value"),
        Output("sigma_f", "value"),
        Output("sigma_c", "value"),
    ],
    [
        Input("model",     "value"),
        Input("model-type","value"),
    ],
)
def set_params(model, typ):
    print(typ, model)
    vals = fw_params[fw_params['Type']==typ][fw_params['Model'] == model].values
    print(vals)
    return list(vals[0,2:])


@app.callback(
    Output("div-graphs", "children"),
    [
        Input("btn-simulate", "n_clicks")
    ],
    [
        State("slider-ml", "value"),
        State("slider-ss", "value"),
        State("periods", "value"),
        State("paths", "value"),
        State("dcc_dd_sim_type", "value"),
    ],
)
def update_graph(
    n_clicks,
    ml,
    ss,
    periods,
    paths,
    sim_type
):
    t_start = time.time()

    gparams = {"mu": ml, "beta": ss, "num_runs": paths, "periods": periods}
    
    ret = generate_constraint(given_params = gparams, run_type="WP")
    return [
        html.Div(
            id="svm-graph-container",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="graph-sklearn-svm", figure=None),
                style={"display": "none"},
            ),
        ),
        html.Div(
            id="graphs-container",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="graph-line-roc-curve", figure=None),
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-pie-confusion-matrix", figure=None
                    ),
                ),
            ],
        ),
    ]


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
