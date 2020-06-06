import time
import importlib

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State

import plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

import utils.figures as figs

from interface import *
from abm_logic import *
from abm_graphs import *

import pandas as pd

import os
import base64
import io

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
        Input("intermediate-value","children"),
    ],
)
def set_params(model, typ, fw_params):
    fw_params = pd.read_json(fw_params, orient='split')
    vals = fw_params[fw_params['Type']==typ][fw_params['Model'] == model]
    if vals.empty:
        raise dash.exceptions.PreventUpdate()

    return list(vals.values[0,2:])

@app.callback(Output('intermediate-value', 'children'),
              [Input('upload', 'contents')],
              )
def update_output(contents):
    if contents is None:
        fw_params = pd.read_csv('data.csv')
        return fw_params.to_json(date_format='iso', orient='split')

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(contents)
    ret = pd.read_csv(io.StringIO(str(contents)))
    print(ret)

    return dict(ret)


@app.callback(
    [
        Output("model",   "options"),
        Output("model-type",   "options"),
    ],
    [
        Input('intermediate-value', 'children')
    ],
)
def set_options(fw_params):
    if fw_params == None:
        raise dash.exceptions.PreventUpdate()

    fw_params = pd.read_json(fw_params, orient='split')
    print(fw_params)
    print(type(fw_params))

    type_options = [
            {'label': typ, 'value': typ}
            for typ in fw_params['Type'].unique()
            ]

    model_options = [
            {'label': typ, 'value': typ}
            for typ in fw_params['Model'].unique()
            ]

    return model_options, type_options


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
        State("model", "value")
    ],
)
def update_graph(
    n_clicks,
    ml,
    ss,
    periods,
    paths,
    sim_type,
):
    t_start = time.time()

    gparams = {"mu": ml, "beta": ss, "num_runs": paths, "periods": periods}
    
    ret = generate_constraint(given_params = gparams, run_type="WP")
    fig = generate_graph(ret)
    
    return [
        html.Div(
            id="svm-graph-container",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="graph-sklearn-svm", figure=fig),
                style={"display": "block"},
            ),
        ),
        # html.Div(
        #     id="graphs-container",
        #     children=[
        #         dcc.Loading(
        #             className="graph-wrapper",
        #             children=dcc.Graph(id="graph-line-roc-curve", figure=None),
        #         ),
        #         dcc.Loading(
        #             className="graph-wrapper",
        #             children=dcc.Graph(
        #                 id="graph-pie-confusion-matrix", figure=None
        #             ),
        #         ),
        #     ],
        # ),
    ]

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
