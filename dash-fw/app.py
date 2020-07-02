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

globdata = {}

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        # print('%r () %2.2f sec' % \
        #      (method.__name__, te-ts))

        # print('%r (%r, %r) %2.2f sec' % \
        #       (method.__name__, args, kw, te-ts))
        return result

    return timed

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server


@timeit
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
    Output("model-select",     "options"),
    [
        Input("intermediate-value","children"),
    ],
)
@timeit
def populate_params(fw_params):
    fw_params = pd.read_json(fw_params, orient='split')
    options = [
            {'label': row['Name'], 'value': index}
            for index, row in fw_params.iterrows()
            ]

    return options

@app.callback(
    [
        Output("card_swipe", "hidden"),

        Output("Phi_start",  "value"),
        Output("Phi_stop",  "value"),
        Output("Phi_step",  "value"),

        Output("Chi_start",  "value"),
        Output("Chi_stop",  "value"),
        Output("Chi_step",  "value"),
    ],
    [
        Input("tabs", "value"),
    ],
    [
        State('Phi', 'value'),
        State('Chi', 'value')
    ]
)
@timeit
def show_swipes(tab, phi, chi):
    if phi is None:
        phi_start = None
        phi_stop = None
        phi_step= None
    else:
        phi_start = phi / 2
        phi_stop = phi * 1.5
        phi_step = phi / 10

    if chi is None:
        chi_start = None
        chi_stop = None
        chi_step= None
    else:
        chi_start = chi / 2
        chi_stop = chi * 1.5
        chi_step = chi / 10
    return [
            tab != 'tab_sensitivity',
            phi_start, phi_stop, phi_step,
            chi_start, chi_stop, chi_step,
            ]


@app.callback(
    [
        Output("model-type",  "value"),
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
        Input("model-select","value"),
    ],
    [
        State("intermediate-value","children"),
    ],
)
@timeit
def set_params(model_num, fw_params):
    if fw_params is None:
        raise dash.exceptions.PreventUpdate()
    fw_params = pd.read_json(fw_params, orient='split')
    vals = fw_params.iloc[model_num]
    if vals.empty:
        raise dash.exceptions.PreventUpdate()

    ret = [vals.values[1]]
    for v in vals.values[2:]:
        ret.append(np.float64(v))

    return ret

@app.callback(Output('rvmean', 'disabled'),
              [Input('rvmean_cb', 'value')],
              )
def enable_revmean(cb):
    return not cb

@app.callback(Output('intermediate-value', 'children'),
              [Input('upload', 'contents')],
              )
@timeit
def update_output(contents):
    if contents is None:
        fw_params = pd.read_csv('data.csv')
        return fw_params.to_json(date_format='iso', orient='split')

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(contents)
    ret = pd.read_csv(io.StringIO(str(contents)))

    return dict(ret)


@app.callback(
    [
        Output("btn-edit",   "style"),
        Output("btn-delete", "style"),
    ],
    [
        Input('model-select', 'value')
    ],
)
@timeit
def set_visible(value):
    if value is None:
        disp = 'none'
    else:
        disp = 'flex'

    style = {'display': disp, 'width': '95%'}
    return [style, style]

@app.callback(
    [
        Output("sens",   "children"),
    ],
    [
        Input('btn_swipe', 'n_clicks')
    ],
    [
        # Model parameters
        State("slider-ml", "value"),
        State("slider-ss", "value"),
        State("periods", "value"),
        State("paths", "value"),
        State("model-type", "value"),
        State("Phi",     "value"),
        State("Chi",     "value"),
        State("Eta",     "value"),
        State("alpha_w", "value"),
        State("alpha_o", "value"),
        State("alpha_n", "value"),
        State("alpha_p", "value"),
        State("sigma_f", "value"),
        State("sigma_c", "value"),
        State("rvmean", "value"),
        State("rvmean", "disabled"),

        # Swipe parameters
        State("Phi_swipe",  "value"),
        State("Phi_start",  "value"),
        State("Phi_stop",  "value"),
        State("Phi_step",  "value"),
    ]
)
@timeit
def do_swipe(n_clicks,
             ml,
             ss,
             periods,
             paths,
             prob_type,
             Phi,
             Chi,
             Eta,
             alpha_w,
             alpha_o,
             alpha_n,
             alpha_p,
             sigma_f,
             sigma_c,
             rvmean,
             rvmean_disabled,
             phi_swipe, phi_start, phi_stop, phi_step):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate()

    phi_range = np.arange(phi_start, phi_stop, phi_step)
    gparams, cparams = make_params(ml, ss, periods, paths, prob_type,
                    Phi, Chi, Eta, alpha_w, alpha_o, alpha_n,
                    alpha_p, sigma_f, sigma_c, rvmean, rvmean_disabled,)

    phi_mean = []
    phi_vol = []
    for phi in phi_range:
        cparams['phi'] = phi
        out = generate_constraint(gparams, cparams)
        p, v = model_stat(out)
        phi_mean.append(p)
        phi_vol.append(v)

    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Price over Phi","Return Vol over Phi"))

    fig.add_trace(go.Scattergl(x=phi_range, y=phi_mean, mode='lines'),
                    row=1, col=1)

    fig.add_trace(go.Scattergl(x=phi_range, y=phi_vol, mode='lines'),
                    row=1, col=2)

    return [ dcc.Graph( id="phi_sens",
            style={
                'height': 300
                },
            figure = fig)
           ]


@app.callback(
    Output("model-type",   "options"),
    [
        Input('intermediate-value', 'children')
    ],
)
@timeit
def set_options(fw_params):
    return [
            {'label': 'DCA', 'value': 'DCA'},
            {'label': 'TPA', 'value': 'TPA'}
           ]


@app.callback(
        Output('card1', 'hidden'),
        [
            Input('btn-edit', 'n_clicks'),
        ],
        [
            State('card1', 'hidden'),
        ]
    )
def card1_hide(n_clicks, hidden):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate()

    return not hidden


@app.callback(
    [
        Output("visible_topvol", "data"),
    ],
    [
        Input("mostvol", "n_clicks"),
    ],
    [
        State("mostvol_val", "value"),
        State("simulated_data", "data"),
    ]
)
@timeit
def set_topvol(n_clicks, npath, data):
    ctx = dash.callback_context
    if n_clicks is None\
            or npath is None\
            or data is None:
        raise dash.exceptions.PreventUpdate()

    if n_clicks % 2 == 0:
        return [None]

    paths = np.array(globdata['exog_signal'])
    vol = np.std(paths, axis = 0)
    top = np.argsort(vol)[-npath:]

    return [sorted(top)]


@app.callback(
    [
        Output("visible_lessvol", "data"),
    ],
    [
        Input("lessvol", "n_clicks"),
    ],
    [
        State("lessvol_val", "value"),
        State("simulated_data", "data"),
    ]
)
@timeit
def set_lessvol(n_clicks, npath, data):
    ctx = dash.callback_context
    if n_clicks is None\
            or npath is None\
            or data is None:
        raise dash.exceptions.PreventUpdate()

    if n_clicks % 2 == 0:
        return [None]

    paths = np.array(globdata['exog_signal'])
    vol = np.std(paths, axis = 0)
    lv = np.argsort(vol)[0:npath]

    return [sorted(lv)]


@app.callback(
    [
        Output("visible_maxdd", "data"),
    ],
    [
        Input("maxdd", "n_clicks"),
    ],
    [
        State("maxdd_val", "value"),
        State("simulated_data", "data"),
    ]
)
@timeit
def set_maxdd(n_clicks, npath, data):
    ctx = dash.callback_context
    if n_clicks is None\
            or npath is None\
            or data is None:
        raise dash.exceptions.PreventUpdate()

    if n_clicks % 2 == 0:
        return [None]

    paths = np.array(globdata['prices'])
    mv = np.min(paths, axis = 0)
    idx = np.argsort(mv)[0:npath]

    return [sorted(idx)]


@app.callback(
    [
        Output("visible_random", "data"),
    ],
    [
        Input("btn_rand", "n_clicks"),
    ],
    [
        State("rand_val", "value"),
        State("simulated_data", "data"),
    ]
)
@timeit
def set_rand(n_clicks, npath, data):
    ctx = dash.callback_context
    if n_clicks is None\
            or npath is None\
            or data is None:
        raise dash.exceptions.PreventUpdate()

    if n_clicks % 2 == 0:
        return [None]

    paths = np.array(globdata['exog_signal'])
    idx = np.random.choice(range(paths.shape[1]), npath, replace=False)
    return [sorted(idx)]


@app.callback(
    [
        Output("selected_curves", "children"),
        Output("old_selected_curves", "children"),
    ],
    [
        Input("graph_all_curves", "clickData"),
    ],
    [
        State("selected_curves", "children"),
        State("simulated_data", "data"), # XXX REMOVE?
    ]
)
@timeit
def select_trace(clickData, sel_curves, data):
    ctx = dash.callback_context
    if data is None:
        raise dash.exceptions.PreventUpdate()

    nplots = 4
    # No click handler
    if clickData is None:
        raise dash.exceptions.PreventUpdate()

    # Get curve
    line_n = clickData['points'][0]['curveNumber'] // nplots

    old_sel_curves = sel_curves[:]

    # update color
    # Currently 4 graphs in subplot
    # Future improvements should allow for more or user based decision
    if line_n not in sel_curves:
        sel_curves.append(line_n)
    else:
        sel_curves.remove(line_n)

    return sel_curves, old_sel_curves


def highlight_trace(figure, trace, yes):
    nplots = 4
    if yes:
        for i in range(nplots):
            figure['data'][trace * nplots + i]['line']['width'] = 1.4
            figure['data'][trace * nplots + i]['line']['color'] = 'orange'
    else:
        for i in range(4):
            figure['data'][trace * nplots + i]['line']['width'] = 0.7
            figure['data'][trace * nplots + i]['line']['color'] = 'rgba(255,255,255,0.3)'


@app.callback(
    [
        Output("graph_all_curves", "figure"),
    ],
    [
        Input("selected_curves", "children"),
    ],
    [
        State("old_selected_curves", "children"),
        State("graph_all_curves", "figure"),
    ],
)
@timeit
def update_trace(sel_curves, old_sel_curves, figure):
    # update color
    # Currently 4 graphs in subplot
    # Future improvements should allow for more or user based decision
    for trace in sel_curves:
        if trace not in old_sel_curves:
            highlight_trace(figure, trace, True)

    for trace in old_sel_curves:
        if trace not in sel_curves:
            highlight_trace(figure, trace, False)

    return [figure]


@app.callback(
    Output("dv", "children"),
    [
        Input('selected_curves', 'children'),
        Input('simulated_data', 'data'),
    ]
)
@timeit
def update_sel_curves(sel_curves, ret):
    if sel_curves == [] or ret is None:
        raise dash.exceptions.PreventUpdate()

    paths = np.array(globdata['exog_signal'])
    nc = np.array(globdata['Nc'])
    scurves = {
            'exog_signal': paths[:,sel_curves],
            'Nc': nc[:,sel_curves],
            }

    #fig = generate_graph_prod(scurves)
    fig = distrib_plots(scurves, sel_curves)

    return dcc.Graph(
            id="graph_sel_curves",
            figure=fig,
            )

@app.callback(
    [
        Output("div-graphs", "children"),
    ],
    [
        Input("simulated_data", "data"),
        Input("visible_random", "data"),
        Input("visible_topvol", "data"),
        Input("visible_lessvol", "data"),
        Input("visible_maxdd", "data"),
    ]
)
@timeit
def update_graph(data, rnd, topvol, lessvol, maxdd):
    if data is None:
        raise dash.exceptions.PreventUpdate()

    fig = generate_graph_prod(globdata, rnd, topvol, lessvol, maxdd)
    return [
            html.Div(
                id="svm-graph-container",
                children=dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="graph_all_curves", figure=fig),
                    style={"display": "block"},
                    ),
                ),
            ]

def make_params(ml, ss, periods, paths, prob_type,
                Phi, Chi, Eta, alpha_w, alpha_o,
                alpha_n, alpha_p, sigma_f, sigma_c,
                rvmean, rvmean_disabled,
                ):
    gparams = {
            "mu": ml,
            "beta": ss,
            "num_runs": paths,
            "periods": periods,
            "rvmean": None if rvmean_disabled else rvmean,
            "prob_type": prob_type,
            }

    cparams = {
            "phi": Phi,  ##AK: demand senstivity of the fundamental agents to price deviations.
            "chi": Chi,  ##AK: demand senstivity of the chartest agents to price deviations.
            "eta": Eta,  ##AK: performance memory (backward looking ewma)
            "alpha_w": alpha_w,  ## AK: importance of backward looking performance
            "alpha_O": alpha_o,  ## a basic predisposition toward the fundmental strategy
            "alpha_p": alpha_p,  ## misalignment; version to a fundamental strategy when price
                                 ## becomes too far from fundamental
            "sigma_f": sigma_f,  ## noise in the fundamental agent demand
            "sigma_c": sigma_c,  ## noise in the chartest agent demand
            }

    for param in cparams:
        val = cparams[param]
        if val is None:
            cparams[param] = 0

    return  gparams, cparams
    

@app.callback(
    [
        Output("simulated_data", "data"),
    ],
    [
        Input("btn-simulate", "n_clicks")
    ],
    [
        State("slider-ml", "value"),
        State("slider-ss", "value"),
        State("periods", "value"),
        State("paths", "value"),
        State("model-type", "value"),
        State("Phi",     "value"),
        State("Chi",     "value"),
        State("Eta",     "value"),
        State("alpha_w", "value"),
        State("alpha_o", "value"),
        State("alpha_n", "value"),
        State("alpha_p", "value"),
        State("sigma_f", "value"),
        State("sigma_c", "value"),
        State("rvmean", "value"),
        State("rvmean", "disabled"),
    ],
)
@timeit
def update_simulated_data(
    n_clicks,
    ml,
    ss,
    periods,
    paths,
    prob_type,
    Phi,
    Chi,
    Eta,
    alpha_w,
    alpha_o,
    alpha_n,
    alpha_p,
    sigma_f,
    sigma_c,
    rvmean,
    rvmean_disabled,
):
    global globdata

    if n_clicks == 0 or n_clicks is None:
        raise dash.exceptions.PreventUpdate()

    gparams, cparams = make_params(ml, ss, periods, paths, prob_type,
                    Phi, Chi, Eta, alpha_w, alpha_o,
                    alpha_n, alpha_p, sigma_f, sigma_c,
                    rvmean, rvmean_disabled,
                    )

    ret =  generate_constraint(gparams, cparams)
    globdata = ret.copy()

    return [ True ]


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
