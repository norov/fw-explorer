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

from scipy import stats

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
    [
        Output("seed_val", "value"),
    ],
    [
        Input("rnd_seed", "n_clicks"),
    ],
)
def random_seed(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate()

    return [ np.random.randint(65536) ]


@app.callback(
    [
        Output("set_seed", "disabled"),
    ],
    [
        Input("set_seed", "n_clicks"),
    ],
    [
        State("seed_val", "value"),
    ],
)
def set_seed(n_clicks, seed_val):
    print('set_seed', n_clicks)
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate()

    print('set_seed', seed_val)
    update_seed(seed_val)

    return [ False ]


@app.callback(
    [
        Output("stop_period", "value"),
    ],
    [
        Input("swipe-type", "value"),
    ],
    [
        State("periods", "value"),
    ]
)
@timeit
def show_return_params(swipe_type, paths):
    return [paths]

@app.callback(
    [
        Output("model-select",     "options"),
        Output("model-select",     "value"),
    ],
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

    print(options[0])
    return [ options, options[0]['value'] ]


@app.callback(
    [
        Output("swipe_start",  "value"),
        Output("swipe_stop",  "value"),
        Output("swipe_step",  "value"),
    ],
    [
        Input("swipe-select", "value"),
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
    ]
)
@timeit
def show_swipes(swipe_select,
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
    if swipe_select is None:
        raise dash.exceptions.PreventUpdate()

    gparams, cparams = make_params(ml, ss, periods, paths, prob_type,
                    Phi, Chi, Eta, alpha_w, alpha_o, alpha_n,
                    alpha_p, sigma_f, sigma_c, rvmean, rvmean_disabled,
                    fillna = False)
    print(swipe_select, cparams)
    param = cparams[swipe_select]

    if param is None:
        swipe_start = None
        swipe_stop = None
        swipe_step= None
    else:
       if swipe_select == 'alpha_o':
           a = np.abs(param)
           swipe_start = param - a / 2
           swipe_stop  = param + a * 1.5
           swipe_step  = (swipe_stop - swipe_start) / 10
       else:
           swipe_start = param / 2
           swipe_stop  = param * 1.5
           swipe_step  = param / 10
    return [
            swipe_start, swipe_stop, swipe_step,
            ]


@app.callback(
    [
        Output("card_swipe", "hidden"),
    ],
    [
        Input("tabs", "value"),
    ],
)
@timeit
def show_swipes(tab):
    return [ tab != 'tab_sensitivity', ]


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

    style = {'display': disp,
             'color': 'inherit',
             'width': '95%'}

    return [style, style]

@app.callback(
    [
        Output("sens",   "children"),
    ],
    [
        Input('btn_swipe', 'n_clicks')
    ],
    [
        State("swipe-type", "value"),

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
        State("swipe-select",  "value"),
        State("swipe_start",  "value"),
        State("swipe_stop",  "value"),
        State("swipe_step",  "value"),
        State("start_period",  "value"),
        State("stop_period",  "value"),
    ]
)
@timeit
def do_swipe(n_clicks,
             swipe_type,
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
             swipe_select, swipe_start, swipe_stop, swipe_step,
             returns_start, returns_stop):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate()

    swipe_range = np.append(np.arange(swipe_start, swipe_stop, swipe_step), swipe_stop)
    gparams, cparams = make_params(ml, ss, periods, paths, prob_type,
                    Phi, Chi, Eta, alpha_w, alpha_o, alpha_n,
                    alpha_p, sigma_f, sigma_c, rvmean, rvmean_disabled,)

    sw_mean = []
    sw_vol = []
    sw_skew = []
    sw_kurt = []
    sw_chartists_mean = []
    distrib_returns = []
    distrib_chartists = []
    qqplots = []
    
    dist_list = [0, len(swipe_range)//2,len(swipe_range)-1]
    
    for i, param in enumerate(swipe_range):
        cparams[swipe_select] = param
        out = generate_constraint(gparams, cparams)
        p, v, c, sk, kur = model_stat(swipe_type, out, returns_start, returns_stop)
        sw_mean.append(p)
        sw_vol.append(v)
        sw_skew.append(sk)
        sw_kurt.append(kur)
        sw_chartists_mean.append(c)
        
         #distribution
        if (i in dist_list):
            if swipe_type == 'Return':
                ret = out['exog_signal'][returns_start : returns_stop, :].ravel()
                qq = stats.probplot(ret, dist='norm', sparams=(1))
                qqplots.append(qq)
                
            else:
                #ret = out['prices'][-1, :]
                ret = out['prices'][returns_start : returns_stop, :].ravel()
                qq = stats.probplot(ret, dist='lognorm', sparams=(1))
                qqplots.append(qq)

            fig_dist = ff.create_distplot([ret],
                    group_labels=[str(param)])
            distrib_returns.append(fig_dist['data'][1])

            chartists = out['Nc'][-1, :]
            fig_dist = ff.create_distplot([chartists], group_labels=[str(param)])
            distrib_chartists.append(fig_dist['data'][1])
            
            
            
            #qq = qqplot(ret, line='s').gca().lines
            #qqplots.append(qq)
        
        
    
    fig = plot_changes_params(swipe_type = swipe_type,
                              param_range=swipe_range,
                              param_mean=sw_mean, 
                              param_vol=sw_vol,
                              param_skew=sw_skew,
                              param_kurt=sw_kurt,
                              chartists_mean=sw_chartists_mean,
                              distrib_ret=distrib_returns,
                              distrib_chartists = distrib_chartists,
                              qqplots_graph=qqplots)
    

    return [ dcc.Graph( id="param_sens",
            style={
                'height': 800
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
    prices = np.array(globdata['prices'])
    nc = np.array(globdata['Nc'])
    scurves = {
            'exog_signal': paths[:,sel_curves],
            'prices': prices[:,sel_curves],
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
                rvmean, rvmean_disabled, fillna = True
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

    if fillna:
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
