import time
import importlib
import glob

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

from dashboard_integration import calculate_returns, init_parameters

globdata = {}
loaddata = {}

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
        Input("Load_trigger", "data"),
    ],
)
def random_seed(n_clicks, Load_trigger):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'Load_trigger.data' and bool(Load_trigger):
        return [ loaddata['seed_val' ]]

    if n_clicks is None:
        raise dash.exceptions.PreventUpdate()

    return [ np.random.randint(65536) ]


@app.callback(
    [
        Output("model-select",     "options"),
        Output("model-select",     "value"),
    ],
    [
        Input("cal_params","data"),
    ],
)
@timeit
def populate_params(fw_params):
    fw_params = pd.read_json(fw_params, orient='split')
    options = [
            {'label': row['Name'], 'value': index}
            for index, row in fw_params.iterrows()
            ]

    #print(options[0])
    return [ options, options[0]['value'] ]

def param_to_swipe(params, model_num, param):
    if param == 'phi':
        start = params.iloc[model_num][11]
        stop = params.iloc[model_num][12]

    elif param == 'chi':
        start = params.iloc[model_num][13]
        stop = params.iloc[model_num][14]

    elif param == 'eta':
        start = params.iloc[model_num][15]
        stop = params.iloc[model_num][16]

    elif param == 'alpha_w':
        start = params.iloc[model_num][17]
        stop = params.iloc[model_num][18]

    elif param == 'alpha_O':
        start = params.iloc[model_num][19]
        stop = params.iloc[model_num][20]

    elif param == 'alpha_n':
        start = params.iloc[model_num][21]
        stop = params.iloc[model_num][22]

    elif param == 'alpha_p':
        start = params.iloc[model_num][23]
        stop = params.iloc[model_num][24]

    elif param == 'sigma_f':
        start = params.iloc[model_num][25]
        stop = params.iloc[model_num][26]

    elif param == 'sigma_c':
        start = params.iloc[model_num][27]
        stop = params.iloc[model_num][28]
    else:
        print('Unknown parameter:', param)

    step = (stop - start) / 10
    return start, step, stop

@app.callback(
    [
        Output("swipe_start",  "value"),
        Output("swipe_step",  "value"),
        Output("swipe_stop",  "value"),
    ],
    [
        Input("model-select", "value"),
        Input("swipe-select", "value"),
        Input("cal_params", "data"),
        Input("Load_trigger", "data"),
    ],
)
@timeit
def set_swipes(model_num, swipe_select, fw_params, Load_trigger,):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'Load_trigger.data':
        return [
                loaddata["swipe_start"],
                loaddata["swipe_step"],
                loaddata["swipe_stop"],
                ]

    if swipe_select is None or fw_params is None or model_num is None:
        raise dash.exceptions.PreventUpdate()

    fw_params = pd.read_json(fw_params, orient='split')
    start, step, stop = param_to_swipe(fw_params, model_num, swipe_select)

    return start, step, stop


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
        Input("Load_trigger","data"),
    ],
    [
        State("cal_params","data"),
    ],
)
@timeit
def set_params(model_num, Load_trigger, fw_params):
    ctx = dash.callback_context

    if ctx.triggered[0]['prop_id'] == 'Load_trigger.data':
        return [
                loaddata["model_type"],
                loaddata["Phi"],
                loaddata["Chi"],
                loaddata["Eta"],
                loaddata["alpha_w"],
                loaddata["alpha_o"],
                loaddata["alpha_n"],
                loaddata["alpha_p"],
                loaddata["sigma_f"],
                loaddata["sigma_c"],
                ]

    if fw_params is None:
        raise dash.exceptions.PreventUpdate()

    fw_params = pd.read_json(fw_params, orient='split')
    vals = fw_params.iloc[model_num]
    if vals.empty:
        raise dash.exceptions.PreventUpdate()

    ret = [vals.values[1]]
    for v in vals.values[2:]:
        ret.append(np.float64(v))

    return ret[:10]

@app.callback(Output('rvmean', 'disabled'),
              [Input('rvmean_cb', 'value')],
              )
def enable_revmean(cb):
    return not cb

@app.callback(Output('retmean', 'disabled'),
              [Input('retmean_cb', 'value')],
              )
def enable_revmean(cb):
    return not cb

@app.callback(Output('cal_params', 'data'),
              [Input('upload', 'contents')],
              )
@timeit
def update_output(contents):
    if contents is None:
        fw_params = pd.read_csv('params.csv')
        return fw_params.to_json(date_format='iso', orient='split')


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
        Output("Swipe_data",  "data"),
    ],
    [
        Input('btn_swipe', 'n_clicks'),
        Input('Load_trigger', 'data'),
    ],
    [
        State("swipe-type", "value"),
        State("seed_val", "value"),

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
        State("retmean", "value"),
        State("retmean", "disabled"),

        # Swipe parameters
        State("swipe-select",  "value"),
        State("swipe_start",  "value"),
        State("swipe_stop",  "value"),
        State("swipe_step",  "value"),
        State("start_period",  "value"),
        State("stop_period",  "value"),

        State("Swipe_data",  "data"),
        State("hold",   "value"),
        State("start_params", "data"),
        State("pick_checkbox", "value"),
    ]
)
@timeit
def do_swipe(n_clicks, load_trigger,
             swipe_type,
             seed_val,
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
             retmean,
             retmean_disabled,
             swipe_select, swipe_start, swipe_stop, swipe_step,
             returns_start, returns_stop,
             swipe_data,
             hold,
             start_params,
             pick_checkbox,
             ):
    ctx = dash.callback_context

    if ctx.triggered[0]['prop_id'] == 'Load_trigger.data':
        if not 'sens' in loaddata or not 'swipe_data' in loaddata:
            raise dash.exceptions.PreventUpdate()
        return [loaddata['sens'], loaddata['swipe_data']]


    if n_clicks is None:
        raise dash.exceptions.PreventUpdate()

    swipe_range = np.append(np.arange(swipe_start, swipe_stop, swipe_step), swipe_stop)
    params = make_params(seed_val, ml, ss, periods, paths, prob_type,
                    Phi, Chi, Eta, alpha_w, alpha_o, alpha_n,
                    alpha_p, sigma_f, sigma_c,
		    rvmean, rvmean_disabled,
		    retmean, retmean_disabled,
		    fillna = True)

    sw_mean = []
    sw_vol = []
    sw_skew = []
    sw_kurt = []
    sw_chartists_mean = []
    distrib_returns = []
    distrib_chartists = []
    qqplots = []
    
    dist_list = [0, len(swipe_range)//2, len(swipe_range)-1]

    nvals = periods * (returns_stop - returns_start)
    
    for i, param in enumerate(swipe_range):
        params[swipe_select] = param

        if pick_checkbox == []:
            start_params = None

        out = generate_constraint(params, start_params)
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
        
    swipe = {
                'swipe_type'        : swipe_type,
                'param_range'       : swipe_range,
                'param_mean'        : sw_mean,
                'param_vol'         : sw_vol,
                'param_skew'        : sw_skew,
                'param_kurt'        : sw_kurt,
                'chartists_mean'    : sw_chartists_mean,
                'distrib_ret'       : distrib_returns,
                'distrib_chartists' : distrib_chartists,
                'qqplots_graph'     : qqplots,
            }

    if swipe_data is None or hold is None or not hold:
        swipe_data = [swipe]
    else:
        swipe_data.append(swipe)

    fig = plot_changes_params(swipe_data)

    return [
            dcc.Graph( id="param_sens", style={ 'height': 800 }, figure = fig),
            swipe_data,
           ]


@app.callback(
    Output("model-type",   "options"),
    [
        Input('cal_params', 'data')
    ],
)
@timeit
def set_options(fw_params):
    return [
            {'label': 'DCA', 'value': 'DCA'},
            {'label': 'TPA', 'value': 'TPA'},
            {'label': 'Shu-Zhu', 'value': 'Shu-Zhu'},
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
        Output("click_data", "data"),
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

    return sel_curves, old_sel_curves, clickData

@app.callback(
    [
        Output("start_params", "data"),
    ],
    [
        Input("pick_start", "n_clicks"),
    ],
    [
        State("click_data", "data"),
        State("simulated_data", "data"),
        State("visible_random", "data"),
        State("visible_topvol", "data"),
        State("visible_lessvol", "data"),
        State("visible_maxdd", "data"),
    ],
)
def pick_start_point(n_clicks, click_data, data,
                    rnd, tv, lv, mdd):
    nplots = 4
    if n_clicks == [] or click_data is None:
        raise dash.exceptions.PreventUpdate()

    line_n = click_data['points'][0]['curveNumber'] // nplots
    x = click_data['points'][0]['x']
    ln = rnd[line_n]
    if x < 2:
        raise dash.exceptions.PreventUpdate()

    start_params = {
           'P'  : globdata['P'].T[ln,  x-2 : x+1],
           'A'  : globdata['A'].T[ln,  x-2 : x+1],
           'Nc' : globdata['Nc'].T[ln, x-2 : x+1],
           'Nf' : globdata['Nf'].T[ln, x-2 : x+1],
           'Df' : globdata['Df'].T[ln, x-2 : x+1],
           'Dc' : globdata['Dc'].T[ln, x-2 : x+1],
           'Wf' : globdata['Wf'].T[ln, x-2 : x+1],
           'Wc' : globdata['Wc'].T[ln, x-2 : x+1],
           'pstar' : globdata['pstar'].T[ln, x-2 : x+1],
           'cstar' : globdata['cstar'].T[ln, x-2 : x+1],
           }
    return [ start_params ]

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
        Input('Load_trigger', 'data'),
    ]
)
@timeit
def update_sel_curves(sel_curves, ret, Load_trigger):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'Load_trigger.data':
        if 'dv' not in loaddata:
            raise dash.exceptions.PreventUpdate()

        return [loaddata['dv']]

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
        Input("Load_trigger", "data"),
    ]
)
@timeit
def update_graph(data, rnd, topvol, lessvol, maxdd, Load_trigger):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'Load_trigger.data':
        if 'main' not in loaddata:
            raise dash.exceptions.PreventUpdate()

        return [loaddata['main']]

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

def make_params(seed, ml, ss, periods, paths, prob_type,
                Phi, Chi, Eta, alpha_w, alpha_o,
                alpha_n, alpha_p, sigma_f, sigma_c,
                rvmean, rvmean_disabled,
		retmean, retmean_disabled, fillna = True
                ):
    params = {
            "seed_val": seed,
            "phi": Phi,  ##AK: demand senstivity of the fundamental agents to price deviations.
            "chi": Chi,  ##AK: demand senstivity of the chartest agents to price deviations.
            "eta": Eta,  ##AK: performance memory (backward looking ewma)
            "alpha_w": alpha_w,  ## AK: importance of backward looking performance
            "alpha_O": alpha_o,  ## a basic predisposition toward the fundmental strategy
            "alpha_p": alpha_p,  ## misalignment; version to a fundamental strategy when price
                                 ## becomes too far from fundamental
            "sigma_f": sigma_f,  ## noise in the fundamental agent demand
            "sigma_c": sigma_c,  ## noise in the chartest agent demand

            "mu": ml,
            "beta": ss,
            "num_runs": paths,
            "periods": periods,
            "prob_type": prob_type,
            }

    if fillna:
        for param in params:
            val = params[param]
            if val is None:
                params[param] = 0

    params["rvmean"] = None if rvmean_disabled else rvmean
    params["retmean"] = None if retmean_disabled else retmean

    return params
    

@app.callback(
    [
        Output("simulated_data", "data"),
    ],
    [
        Input("btn-simulate", "n_clicks")
    ],
    [
        State("seed_val", "value"),
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
        State("retmean", "value"),
        State("retmean", "disabled"),
        State("start_params", "data"),
        State("pick_checkbox", "value"),
    ],
)
@timeit
def update_simulated_data(
    n_clicks,
    seed_val,
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
    retmean,
    retmean_disabled,
    start_params,
    pick_checkbox,
):
    global globdata

    if n_clicks == 0 or n_clicks is None:
        raise dash.exceptions.PreventUpdate()


    if prob_type == 'Shu-Zhu':
        g, c =  init_parameters()
        ret = calculate_returns(g, c)
        globdata = ret.copy()
        return [ True ]

    params = make_params(seed_val, ml, ss, periods, paths, prob_type,
                    Phi, Chi, Eta, alpha_w, alpha_o,
                    alpha_n, alpha_p, sigma_f, sigma_c,
                    rvmean, rvmean_disabled,
		    retmean, retmean_disabled,
                    )
    if pick_checkbox == []:
        start_params = None

    ret =  generate_constraint(params, start_params)
    globdata = ret.copy()

    return [ True ]

@app.callback(
    [
        Output("load_filename", "options"),
    ],
    [
        Input("btn_save", "n_clicks")
    ],
    [
        State("filename", "value"),
        State("seed_val", "value"),
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
        State("rvmean_cb", "value"),
        State("rvmean", "disabled"),
        State("retmean", "value"),
        State("retmean_cb", "value"),
        State("retmean", "disabled"),
        State("start_params", "data"),
        State("pick_checkbox", "value"),

        State("simulated_data","data"),
        State("sens",   "children"),
        State("Swipe_data",   "data"),
        State("div-graphs",   "children"),
        State("dv",   "children"),

        State("swipe-select",  "value"),
        State("swipe-type",  "value"),
        State("swipe_start",  "value"),
        State("swipe_step",  "value"),
        State("swipe_stop",  "value"),
        State("start_period",  "value"),
        State("stop_period",  "value"),
        
        State("comments_txt",  "value"),
        
    ],
)
def btn_save(
    n_clicks,

    filename,
    seed_val,
    ml,
    ss,
    periods,
    paths,
    model_type,
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
    rvmean_cb,
    rvmean_disabled,
    retmean,
    retmean_cb,
    retmean_disabled,
    start_params,
    pick_checkbox,

    sim_data,
    sens,
    swipe_data,
    main,
    dv,

    swipe_select,
    swipe_type,
    swipe_start,
    swipe_step,
    swipe_stop,
    start_period,
    stop_period,
    
    comments_txt,
):
    global globdata
    args = locals().copy()

    if not (n_clicks == 0 or n_clicks is None or filename is None):
        args['globdata'] = globdata
        path = 'save/' + filename + '.pickle'
        with open(path, 'wb') as fd:
            pickle.dump(args, fd)

    fnames = [os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob('save/*.pickle')]

    options = [ {'label': fname, 'value': fname} for fname in sorted(fnames) ]

    return [options]


@app.callback(
    [
        Output("Load_trigger", "data"),
        Output("slider-ml", "value"),
        Output("slider-ss", "value"),
        Output("periods", "value"),
        Output("paths", "value"),
#        Output("model-type", "value"),
#        Output("Phi",     "value"),
#        Output("Chi",     "value"),
#        Output("Eta",     "value"),
#        Output("alpha_w", "value"),
#        Output("alpha_o", "value"),
#        Output("alpha_n", "value"),
#        Output("alpha_p", "value"),
#        Output("sigma_f", "value"),
#        Output("sigma_c", "value"),
#        Output("rvmean", "value"),
        Output("rvmean_cb", "value"),
#        Output("rvmean", "disabled"),
#        Output("retmean", "value"),
        Output("retmean_cb", "value"),
#        Output("retmean", "disabled"),
#        Output("start_params", "data"),
        Output("pick_checkbox", "value"),

#        Output("simulated_data","data"),
#        Output("sens",   "children"),
#        Output("Swipe_data","data"),
#        Output("div-graphs",   "children"),
        Output("swipe-select",  "value"),
        Output("swipe-type",  "value"),
        Output("start_period",  "value"),
        Output("stop_period",  "value"),
        
        Output("comments_txt",  "value"),
    ],
    [
        Input("btn_load", "n_clicks")
    ],
    [
        State("load_filename", "value"),
    ],
)
def btn_load(n_clicks, filename,):
    global globdata
    global loaddata

    ctx = dash.callback_context
    print(ctx.triggered[0]['prop_id'])

    if n_clicks == 0 or n_clicks is None:
        raise dash.exceptions.PreventUpdate()
    if filename is None:
        raise dash.exceptions.PreventUpdate()

    path = 'save/' + filename + '.pickle'
    with open(path, 'rb') as fd:
        args = pickle.load(fd)

    loaddata = args.copy()
    globdata = loaddata['globdata'].copy()
    #print(args)

    return [
            [0],
             args["ml"],
             args["ss"],
             args["periods"],
             args["paths"],
#             args["model-type"],
#             args["Phi"],
#             args["Chi"],
#             args["Eta"],
#             args["alpha_w"],
#             args["alpha_o"],
#             args["alpha_n"],
#             args["alpha_p"],
#             args["sigma_f"],
#             args["sigma_c"],
#             args["rvmean"],
             args["rvmean_cb"],
#             args["rvmean_disabled"],
#             args["retmean"],
             args["retmean_cb"],
#             args["retmean_disabled"],
#             args["start_params"],
             args["pick_checkbox"],
# 
#             args["simulated_data"],
#             args["sens"],
#             args["swipe_data"],
#             args["div-graphs"],
            args["swipe_select"],
            args["swipe_type"],
            args["start_period"],
            args["stop_period"],
             
             args["comments_txt"],

            ]


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
