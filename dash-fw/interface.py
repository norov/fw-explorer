import dash
import dash_core_components as dcc
import dash_html_components as html
import utils.dash_reusable_components as drc
import numpy as np
import pandas as pd
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
        else:
            raise Exception('Run once: ' + str(f))
    wrapper.has_run = False
    return wrapper


def div_header(title, link, banner, link2) :
    return html.Div(
        className="banner",
        children=[
            # Change App Name here
            html.Div(
                className="container scalable",
                children=[
                    # Change App Name here
                    html.H2(
                        id="banner-title",
                        children=[
                            html.A(
                                title,
                                href=link,
                                style={
                                    "text-decoration": "none",
                                    "color": "inherit",
                                    },
                                )
                            ],
                        ),
                    html.A(
                        id="banner-logo",
                        children=[ html.Img(src=banner) ],
                        href=link2,
                        ),
                    ],
                )
            ],
        )

def div_input_line(name, id,  value):
    return html.Div([
            drc.NamedInput(
                name=name[i],
                id=id[i],
                type="number",
                value=value[i]
                ) for i in range(len(name))
            ], style={
                "display": "flex",
                "width": "95%",
                },
            )

def card0():
    return drc.Card(
        id = 'select_card',
        children=[
            dcc.Dropdown(
                id="model-select",
                value = None,
                clearable=False,
                searchable=False,
                ),
            html.Div([
                html.Button(
                    "Show / Hide",
                    id="btn-edit",
                    style={
                    "display": "flex",
                    "width": "95%",
                    },
                    ),
                html.Button(
                    "Delete",
                    id="btn-delete",
                    style={
                    "display": "flex",
                    "width": "95%",
                    },
                    ), 
            ], style={
                "display": "flex",
                },
            ),
            ]
        )

def card1():
    return drc.Card(
        id="card1",
        hidden = True,
        children=[
            drc.NamedRadioItems(
                name='Model',
                id='model-type',
                labelStyle={
                    'margin-right': '7px',
                    'display': 'inline-block'
                    },
                value=None,
                ),
            div_input_line(
                ["Phi", "Chi", "Eta"],
                ["Phi", "Chi", "Eta"],
                [None, None, None],
                ),
            div_input_line(
                ["alpha_w", "alpha_o", "alpha_n"],
                ["alpha_w", "alpha_o", "alpha_n"],
                [None, None, None],
                ),
            div_input_line(
                ["alpha_p", "sigma_f", "sigma_c"],
                ["alpha_p", "sigma_f", "sigma_c"],
                [None, None, None],
                ),
            ]
        )


def card2():
    return drc.Card(
        id="button-card",
        children=[
            div_input_line(
                ["Periods", "Paths"],
                ["periods", "paths"],
                [500, 100],
                ),
            drc.NamedSlider(
                name="Market Liquidity",
                id="slider-ml",
                min=0,
                max=0.1,
                step=0.001,
                marks={
                    i/100: str(i/100)
                    for i in range(0, 11, 2)
                    },
                tooltip = {"always_visible": False,
                        "placement": "top"},
                value=0.01,
                ),
            drc.NamedSlider(
                name="Switching Strength",
                id="slider-ss",
                min=0,
                max=2,
                marks={
                    i / 10: str(i / 10)
                    for i in range(0, 20, 4)
                    },
                step=0.1,
                tooltip = {"always_visible": False,
                        "placement": "top"},
                value=1
                ),
            drc.CheckboxSlider("Reverse to mean", 'rvmean_cb', enabled = False,
                id = 'rvmean',
                min = 1,
                max = 365,
                marks={
                    i: str(i)
                    for i in range(0, 365, 90)
                    },
                tooltip = {
                    "always_visible": False,
                    "placement": "top"
                    },
                value = 180,
                ),
            html.Button(
                "Simulate",
                id="btn-simulate",
                ),
            ],
        )


def card3():
    return drc.Card(
        id="last-card",
        children=[
            html.Div([
                drc.ButtonInput('Random', 'btn_rand',
                   id =  'rand_val', value = 20),
                drc.ButtonInput('Most volatile', 'mostvol',
                   id =  'mostvol_val', value = 10),
            ], style={
            #    "display": "flex",
                },
            ),
            ],
        )


def graph_tabs():
    return html.Div([
                dcc.Tabs(id="tabs",
                         vertical=False,
                         value = 'One',
                         parent_style={'flex-direction': 'column',
                                       '-webkit-flex-direction': 'column',
                                       '-ms-flex-direction': 'column',
                                       'display': 'flex'},
                         colors={"border": "white",
                                 "primary": "white",
                                 "background": "grey"
                                 },
                         children=[
                        #      dcc.Tab(
                        #          label='Calibration 1',
                        #          style={'backgroundColor': "inherit"},
                        #          selected_style={'backgroundColor': "inherit",'color':'white'},
                        #          value = 'ABM1',
                        #          children = [
                        #              html.P("ABM1: calibration process here")
                        #              ]
                        #         ),
                             dcc.Tab(
                                 label='Calibration 2',
                                 style={'backgroundColor': "inherit"},
                                 selected_style={'backgroundColor': "inherit",'color':'white'},
                                 value = 'ABM1',
                                 children = [
                                     html.Div(
                                         id = 'abm2_params',
                                         children = [
                                         dcc.Dropdown(
                                            options = [
                                                    {'label': 'S&p 500', 'value': '^GSPC'},
                                                    {'label': 'Dow Jones', 'value': '^DJ'},
                                                    {'label': 'Russian Rubble', 'value': 'USDRUB'},
                                                    {'label': 'Louis d\'or', 'value': 'USDLDR'},
                                                    ],
                                             id="ticker-select",
                                             value = None,
                                             clearable=False,
                                             searchable=False,
                                             style={
                                                 'display': 'block',
                                                 },
                                             ),
                                         dcc.DatePickerRange(
                                             id='my-date-picker-range',
                                             start_date=dt.today() - relativedelta(years=5),
                                             end_date=dt.today(),
                                             ),
                                         ],
                                         style={
                                             'display': 'inline-block',
                                             },
                                         )
                                     ],
                                ),
                             dcc.Tab(
                                 label='Main view',
                                 style={'backgroundColor': "inherit"}, 
                                 selected_style={'backgroundColor': "inherit",'color':'white'},
                                 value = 'One',
                                 children = [
                                     html.Div(
                                         id="div-graphs",
                                             children=dcc.Graph(
                                                 id="graph_all_curves",
                                                 figure=dict(
                                                     layout=dict(
                                                         plot_bgcolor="#282b38",
                                                         paper_bgcolor="#282b38"
                                                     )
                                                 ),
                                             ),
                                         )
                                     ]
                                ),
                               dcc.Tab(
                                   label='Detailed view',
                                   value = 'Two',
                                   style={'backgroundColor': "inherit"}, 
                                   selected_style={'backgroundColor': "inherit",'color':'white'},
                                   children = [
                                       html.Div(
                                           id="dv",
                                           children=dcc.Graph(
                                               id="graph_sel_curves",
                                               figure=dict(
                                                   layout=dict(
                                                       plot_bgcolor="#282b38",
                                                       paper_bgcolor="#282b38"
                                                       )
                                                   ),
                                               ),
                                           )
                                       ]
                                   ),
                               ],
                         ),
                           html.Div(
                               id = 'selected_curves',
                               children = [],
                               style={'display': 'none'}
                               ),
                           html.Div(
                               id = 'old_selected_curves',
                               children = [],
                               style={'display': 'none'}
                               ),
                ],
                style={'width': '75%',
                       #'float': 'left'
                       }
                )


def div_panel():
    return html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                dcc.Upload(
                                    id='upload',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select FW file.')
                                        ]),
                                    disabled = True,
                                    style={'display': 'none'},
                                    #multiple = False,
                                    ),
                                card0(),
                                card1(),
                                card2(),
                                card3(),
                                html.Div(id='intermediate-value', style={'display': 'none'}),
                                dcc.Store(id='simulated_data'),
                                dcc.Store(id='visible_topvol', data = None),
                                dcc.Store(id='visible_lessvol', data = None),
                                dcc.Store(id='visible_random', data = None),
                        ],
                        #style={'width':'30%'},
                    ),
                        graph_tabs(),
                    ],
                    #style={'width':'30%'},
                    #style={'display': 'flex'},
            )
        ],
        style={'display': 'flex'},
    ) 
