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
            drc.ButtonInputButton('Random seed', 'Set seed',
                'rnd_seed', 'set_seed',
                id =  'seed_val', value = 0),
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
                    'color': 'inherit',
                    "display": "flex",
                    "width": "95%",
                    },
                    ),
                html.Button(
                    "Delete",
                    id="btn-delete",
                    style={
                    'color': 'inherit',
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
def card_save():
    return drc.Card(
        id = 'save_card',
        children=[
            html.Div([
                dcc.Input(
                    id="filename",
                    type='text',
                    style={
                    'color': 'inherit',
                    "display": "flex",
                    "width": "95%",
                    },
                    placeholder="File Name"
                    ),
                html.Button(
                    "Save",
                    id="btn_save",
                    style={
                    'color': 'inherit',
                    "display": "flex",
                    "width": "95%",
                    },
                    ),
            ], style={
                "display": "flex",
                },
            ),
            html.Div([
            dcc.Dropdown(
                id="load_filename",
                value = None,
                clearable=False,
                searchable=False,
                style={
                    'color': 'inherit',
#                    "display": "flex",
                    "width": "100%",
                },
                placeholder="File Name"
                ),
                html.Button(
                    "Load",
                    id="btn_load",
                    style={
                    'color': 'inherit',
                    "display": "flex",
                    #"width": "30%",
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

def card_swipe():
    istyle = {'color': 'inherit',
            'width': '95%',
             }
    return drc.Card(
        id = "card_swipe",
        hidden = True,
        children=[
             dcc.Dropdown(
                options = [
                        {'label': 'Phi',     'value': 'phi'},
                        {'label': 'Chi',     'value': 'chi'},
                        {'label': 'Eta',     'value': 'eta'},
                        {'label': 'alpha_w', 'value': 'alpha_w'},
                        {'label': 'alpha_o', 'value': 'alpha_O'},
                        {'label': 'alpha_n', 'value': 'alpha_n'},
                        {'label': 'alpha_p', 'value': 'alpha_p'},
                        {'label': 'sigma_f', 'value': 'sigma_f'},
                        {'label': 'sigma_c', 'value': 'sigma_c'},
                        ],
                 id="swipe-select",
                 value = 'phi',
                 clearable=False,
                 searchable=False,
                 style={
                     'display': 'block',
                     },
                 ),
            html.Div(
                style={
                    #"margin-left": "6px",
                     'display': 'flex',
                    },
                children = [
                    drc.NInput('Start', id = 'swipe_start', value = None,
                        type = 'number', style = istyle,),

                    drc.NInput('Step', id = 'swipe_step', value = None,
                        type = 'number', style = istyle,),

                    drc.NInput('Stop', id = 'swipe_stop', value = None,
                        type = 'number', style = istyle,),
                    ]
                ),
             dcc.Dropdown(
                options = [
                        {'label': 'Price',   'value': 'Price'},
                        {'label': 'Return',  'value': 'Return'},
                        ],
                 id="swipe-type",
                 value = 'Return',
                 clearable=False,
                 searchable=False,
                 style={
                     'display': 'block',
                     },
                 ),
            html.Div(
                id = 'return_params',
                style={
                     'display': 'flex',
                     #'display': 'none',
                    },
                children = [
                    drc.NInput('Start Period', id = 'start_period', value = 2,
                        type = 'number', style = istyle,),

                    drc.NInput('Stop Period', id = 'stop_period', value = None,
                        type = 'number', style = istyle,),

                    ]
                ),
            dcc.Checklist(
                id = 'hold',
                options=[
                    {'label': 'Hold', 'value': 'hold'},
                ],
                ),
            html.Button(
                "Swipe",
                id="btn_swipe",
                style = {'color': 'inherit',
                         'display': 'flex',
                         "width": "100%"},
                )
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
                min=0.1,
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
            drc.CheckboxSlider("Fundamental price window", 'rvmean_cb', enabled = False,
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
            drc.CheckboxSlider("Effective return window", 'retmean_cb', enabled = False,
                id = 'retmean',
                min = 1,
                max = 21,
                marks={
                    i: str(i)
                    for i in range(0, 21, 5)
                    },
                tooltip = {
                    "always_visible": False,
                    "placement": "top"
                    },
                value = 5,
                ),
            html.Button(
                "Simulate",
                id="btn-simulate",
                style = {'color': 'inherit', },
                ),
            dcc.Checklist(
                id = 'pick_checkbox',
                options=[
                    {'label': 'Start from picked point', 'value': 'enable'},
                ],
		value = []
                ),
            html.Button(
                "Pick Start",
                id="pick_start",
                style = {'color': 'inherit', },
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
                drc.ButtonInput('Least volatile', 'lessvol',
                   id =  'lessvol_val', value = 10),
                drc.ButtonInput('Max drawdown', 'maxdd',
                   id =  'maxdd_val', value = 10),
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
                                             value = '^GSPC',
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
                               dcc.Tab(
                                   label='Sensitivity',
                                   value = 'tab_sensitivity',
                                   style={'backgroundColor': "inherit"}, 
                                   selected_style={'backgroundColor': "inherit",'color':'white'},
                                   children = [
                                       html.Div(
                                           id="sens",
                                           #style = {'display': 'inline-block'},
                                           children = [
                                               dcc.Graph(
                                                   id="phi_sens",
                                                   figure=dict(
                                                       layout=dict(
                                                           plot_bgcolor="#282b38",
                                                           paper_bgcolor="#282b38"
                                                           )
                                                       ),
                                                   ),
                                               dcc.Graph(
                                                   id="chi_sens",
                                                   figure=dict(
                                                       layout=dict(
                                                           plot_bgcolor="#282b38",
                                                           paper_bgcolor="#282b38"
                                                           )
                                                       ),
                                                   ),
                                               ]
                                           )
                                       ]
                                   ),
                               dcc.Tab(
                                   label='User Comments',
                                   value = 'Comment',
                                   style={'backgroundColor': "inherit"}, 
                                   selected_style={'backgroundColor': "inherit",'color':'white'},
                                   children = [
                                       html.Div(
                                           id="comments_title",
                                           #style = {'display': 'inline-block'},
                                           children = [
                                               dcc.Markdown(''' ## User Comments '''),
                                               ]
                                           ),
                                       html.Div(
                                           #id="comments_txt",
                                           #style = {'display': 'inline-block'},
                                           children = [
                                               dcc.Textarea(id="comments_txt", 
                                                            style={'width': 1500, 
                                                                   'height':600, 
                                                                   'font-size': 18,
                                                                   'color':'white'}),
                                               ],
                                           style=dict(display='flex')
                                           )
                                       ]
                                   ),
                               dcc.Tab(
                                   label='User Manual',
                                   value = 'Manual',
                                   style={'backgroundColor': "inherit"}, 
                                   selected_style={'backgroundColor': "inherit",'color':'white'},
                                   children = [
                                       html.Div(
                                           id="manual",
                                           #style = {'display': 'inline-block'},
                                           children = [
                                               dcc.Markdown('''
## User Manual

### Model controls.
#### Section 1. Basic settings

This is the Exploration tool for the Franke-Westerhoff structural stochastic volatility model.
Start working with the explorer by choosing appropriate **SEED**, or let the tool pick random
value for you.

7 models from the original paper are presented as presets. At the end of the first section,
choose a model to simulate from the drop box. The models may be customized by user if needed.
All parameters are editable by clicking **SHOW / HIDE** button. Original values of the parameters
are preserved while experimenting.

#### Section 2. Model arguments

Use **Periods** and **Paths** input boxes to set length of simulation and number of individual
trajectory paths accordingly.

Market Liquidity and Switching Strength sliders set corresponding model parameters as appropriate.

**Fundamental price window** and **Effective return window** are two extensions added to the model
that allow to extend simulation flexibility.

**Fundamental price window** is used to determine a time period for calculating the fundamental price.
If disabled, the fundamental price is always **1**, while if the feature enabled, fundamental price
is calculated as average price for last **N** periods, as set on slider control. This control affects
the Fundamentalists behaviour. If at some specific simulation price deviates from **1** for a period of
time comparable to the window, the price effectively becomes a new fundamental value.

**Effective return window** affects the behaviour of chartist part of simulation. If disabled, chartists's
prediction on market is based on last return. If the feature enabled, few last returns average used
to calculate the 'effective return'. This feature may be used to simulate 'weekly chartists', for example.

**Simulate** button starts the simulaton. No simulation traces will appear when simulation is complete.
Use the following section controls to display results.

**Pick Start** and **Start from picked point** are discussed in Model Display.

#### Section 3. Display Results

Displaying all simulated traces is computationally difficult problem and trashes output with numerous random
lines. We structured output traces by properties, and marked them with individual colours. User can pick 
traces with desired properties to display at will, together with the number of traces per option.

The supported properties are:
* Random: pick random traces;
* Most volatile: pick N most volatile traces;
* Least volatile: pick N least volatile traces;
* Max drawdown: pick traces demonstrating maximal loss in price.

The model outputs is displayed at the right part of the screen. It is also interactive and is discussed at
the section **Model Display**


#### Section 3. Saving Results

**Save/Load** buttons together with corresponding input and dropbox allow to save and restore the state
of the dash at the server, and recover an interesting simulation later at convenience. 

### Model Display.

The program has interactive graphical representation of the model output. There are **Main Vieew** and 
**Detailed View** tabs to explore the model.

''')
                                               ]
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
            #className="container scalable",
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
                                card_swipe(),
                                card2(),
                                card3(),
                                card_save(),
                                dcc.Store(id='cal_params'),
				dcc.Store(id='click_data'),
                                dcc.Store(id='start_params'),
                                dcc.Store(id='simulated_data'),
                                dcc.Store(id='visible_topvol', data = None),
                                dcc.Store(id='visible_lessvol', data = None),
                                dcc.Store(id='visible_maxdd', data = None),
                                dcc.Store(id='visible_random', data = None),
                                dcc.Store(id='Swipe_data', data = None),
                                dcc.Store(id='Load_trigger', data = None),
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
