import dash
import dash_core_components as dcc
import dash_html_components as html
import utils.dash_reusable_components as drc
import numpy as np
import pandas as pd

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
                value=value[i]
                ) for i in range(len(name))
            ], style={
                "display": "flex",
                "width": "95%",
                },
            )

def card1():
    return drc.Card(
        id="first-card",
        children=[
            dcc.Upload(
                id='upload',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select FW file.')
                    ]),
                #multiple = False,
                ),
            drc.NamedRadioItems(
                name='Model',
                id='model-type',
                labelStyle={
                    'margin-right': '7px',
                    'display': 'inline-block'
                    },
                #options=[
                #    {'label': typ, 'value': typ}
                #    for typ in fw_params['Type'].unique()
                #    ],
                value=None,
                ),
            dcc.Dropdown(
                id="model",
                # options=[
                #     {'label': typ, 'value': typ}
                #     for typ in fw_params['Model'].unique()
                #     ],
                value = None,
                clearable=False,
                searchable=False,
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
            div_input_line(
                ["Periods", "Paths"],
                ["periods", "paths"],
                [500, 200],
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
            ]
        )
    

def card2():
    return drc.Card(
        id="button-card",
        children=[
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
            html.P(children="Statistics here"),
            ],
        )

def dcc_graph():
    return html.Div(
            id="div-graphs",
            children=dcc.Graph(
                id="graph-sklearn-svm",
                figure=dict(
                    layout=dict(
                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                        )
                    ),
                ),
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
                                card1(),
                                card2(),
                                card3(),
                                html.Div(id='intermediate-value',
                                    style={'display': 'none'})
                        ],
                    ),
                        dcc_graph(),
                    ],
            )
        ],
    ) 
