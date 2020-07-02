from textwrap import dedent

import dash_core_components as dcc
import dash_html_components as html


# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


# Custom Display Components
def Card(children, **kwargs):
    return html.Section(className="card", children=children, **_omit(["style"], kwargs))


def FormattedSlider(**kwargs):
    return html.Div(
        style=kwargs.get("style", {}), children=dcc.Slider(**_omit(["style"], kwargs))
    )


def NamedSlider(name, **kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[
            html.P(f"{name}:"),
            html.Div(style={"margin-left": "6px"}, children=dcc.Slider(**kwargs)),
            ],
    )


istyle = {'color': 'inherit',
          'width': '30%',
         }

def CheckboxSwipe(cname, swipe, enabled = False,  **kwargs):
    cid = cname + '_slider'
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[
            dcc.Checklist(
                id = cid,
                options = [
                    {"label": cname, "value": 'rvmean'},
                ],
                value = []
            ),
            html.Div(style={"margin-left": "6px"},
                children = [
                    dcc.Input(id = cname + 'start',
                        value = swipe[0],
                        type = 'number',
                        style = istyle,
                        ),
                    dcc.Input(id = cname + 'step',
                        value = swipe[1],
                        type = 'number',
                        style = istyle,
                        ),
                    dcc.Input(id = cname + 'stop',
                        value = swipe[2],
                        type = 'number',
                        style = istyle,
                        ),
                    ]
                ),
            ],
    )

def CheckboxSlider(cname, cid, enabled = False, **kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[
            dcc.Checklist(
                id = cid,
                options = [
                    {"label": cname, "value": 'rvmean'},
                ],
                value = []
            ),
            html.Div(style={"margin-left": "6px"},
                children=dcc.Slider(**kwargs)),
            ],
    )


def NamedInput(name, **kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[
            html.P(f"{name}:"),
            html.Div(style={"margin-left": "6px"},
                    children=dcc.Input(**kwargs, className = "form-control")),
        ],
    )


def ButtonInput(btn_name, btn_id, **kwargs):
    return html.Div(
        style={#"padding": "20px 10px 25px 4px",
                "display": "inline-block",
                 'width': '100%',
                },
        children=[
            html.Button(btn_name, btn_id,
                style = {'color': 'inherit',
                         'width': '60%',
                         'float': 'left'}
                ),
            dcc.Input(**kwargs,
                type = 'number',
                style = {'color': 'inherit',
                         'width': '30%',
                         'float': 'right'}),
        ],
    )


def NamedDropdown(name, **kwargs):
    return html.Div(
        style={"margin": "10px 0px"},
        children=[
            html.P(children=f"{name}:", style={"margin-left": "3px"}),
            dcc.Dropdown(**kwargs),
        ],
    )


def NamedRadioItems(name, **kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[html.P(children=f"{name}:"), dcc.RadioItems(**kwargs)],
    )


# Non-generic
def DemoDescription(filename, strip=False):
    with open(filename, "r") as file:
        text = file.read()

    if strip:
        text = text.split("<Start Description>")[-1]
        text = text.split("<End Description>")[0]

    return html.Div(
        className="row",
        style={
            "padding": "15px 30px 27px",
            "margin": "45px auto 45px",
            "width": "80%",
            "max-width": "1024px",
            "borderRadius": 5,
            "border": "thin lightgrey solid",
            "font-family": "Roboto, sans-serif",
        },
        children=dcc.Markdown(dedent(text)),
    )
