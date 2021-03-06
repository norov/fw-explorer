import time
import numpy as np
import plotly
from plotly.tools import mpl_to_plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import matplotlib
import random


# Generate 3 graphs based on dict returned by generate constraints:
# Prices paths, Returns, Chartists share
def generate_graph_prod(ret, rnd, tv, lv, maxdd):
    pstar = np.mean(ret['pstar'], axis = 1)
    pstar_std = np.std(ret['pstar'], axis = 1)
    
    # Extract dict
    simple_R = np.array(ret["exog_signal"])
    prices = np.array(ret["prices"])

    Nc = np.array(ret["Nc"])
    
    len_sim = simple_R.shape[0]
    num_sim = simple_R.shape[1]
    
    # x axis
    x = [j for j in range(len_sim)]

    # Create figure
    fig = make_subplots(
        rows=5, cols=1,
        specs=[[{"rowspan": 2}],
               [None],
               [{}],
               [{}],
               [{}]],
        horizontal_spacing = 0.05,
        vertical_spacing = 0.15,
        shared_xaxes=True,
        subplot_titles=("Simulated Prices","Simulated Returns","Simulated Volatility", "Chartists share"))

    fig.update_layout(height=900,legend=dict(bordercolor="Black",borderwidth=0.5, font=dict(color='white')), 
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      hovermode="closest")
    
    def add_traces(fig, idx, color):
        # Generate graphs for each simulation
        for i in idx:
            # Prices
            fig.add_trace(go.Scattergl(x=x,y=prices[:,i],mode='lines',
                                     name='Sim_'+str(i+1),legendgroup='Sim'+str(i+1),
                                     marker=dict(color=color),line=dict(width=0.7)),row=1, col=1)
            # Returns
            fig.add_trace(go.Scattergl(x = x, y = simple_R[:,i],
                                       mode = 'lines',
                                       name = 'Sim_'+ str(i+1),
                                       legendgroup='Sim'+str(i+1),
                                       marker = dict(color=color),
                                       line = dict(width=0.7),
                                       showlegend=False),
                           row=3, col=1)
            # Vol
            fig.add_trace(go.Scattergl(x=x,y=np.abs(simple_R[:,i]),mode='lines',
                                     name='Sim_'+str(i+1),legendgroup='Sim'+str(i+1),
                                     marker=dict(color=color),line=dict(width=0.7),
                                     showlegend=False),row=4, col=1)
            # Chartists      
            fig.add_trace(go.Scattergl(x=x,y=Nc[:,i],mode='lines',
                                     name='Sim_'+str(i+1),legendgroup='Sim'+str(i+1),
                                     marker=dict(color=color),line=dict(width=0.7),
                                     showlegend=False),row=5, col=1)

    
    fig.add_trace(go.Scattergl(x = x, y = np.exp(pstar), mode = 'lines',
                                name = 'PSTAR', legendgroup = 'PSTAR',
                                marker = dict(color = 'rgb(255,255,255)'),
                                line = dict(width = 2)),
                    row=1, col=1)

    fig.add_trace(go.Scattergl(
                   name = 'PSTAR conf_int',
                   x = list(x) + list(x)[::-1],
                   y = list(np.exp(pstar - pstar_std)) + list(np.exp(pstar + pstar_std))[::-1],
                   fill = 'toself',
                   fillcolor = 'rgb(255,255,255)',
                   line = dict(color='rgba(255,255,255, 0)'),
                   opacity = 0.5,
                   hoverinfo="skip",
                   showlegend=False
                   ),
                    row=1, col=1)

    if rnd is not None:
        add_traces(fig, rnd, 'rgba(255,255,255,0.3)')

    if tv is not None:
        add_traces(fig, tv, 'rgba(255,0,0,0.8)')

    if lv is not None:
        add_traces(fig, lv, 'rgba(0,255,0,0.8)')

    if maxdd is not None:
        add_traces(fig, maxdd, 'rgba(0,0,255,0.8)')


    # Layout
    r = [1,3,4,5]
    for i in r:

        fig.update_xaxes(showgrid=False,zeroline=False,color='white', row=i, col=1)
        fig.update_yaxes(showgrid=False,zeroline=False,color='white', row=i, col=1)

    for l in fig['layout']['annotations']:
        l['font'] = dict(size=14,color='white')

    return fig



def plot_changes_params(swipe):

    s = swipe[0]
    swipe_type = s['swipe_type']
    hold = len(swipe) > 1

    fig = make_subplots(
        rows=4, cols=3,
        specs=[[{"colspan": 1},{"colspan": 1},{"colspan": 1}],
               [{"colspan": 1},{"colspan": 1},{"colspan": 1}],
               [{"colspan": 2, "rowspan":2},None,{"rowspan":2}],
               [None,None,None]
              ],
        subplot_titles=("Mean " + swipe_type,
                        swipe_type + " std",
                        swipe_type + " skew",
                        swipe_type + " kurtosis",
                        "Mean chartists level",
                        'Chartists distribution',
                        swipe_type + " distribution",
                        "QQ-Plot"))

    cc = ['black', 'red', 'blue', 'green']
    for idx, s in enumerate(swipe):
        param_range       = s[ 'param_range' ]
        param_mean        = np.array(s[ 'param_mean' ])
        param_vol         = s[ 'param_vol' ]
        param_skew        = s[ 'param_skew' ]
        param_kurt        = s[ 'param_kurt' ]
        chartists_mean    = s[ 'chartists_mean']
        distrib_ret       = s[ 'distrib_ret' ]
        distrib_chartists = s[ 'distrib_chartists']
        qqplots_graph     = s[ 'qqplots_graph']

        c = cc[idx % len(cc)]
        fig.add_trace(
	               go.Scattergl(x=param_range, y=param_mean[:, 0],
		            mode='lines',
                            showlegend=False,
		            name = 'Mean',
                            line = dict(color=c, width=2)),
                        row=1, col=1)
        fig.add_trace(
	               go.Scattergl(
		            name = 'Mean conf_int',
		            x = list(param_range) + list(param_range)[::-1],
			    y = list(param_mean[:, 1]) + list(param_mean[:, 2])[::-1],
			    fill = 'toself',
			    fillcolor = c,
			    line = dict(color='rgba(255,255,255, 0)'),
		            opacity = 0.2,
			    hoverinfo="skip",
			    showlegend=False
			    ),
                        row=1, col=1)

        fig.add_trace(go.Scattergl(x=param_range, y=param_vol, mode='lines',
                            showlegend=False,
                            line=dict(color=c, width=2)),
                        row=1, col=2)

        fig.add_trace(go.Scattergl(x=param_range, y=param_skew, mode='lines',
                            showlegend=False,
                            line=dict(color=c, width=2)),
                        row=1, col=3)

        fig.add_trace(go.Scattergl(x=param_range, y=param_kurt, mode='lines',
                            showlegend=False,
                            line=dict(color=c, width=2)),
                        row=2, col=1)

        fig.add_trace(go.Scattergl(x=param_range, y=chartists_mean, mode='lines',
                            showlegend=False,
                            line=dict(color=c, width=2)),
                        row=2, col=2)
    
        st = ['dash', None, 'dashdot']
        ms = ['square', 'circle', 'diamond']
        for i in range(len(distrib_ret)):
            if hold and i != 1:
                continue
            fig.add_trace(go.Scatter(distrib_ret[i],
                                    line=dict(dash = st[i], width=2, color = c),
                                    ), row=3, col=1)

        for i in range(len(distrib_chartists)):
            if hold and i != 1:
                continue
            fig.add_trace(go.Scatter(distrib_chartists[i],
                             line = dict(dash = st[i], width=2, color = c),
                                    ), row=2, col=3)

        for i, qq in enumerate(qqplots_graph):
            if hold and i != 1:
                continue
            x = np.array([qq[0][0][0], qq[0][0][-1]])
            fig.add_trace(go.Scattergl(x=qq[0][0], y=qq[0][1], mode='markers',
                                       marker_symbol = ms[i],
                                       line=dict( color = c, dash = st[i])),
                                       row=3, col=3)

            fig.add_trace(go.Scattergl(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines',
                                       line=dict(dash = st[i], color = c)),
                                       row=3, col=3)

            fig.layout.update(showlegend=False)
    
    
    fig.update_layout(legend=dict(bordercolor="Black",borderwidth=0.5, font=dict(color='white')), 
                      paper_bgcolor='rgba(0,0,0,0)',
                      hovermode="closest")
    
    for l in fig['layout']['annotations']:
        l['font'] = dict(size=14,color='white')
    
    fig.update_xaxes(showgrid=True,zeroline=False,color='white', row=1, col=1)
    fig.update_yaxes(showgrid=True,zeroline=False,color='white', row=1, col=1)
    fig.update_xaxes(showgrid=True,zeroline=False,color='white', row=1, col=2)
    fig.update_yaxes(showgrid=True,zeroline=False,color='white', row=1, col=2)
    fig.update_xaxes(showgrid=True,zeroline=False,color='white', row=1, col=3)
    fig.update_yaxes(showgrid=True,zeroline=False,color='white', row=1, col=3)
    
    fig.update_xaxes(showgrid=True,zeroline=False,color='white', row=2, col=1)
    fig.update_yaxes(showgrid=True,zeroline=False,color='white', row=2, col=1)
    fig.update_xaxes(showgrid=True,zeroline=False,color='white', row=2, col=2)
    fig.update_yaxes(showgrid=True,zeroline=False,color='white', row=2, col=2)
    fig.update_xaxes(showgrid=True,zeroline=False,color='white', row=2, col=3)
    fig.update_yaxes(showgrid=True,zeroline=False,color='white', row=2, col=3)
    
    fig.update_xaxes(showgrid=True,zeroline=False,color='white', row=3, col=1)
    fig.update_yaxes(showgrid=True,zeroline=False,color='white', row=3, col=1)
    
    fig.update_xaxes(showgrid=True,zeroline=False,color='white', row=3, col=3)
    fig.update_yaxes(showgrid=True,zeroline=False,color='white', row=3, col=3)
    
    
    return fig


def distrib_plots(ret, sel_curves):
    
    cc = ['red', 'blue', 'black', 'green', 'orange', 'yellow', 'cyan']
        
    # Extract dict
    simple_R = ret["exog_signal"]
    prices = ret["prices"]
    Nc = ret["Nc"][:, :]
    
    len_sim = simple_R.shape[0]
    num_sim = simple_R.shape[1]

    x = [j for j in range(len_sim)]

    fig_dist = ff.create_distplot([simple_R[:,i] for i in range(simple_R.shape[1])], 
                          group_labels=['ret_'+str(i+1) for i in range(simple_R.shape[1])], 
                          bin_size=.001)
    
    
    fig = go.FigureWidget(make_subplots(
    rows=5, cols=2,
    specs=[[{"rowspan": 2}, {"rowspan": 2}],
           [None, None],
           [{"rowspan": 2, "colspan": 2}, None],
           [None, None],
           [{"colspan": 2}, None]],
    horizontal_spacing = 0.05,
    vertical_spacing = 0.1,
    shared_xaxes=True,
    subplot_titles=("Prices", "Volatility", "Distribution of Returns")))

    yaxis = [0.1*i + 1 for i in range(simple_R.shape[1])]

    for i in range(prices.shape[1]):
        color_selected = random.randrange(0, 148)
        # top left
        fig.add_trace(go.Scatter(x=x,y=np.abs(simple_R[:,i]),mode='lines',
                                 name='Sim_'+str(sel_curves[i]+1),legendgroup='Sim'+str(i+1),
                                 marker=dict(color=cc[i]),line=dict(width=0.7),
                                 showlegend=True),row=1, col=2)
        # top right
        fig.add_trace(go.Scatter(x=x,y=prices[:,i],mode='lines',
                                 name='Sim_'+str(i+1),legendgroup='Sim'+str(i+1),
                                 marker=dict(color=cc[i]),line=dict(width=0.7),
                                 showlegend=False),row=1, col=1)
        # middle
        fig.add_trace(go.Histogram(fig_dist['data'][i],xbins=dict(size=0.002),
                                 name='Sim_'+str(i+1),legendgroup='Sim'+str(i+1),
                                 marker=dict(color=cc[i]),
                                 showlegend=False), row=3, col=1)
        
        fig.add_trace(go.Scatter(fig_dist['data'][i+num_sim],line=dict(width=1.5),
                                 name='Sim_'+str(i+1),legendgroup='Sim'+str(i+1),
                                 marker=dict(color=cc[i]),showlegend=False), 
                                 row=3, col=1)
        # bottom
        fig.add_trace(go.Scatter(x=simple_R[:,i], y = [0.1*i + 1 for j in range(simple_R[:,i].shape[0])], 
                                 mode = 'markers', marker=dict(color=cc[i], symbol='line-ns-open'),
                                 line=dict(width=0.7),name='Sim_'+str(i+1) ,showlegend=False,
                                 legendgroup='Sim'+str(i+1)), row=5, col=1)
        
    # Layout
    r = [1,3,5]
    fig.update_xaxes(showgrid=True,zeroline=False,color='white', row=1, col=2)
    fig.update_yaxes(showgrid=True,zeroline=False,color='white', row=1, col=2)
    for i in r: 
        fig.update_xaxes(showgrid=True,zeroline=False,color='white', row=i, col=1)
        fig.update_yaxes(showgrid=True,zeroline=False,color='white', row=i, col=1)
    
    for l in fig['layout']['annotations']:
        l['font'] = dict(size=14,color='white')
        
    fig.update_layout(height=700,legend=dict(bordercolor="Black",borderwidth=0.5, font=dict(color='white')), 
                      xaxis3_showticklabels=False,
                      paper_bgcolor='rgba(0,0,0,0)',
                      hovermode="closest")
    
    return fig



def generate_graph(ret):
    
    colors = px.colors.sequential.Rainbow
    simple_R = ret["exog_signal"][0, :, :]
    prices = np.cumprod(simple_R +1,0)
    
    num_sim = prices.shape[1]

    x = [j for j in range(prices.shape[0])]
    
    fig_dist = ff.create_distplot([simple_R[:,i] for i in range(simple_R.shape[1])], 
                              group_labels=['ret_'+str(i+1) for i in range(simple_R.shape[1])], 
                              bin_size=.001)

    fig = make_subplots(
        rows=5, cols=2,
        specs=[[{"rowspan": 2}, {"rowspan": 2}],
               [None, None],
               [{"rowspan": 2, "colspan": 2}, None],
               [None, None],
               [{"colspan": 2}, None]],
        horizontal_spacing = 0.05,
        vertical_spacing = 0.1,
        subplot_titles=("Returns","Prices", "Distribution of Returns"))

    yaxis = [0.1*i + 1 for i in range(simple_R.shape[1])]
    
    
    for i in range(prices.shape[1]):
        # top left
        fig.add_trace(go.Scatter(x=x,y=simple_R[:,i],mode='lines',
                                 name='ret_'+str(i+1),legendgroup='Sim'+str(i+1),
                                 marker=dict(color='rgba(0,0,0,0.1)')),row=1, col=1)
        # top right
        fig.add_trace(go.Scatter(x=x,y=prices[:,i],mode='lines',
                                 name='price_path'+str(i+1),legendgroup='Sim'+str(i+1),
                                 marker=dict(color=colors[i])),row=1, col=2)
        # middle
        fig.add_trace(go.Histogram(fig_dist['data'][i],xbins=dict(size=0.002),
                                   legendgroup='Sim'+str(i+1), marker=dict(color=colors[i])), row=3, col=1)
        fig.add_trace(go.Scatter(fig_dist['data'][i+num_sim],line=dict(width=2.5),
                                 legendgroup='Sim'+str(i+1), marker=dict(color=colors[i])), row=3, col=1)
        # bottom
        fig.add_trace(go.Scatter(x=simple_R[:,i], y = [1- 0.01*i for j in range(simple_R[:,i].shape[0])], 
                                 mode = 'markers', name='ret_'+str(i+1), legendgroup='Sim'+str(i+1),
                                 marker=dict(color=colors[i], symbol='line-ns-open')), row=5, col=1)

    fig.update_layout(height=800,legend=dict(bordercolor="Black",borderwidth=0.5), title_text="specs examples")
    
    return fig
