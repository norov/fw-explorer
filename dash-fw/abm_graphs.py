import numpy as np
import plotly
from plotly.tools import mpl_to_plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px


# Generate 3 graphs based on dict returned by generate constraints:
# Prices paths, Returns, Chartists share
def generate_graph_prod(ret):
    
    # Extract dict
    simple_R = ret["exog_signal"][0, :, :]
    prices = np.cumprod(simple_R +1,0)
    Nc = ret["Nc"][:, :]
    
    len_sim = simple_R.shape[0]
    num_sim = simple_R.shape[1]
    
    # x axis
    x = [j for j in range(len_sim)]

    # Create figure
    fig = make_subplots(
        rows=4, cols=1,
        specs=[[{"rowspan": 2}],
               [None],
               [{}],
               [{}]],
        horizontal_spacing = 0.05,
        vertical_spacing = 0.15,
        shared_xaxes=True,
        subplot_titles=("Simulated Prices","Simulated Returns", "Chartists share"))

    # Generate graphs for each simulation
    for i in range(prices.shape[1]):
        
        # Prices
        fig.add_trace(go.Scatter(x=x,y=prices[:,i],mode='lines',
                                 name='Sim_'+str(i+1),legendgroup='Sim'+str(i+1),
                                 marker=dict(color='rgba(255,255,255,0.3)'),line=dict(width=0.7)),row=1, col=1)
        
        # Returns
        fig.add_trace(go.Scatter(x=x,y=simple_R[:,i],mode='lines',
                                 name='Sim_'+str(i+1),legendgroup='Sim'+str(i+1),
                                 marker=dict(color='rgba(255,255,255,0.3)'),line=dict(width=0.7),
                                 showlegend=False),row=3, col=1)
        
        # Chartists      
        fig.add_trace(go.Scatter(x=x,y=Nc[:,i],mode='lines',
                                 name='Sim_'+str(i+1),legendgroup='Sim'+str(i+1),
                                 marker=dict(color='rgba(255,255,255,0.3)'),line=dict(width=0.7),
                                 showlegend=False),row=4, col=1)
    
    # Layout
    fig.update_xaxes(showgrid=False,zeroline=False,color='white', row=1, col=1)
    fig.update_yaxes(showgrid=False,zeroline=False,color='white', row=1, col=1)
    fig.update_xaxes(showgrid=False,zeroline=False,color='white', row=3, col=1)
    fig.update_yaxes(showgrid=False,zeroline=False,color='white', row=3, col=1)
    fig.update_xaxes(showgrid=False,zeroline=False,color='white', row=4, col=1)
    fig.update_yaxes(showgrid=False,zeroline=False,color='white', row=4, col=1)
    
    for l in fig['layout']['annotations']:
        l['font'] = dict(size=14,color='white')
    
    fig.update_layout(height=700,legend=dict(bordercolor="Black",borderwidth=0.5, font=dict(color='white')), 
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
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