
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_features(features_to_plot, x_feature, df, plot_width=800, plot_height=400, mode='lines'):
    '''This function creates an interactive plot of specified size 
    of specified 'features_to_plot' vs specified x_feature.
    Select mode='lines', mode='markers' or mode='lines+markers'.
    '''

    # Define a list of colors for the plots
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    # Create a subplot figure with shared x-axes
    fig = make_subplots(rows=len(features_to_plot), cols=1, shared_xaxes=True, vertical_spacing=0.03)

    for i, feature in enumerate(features_to_plot):
        if feature not in df.columns:
            raise KeyError(f"Feature '{feature}' not found in DataFrame columns")
        
        # Select a color from the list, cycling through if there are more features than colors
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=df[x_feature],
            y=df[feature],
            mode=mode,
            name=f'Actual {feature}',
            line=dict(color=color),
            hovertemplate=f'Step: %{{text}}<br>{x_feature}: %{{x}}<br>{feature}: %{{y}}<extra></extra>',
            text=df['Step']  # Add the time to the hover text
        ), row=i+1, col=1)
        
        # Update y-axis title for each subplot
        fig.update_yaxes(title_text=feature, row=i+1, col=1)
        
        # Ensure x-axis labels are displayed for every subplot
        fig.update_xaxes(title_text=x_feature, row=i+1, col=1, showticklabels=True)

    fig.update_layout(
        showlegend=False,  # Hide the legend
        width=plot_width,  # Set the width of the plot
        height=plot_height * len(features_to_plot),  # Adjust height based on the number of subplots
    )
    
    # Show the plot
    fig.show()


def plot_features_plotly(features_to_plot, df, x_axis='index', title=None, y_axis_title='Title', plot_width=800, plot_height=300, x_min=0, x_max=170, y_min=2.4, y_max=4.4):
    """ Plots selected features from a DataFrame on the same Plotly figure. """

    # Check if features exist in DataFrame
    for feature in features_to_plot:
        if feature not in df.columns:
            raise KeyError(f"Feature '{feature}' not found in DataFrame columns")

    # Check if x_axis is a valid column or 'index'
    if x_axis != 'index' and x_axis not in df.columns:
        raise KeyError(f"X-axis field '{x_axis}' not found in DataFrame columns")

    # Create a list of colors for the plots
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    # Create the figure
    fig = go.Figure()

    # Determine x values
    x_values = df.index if x_axis == 'index' else df[x_axis]

    # Iterate through features and add traces to the figure
    for i, feature in enumerate(features_to_plot):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df[feature],
            mode='lines',
            name=feature,
            line=dict(color=color, width=1)
        ))

    # Update y-axis title
    fig.update_yaxes(title_text=y_axis_title)

    # Update x-axis range if specified
    if x_min is not None or x_max is not None:
        fig.update_xaxes(range=[x_min, x_max])

    # Update y-axis range if specified
    if y_min is not None or y_max is not None:
        fig.update_yaxes(range=[y_min, y_max])

    # Update layout with reduced margins
    fig.update_layout(
        title=title,
        xaxis_title=x_axis if x_axis != 'index' else 'Index',
        width=plot_width,
        height=plot_height,
        margin=dict(l=20, r=20, t=40, b=20)  # Adjust these values as needed
    )

    # Show the plot
    fig.show()




def overlap_plot(df, x='Cycle Time (h)', y=['Voltage (V)'], plot_width=800, plot_height=300, plot_title=''):
    ''' 
    This function plots one or more features for all cycles
    '''
    
    # Ensure y is a list
    if isinstance(y, str):
        y = [y]
    
    # Initialize the plotly figure
    fig = go.Figure()
    
    # Loop through each cycle and each feature to add traces
    for step, data in df.groupby('Step'):
        for feature in y:
            fig.add_trace(
                go.Scatter(
                    x=data[x],
                    y=data[feature],
                    mode='lines',
                    name=f'Step {step} - {feature}'
                )
            )
    
    # Customize the layout
    fig.update_layout(
        title=plot_title,
        xaxis_title=x,
        yaxis_title=', '.join(y),
        legend_title='Steps & Features',
        width=plot_width,
        height=plot_height  
    )
    
    fig.show()



    
def plot_dataframe(df, x_field, y_field, plot_type='line', line_thickness=2, dot_size=50, dot_border=1, label=None, color='blue', ax=None):
    ''' Plots df fields as line, scatter or line_plus_dots'''
    if ax is None:
        ax = plt.gca()
    
    if plot_type == 'line':
        ax.plot(df[x_field], df[y_field], linewidth=line_thickness, label=label, color=color)
    elif plot_type == 'scatter':
        ax.scatter(df[x_field], df[y_field], s=dot_size, edgecolor='black', linewidth=dot_border, label=label, color=color)
    elif plot_type == 'line_plus_dots':
        ax.plot(df[x_field], df[y_field], linewidth=line_thickness, label=label, color=color)
        ax.scatter(df[x_field], df[y_field], s=dot_size, edgecolor='black', linewidth=dot_border, label=label, color=color)
    else:
        raise ValueError("Invalid plot_type. Choose from 'line', 'scatter', or 'line_plus_dots'.")
    
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    if label:
        ax.legend()


def plot_dual_axis(df, x_field, y1_field, y2_field, width=800, height=600):
    fig = go.Figure()

    # Add first trace for the first y-axis
    fig.add_trace(
        go.Scatter(
            x=df[x_field],
            y=df[y1_field],
            name=y1_field,
            yaxis='y1'
        )
    )

    # Add second trace for the second y-axis
    fig.add_trace(
        go.Scatter(
            x=df[x_field],
            y=df[y2_field],
            name=y2_field,
            yaxis='y2'
        )
    )

    # Update layout for dual y-axes and size
    fig.update_layout(
        title=f'{y1_field} and {y2_field} vs {x_field}',
        xaxis=dict(title=x_field),
        yaxis=dict(
            title=y1_field,
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title=y2_field,
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        ),
        width=width,
        height=height,
        showlegend=False
    )

    fig.show()




def plot_multiple_features(df, x_feature, y_features):
    ''' Plots one feature on axis 1 vs many scaled features on axis 2'''
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[y_features])

    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel(x_feature)
    ax1.set_ylabel(y_features[0], color=color)
    ax1.plot(df[x_feature], scaled_features[:, 0], color=color, label=y_features[0])
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.plot(df[x_feature], scaled_features[:, 1], color=color, label=y_features[1])
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticklabels([])  # remove y-axis labels

    color = 'tab:green'
    ax2.plot(df[x_feature], scaled_features[:, 2], color=color,  label=y_features[2])
    ax2.tick_params(axis='y', labelcolor=color)

    ax2.set_ylabel('')  # remove the label of the second y-axis

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    
    plt.show()




def overlap_plot(df, feature='U (V)', mode='lines+markers'):
    ''' 
    This function plots a feature for all cycles except for the last one and a specified number of the first ones.
    '''

    # Get unique cycle numbers sorted in ascending order
    cycle_numbers = sorted(df['Cycle number'].unique())

    # Define cycles to exclude: first n cycles and the last one
    #cycles_to_exclude = cycle_numbers[:cycles_to_hide] + [cycle_numbers[-1]]

    # Filter the dataframe to exclude the specified cycles
    #filtered_df = df[~df['Cycle_Number'].isin(cycles_to_exclude)]

    # Initialize the plotly figure
    fig = go.Figure()

    # Loop through each cycle and add traces
    for cycle, data in df.groupby('Cycle number'):
        fig.add_trace(
            go.Scatter(
                x=data['Cycle time [h]'],
                y=data[feature],
                mode=mode,
                name=f'Cycle {cycle}'
            )
        )

    # Customize the layout
    fig.update_layout(
        title='Cycles Overlapped',
        xaxis_title='Cycle time [h]',
        yaxis_title=feature,
        legend_title='Cycles',
        width=900,  # Adjust width as needed
        height=500  # Adjust height as needed
    )

    fig.show()    


    import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_features(features_to_plot, x_feature, df, plot_width=800, plot_height=400, mode='lines'):
    '''This function creates an interactive plot of specified size 
    of specified 'features_to_plot' vs specified x_feature
    select mode='lines', mode='markers' or mode='lines+markers'

    '''

    # Define a list of colors for the plots
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    # Create a subplot figure without shared x-axes
    fig = make_subplots(rows=len(features_to_plot), cols=1, shared_xaxes=True, vertical_spacing=0.02)



    for i, feature in enumerate(features_to_plot):
        if feature not in df.columns:
            raise KeyError(f"Feature '{feature}' not found in DataFrame columns")
        
        # Select a color from the list, cycling through if there are more features than colors
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=df[x_feature],
            y=df[feature],
            mode=mode,
            name=f'Actual {feature}',
            line=dict(color=color),
            #hovertemplate=f'Cycle Number: %{{text}}<br>{x_feature}: %{{x}}<br>{feature}: %{{y}}<extra></extra>',
            #text=df['Cycle Number']  # Add the cycle number to the hover text
        ), row=i+1, col=1)
        
        # Update y-axis title for each subplot
        fig.update_yaxes(title_text=feature, row=i+1, col=1)
        
        # Update x-axis title for each subplot
        fig.update_xaxes(title_text=x_feature, row=i+1, col=1)
    
    fig.update_layout(
        showlegend=False,  # Hide the legend
        width=plot_width,  # Set the width of the plot
        height=plot_height * len(features_to_plot)  # Adjust height based on the number of subplots
    )
    
    # Show the plot
    fig.show()


def overlap_plot(df, feature='U (V)', mode='lines+markers'):
    ''' 
    This function plots a feature for all cycles except for the last one and a specified number of the first ones.
    '''

    # Get unique cycle numbers sorted in ascending order
    cycle_numbers = sorted(df['Cycle number'].unique())

    # Define cycles to exclude: first n cycles and the last one
    #cycles_to_exclude = cycle_numbers[:cycles_to_hide] + [cycle_numbers[-1]]

    # Filter the dataframe to exclude the specified cycles
    #filtered_df = df[~df['Cycle_Number'].isin(cycles_to_exclude)]

    # Initialize the plotly figure
    fig = go.Figure()

    # Loop through each cycle and add traces
    for cycle, data in df.groupby('Cycle number'):
        fig.add_trace(
            go.Scatter(
                x=data['Cycle time [h]'],
                y=data[feature],
                mode=mode,
                name=f'Cycle {cycle}'
            )
        )

    # Customize the layout
    fig.update_layout(
        title='Cycles Overlapped',
        xaxis_title='Cycle time [h]',
        yaxis_title=feature,
        legend_title='Cycles',
        width=900,  # Adjust width as needed
        height=500  # Adjust height as needed
    )

    fig.show()    



import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_features(features_to_plot, x_feature, df, plot_width=800, plot_height=400, mode='lines'):
    '''Creates an interactive plot of specified size 
    of specified 'features_to_plot' vs specified x_feature.
    Select mode='lines', mode='markers' or mode='lines+markers'.
    '''

    # Define a list of colors for the plots
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    # Create a subplot figure with shared x-axes
    fig = make_subplots(rows=len(features_to_plot), cols=1, shared_xaxes=True, vertical_spacing=0.05)

    for i, feature in enumerate(features_to_plot):
        if feature not in df.columns:
            raise KeyError(f"Feature '{feature}' not found in DataFrame columns")
        
        # Select a color from the list, cycling through if there are more features than colors
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=df[x_feature],
            y=df[feature],
            mode=mode,
            name=f'Actual {feature}',
            line=dict(color=color),
        ), row=i+1, col=1)
        
        # Update y-axis and x-axis properties for each subplot
        fig.update_yaxes(
            title_text=feature,
            row=i+1,
            col=1,
            showgrid=True,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True
        )
        
        fig.update_xaxes(
            title_text=x_feature if i == len(features_to_plot) - 1 else "",
            row=i+1,
            col=1,
            showticklabels=True,
            showgrid=True,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True
        )

    fig.update_layout(
        showlegend=False,
        width=plot_width,
        height=plot_height * len(features_to_plot),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    fig.show()




import plotly.graph_objects as go

def overlap_plot(df, x='Cycle Time (h)', y=['Voltage (V)'], plot_width=800, plot_height=300, plot_title='', groupby_field='Cycle number'):
    ''' 
    Generates an interactive plot of multiple features for all cycles
    '''

    # Initialize the plotly figure
    fig = go.Figure()

    # Loop through each cycle and add traces for each y-feature
    for f_value, data in df.groupby(groupby_field):
        for feature in y:
            fig.add_trace(
                go.Scatter(
                    x=data[x],
                    y=data[feature],
                    mode='lines',
                    name=f'{groupby_field} = {f_value}'
                )
            )

    # Customize the layout: white background, gridlines, full frame
    fig.update_layout(
        title=plot_title,
        width=plot_width,
        height=plot_height,
        legend_title='Cycles and Features',
        xaxis_title=x,
        yaxis_title=', '.join(y),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True
        )
    )

    fig.show()



import matplotlib.pyplot as plt

def plot_dataframe(df, x_field, y_field, plot_type='line', line_thickness=2, dot_size=50, dot_border=1, label=None, color='blue', ax=None, secondary_y=False):
    '''  Plots df featres as line, scatter, line_plus_dots, or dash using matplotlib.       The function can be called several times to plot on the same plot.
        It can also plot a feature on the secondary axis if secondary_y=True. In this case include fig, ax = plt.subplots() before calling the function 
       Choose from 'line', 'scatter', 'line_plus_dots', or 'dash'''

    if ax is None:
        ax = plt.gca()
    
    if secondary_y:
        ax = ax.twinx()
    
    if plot_type == 'line':
        ax.plot(df[x_field], df[y_field], linewidth=line_thickness, label=label, color=color)
    elif plot_type == 'scatter':
        ax.scatter(df[x_field], df[y_field], s=dot_size, edgecolor='black', linewidth=dot_border, label=label, color=color)
    elif plot_type == 'line_plus_dots':
        ax.plot(df[x_field], df[y_field], linewidth=line_thickness, label=label, color=color)
        ax.scatter(df[x_field], df[y_field], s=dot_size, edgecolor='black', linewidth=dot_border, label=label, color=color)
    elif plot_type == 'dash':
        ax.plot(df[x_field], df[y_field], linewidth=line_thickness, linestyle='--', label=label, color=color)
    else:
        raise ValueError("Invalid plot_type. Choose from 'line', 'scatter', 'line_plus_dots', or 'dash'.")
    
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    #if label:
       # ax.legend()



def plot_dataframe_with_hue(df, x_field, y_field, plot_type='line', line_thickness=2, dot_size=50, dot_border=1, label=None, color='blue', ax=None, secondary_y=False, color_field=None, dash_pattern=(5, 5)):
    ''' Plots df features as line, scatter, line_plus_dots, or dash using matplotlib.
        The function can be called several times to plot on the same plot.
        It can also plot a feature on the secondary axis if secondary_y=True. In this case include fig, ax = plt.subplots() before calling the function.
        If color_field is provided, it will be used to color the dots in the scatter plot.
    '''
    if ax is None:
        ax = plt.gca()
    
    if secondary_y:
        ax = ax.twinx()
    
    if plot_type == 'line':
        ax.plot(df[x_field], df[y_field], linewidth=line_thickness, label=label, color=color)
    elif plot_type == 'scatter':
        if color_field:
            unique_values = df[color_field].unique()
            colors = plt.cm.get_cmap('viridis', len(unique_values))
            color_map = {val: colors(i) for i, val in enumerate(unique_values)}
            df['color'] = df[color_field].map(color_map)
            for val in unique_values:
                subset = df[df[color_field] == val]
                ax.scatter(subset[x_field], subset[y_field], s=dot_size, edgecolor='black', linewidth=dot_border, label=val, c=subset['color'])
        else:
            ax.scatter(df[x_field], df[y_field], s=dot_size, edgecolor='black', linewidth=dot_border, label=label, color=color)
    elif plot_type == 'line_plus_dots':
        ax.plot(df[x_field], df[y_field], linewidth=line_thickness, label=label, color=color)
        if color_field:
            unique_values = df[color_field].unique()
            colors = plt.cm.get_cmap('viridis', len(unique_values))
            color_map = {val: colors(i) for i, val in enumerate(unique_values)}
            df['color'] = df[color_field].map(color_map)
            for val in unique_values:
                subset = df[df[color_field] == val]
                ax.scatter(subset[x_field], subset[y_field], s=dot_size, edgecolor='black', linewidth=dot_border, label=val, c=subset['color'])
        else:
            ax.scatter(df[x_field], df[y_field], s=dot_size, edgecolor='black', linewidth=dot_border, label=label, color=color)
    elif plot_type == 'dash':
        ax.plot(df[x_field], df[y_field], linewidth=line_thickness, linestyle='--', label=label, color=color, dashes=dash_pattern)
    else:
        raise ValueError("Invalid plot_type. Choose from 'line', 'scatter', 'line_plus_dots', or 'dash'.")
    
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    if label or color_field:
        ax.legend()


def overlap_plot_one_cycle(df, x='Cycle Time (h)', y=['Voltage (V)'], plot_width=800, plot_height=300, plot_title=''):
    ''' 
    Generates an interactive plot of multiple features overlapping without grouping by cycle
    '''
    # Initialize the plotly figure
    fig = go.Figure()

    # Add a trace for each y-feature across the full dataset
    for feature in y:
        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[feature],
                mode='lines',
                name=feature
            )
        )

    # Customize the layout
    fig.update_layout(
        title=plot_title,
        width=plot_width,
        height=plot_height,
        legend_title='Features',
        xaxis_title=x,
        yaxis_title=', '.join(y),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True
        )
    )

    fig.show()





import holoviews as hv
from holoviews.operation.datashader import datashade

hv.extension('bokeh')

def interactive_datashader_plot(features_to_plot, x_feature, df):
    """ Fast plotting of large datasets with holoviews """
    plots = []
    for feature in features_to_plot:
        curve = hv.Curve(df, kdims=[x_feature], vdims=[feature])
        shaded = datashade(
            curve,
            cmap=["black"],      # fixed color
            
        ).opts(width=800, height=300)
        plots.append(shaded)
    return hv.Layout(plots).cols(1)



def interactive_datashader_plot_2(features_to_plot, x_feature, df):
    plots = []
    for feature in features_to_plot:
        curve = hv.Curve(df, kdims=[x_feature], vdims=[feature])
        shaded = datashade(
            curve,
            cmap=["black"]
        ).opts(width=800, height=300)
        plots.append(shaded)
    
    layout = hv.Layout(plots).cols(1)
    return layout.opts(shared_axes=True)  # link x-axes across plots

