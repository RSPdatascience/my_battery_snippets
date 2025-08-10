import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import joblib

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential # type: ignore
from keras.layers import InputLayer, GRU, Dense, Masking # Bidirectional, Dropout, BatchNormalization, Flatten # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data(file_name, input_features, output_features):
    ''' Reads data from an excel file and genrates an input and an output dataframe 
    acording to the specified input_features and output_features'''
    joblib.dump(input_features, 'stored_input_features.pkl')
    joblib.dump(output_features, 'stored_output_features.pkl')

    raw_df = pd.read_excel(file_name)

    in_df = raw_df[input_features]
    out_df = raw_df[output_features]

    cycle_num_df = raw_df['Cycle number']

    return in_df, out_df, cycle_num_df


# Generate scaled in and out df-s
from sklearn.preprocessing import MinMaxScaler # type: ignore

def generate_scaled_in_out_dfs(df, input_features,output_features):
    ''' Generate scaled in and out dataframes'''
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    scaled_input_data = input_scaler.fit_transform(df[input_features])
    scaled_output_data = output_scaler.fit_transform(df[output_features])

    joblib.dump(input_scaler, 'input_scaler.pkl')
    joblib.dump(output_scaler, 'output_scaler.pkl')  

    in_df = pd.DataFrame(scaled_input_data, columns = input_features)
    out_df = pd.DataFrame(scaled_output_data, columns = output_features)

    return in_df, out_df


def determine_max_sequence_length(df):
    '''Determine the maximum sequence length, requires cycle_number field'''
    max_len = 0
    for cycle_number in df['Cycle number'].unique():
        cycle_data = df[df['Cycle number'] == cycle_number]
        max_len = max(max_len, len(cycle_data))
    return max_len





def scale_data(in_df, out_df, load_scaler = False):
    '''
    Convert in and out df-s into numpy arrays. 
    Scalers are dumped for later use in testing and validation
    unless load_scaler = True in which casethe stored scalers are loaded
    '''

    input_data = in_df.to_numpy()
    output_data = out_df.to_numpy()

    if load_scaler:

        input_scaler = joblib.load('input_scaler.pkl')
        output_scaler = joblib.load('input_scaler.pkl')

        scaled_input_data = input_scaler.fit_transform(input_data)
        scaled_output_data = output_scaler.fit_transform(output_data)

    else:
        input_scaler = MinMaxScaler()
        output_scaler = MinMaxScaler()

        scaled_input_data = input_scaler.fit_transform(input_data)
        scaled_output_data = output_scaler.fit_transform(output_data)

        joblib.dump(input_scaler, 'input_scaler.pkl')
        joblib.dump(output_scaler, 'output_scaler.pkl')  
    
    return scaled_input_data, scaled_output_data


from keras.preprocessing.sequence import pad_sequences # type: ignore

def generate_padded_sequences(df, input_features, output_features, max_cycle_size, fragment_step, padding_value=-9999.0):
    ''' Generates padded sequences with increasing fragment sizes from fragment_step to max_cycle_size. '''

    X, y = [], []
    
    for cycle_number in df['Cycle number'].unique():
        cycle_data = df[df['Cycle number'] == cycle_number]
        cycle_input = cycle_data[input_features].to_numpy()  # Convert to NumPy array
        cycle_output = cycle_data[output_features].to_numpy()  # Convert to NumPy array
        
        # Generate fragments of varying sizes
        for size in range(fragment_step, max_cycle_size + 1, fragment_step):
            fragment_input = cycle_input[:size]
            fragment_output = cycle_output[:size]
            
            # Append the fragments to the lists
            X.append(fragment_input)
            y.append(fragment_output)
    
    # Pad sequences using the padding value
    X_padded = pad_sequences(X, maxlen=max_cycle_size, padding='post', dtype='float32', value=padding_value)
    y_padded = pad_sequences(y, maxlen=max_cycle_size, padding='post', dtype='float32', value=padding_value)
    
    return X_padded, y_padded



def generate_sequences(scaled_input_data, scaled_output_data, window_size=20, overlap=0.5, shift=0):
    """
    Generate numpy array X and y sequences using sliding windows with adjustable size, overlap, and shift (displacement between in and out sequences).
    """
    assert 0 <= overlap <= 1, "Overlap should be between 0 and 1."
    
    # Define the step size based on overlap
    step_size = int(window_size * (1 - overlap)) if overlap > 0 else window_size
    
    X_sequences = []
    y_sequences = []

    # Generate sliding windows for input (X) and output (y)
    for i in range(0, len(scaled_input_data) - window_size - shift + 1, step_size):
        # Input window
        X_window = scaled_input_data[i:i + window_size]
        # Output window (shifted by 'shift' steps)
        y_window = scaled_output_data[i + shift:i + window_size + shift]
        
        X_sequences.append(X_window)
        y_sequences.append(y_window)
    
    # Convert lists to numpy arrays
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    return X_sequences, y_sequences





def build_model(X_train, y_train):
    ''' Build the GRU model with masking for the padded values  '''
    
    max_cycle_size = X_train.shape[1]
    num_input_features = X_train.shape[2]
    num_output_features = y_train.shape[2]

    model = Sequential()

    model.add(InputLayer(shape=(max_cycle_size, num_input_features)))
    model.add(Masking(mask_value=-9999.0)) 

    model.add(GRU(64*4, return_sequences=True)) 
    model.add(GRU(64*3, return_sequences=True)) 
    model.add(GRU(64*2, return_sequences=True)) 

    model.add(Dense(64*3, activation='relu'))  
    model.add(Dense(64*2, activation='relu'))
    model.add(Dense(64*1, activation='relu'))  
    model.add(Dense(num_output_features, activation='linear')) 

    model.compile(optimizer='adam', loss='mse')

    return model


def plot_loss(history):
    ''' Plots the training and validation loss based on the model history'''

    # Get the best loss and validation loss
    best_loss = min(history.history['loss'])
    best_val_loss = min(history.history['val_loss'])

    # Plot the loss function
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Set the y-axis to a logarithmic scale
    plt.legend()
    plt.title(f"Best Loss: {best_loss:.3e}\nBest Validation Loss: {best_val_loss:.3e}")
    plt.show()


def get_predictions(model, input_scaler, output_scaler, y_data, X_data, padding_value=-9999.0):
    ''' Generates predictions for U and T vs cycle time. The differences are also included'''

    # Reshape and convert y_data to DataFrame
    y_reshaped = y_data.reshape(-1, 2)  # Reshape to 2D
    y_reshaped_df = pd.DataFrame(y_reshaped)  # Convert to DataFrame

    # Find the indexes of the padded rows
    padding_indexes = y_reshaped_df[y_reshaped_df[0] == padding_value].index

    # Remove the padded rows
    unpadded_y_df = y_reshaped_df[~y_reshaped_df.index.isin(padding_indexes)].reset_index(drop=True)
    unpadded_y_df.columns = ['U', 'T']

    # Inverse transform y_data
    y_unscaled = output_scaler.inverse_transform(unpadded_y_df)

    # Predicting from X_data
    y_pred = model.predict(X_data)

    # Reshape and convert predictions to DataFrame
    y_pred_reshaped = y_pred.reshape(-1, 2) 

    # Inverse transform predicted data
    y_pred_reshaped = output_scaler.inverse_transform(y_pred_reshaped)

    y_pred_reshaped_df = pd.DataFrame(y_pred_reshaped)  # Convert to DataFrame

    # Remove the padded rows from predictions
    unpadded_y_pred_df = y_pred_reshaped_df[~y_pred_reshaped_df.index.isin(padding_indexes)].reset_index(drop=True)
    unpadded_y_pred_df.columns = ['U_pred', 'T_pred']

    # Combine the unpadded DataFrames
    comparison_df = pd.concat([pd.DataFrame(y_unscaled, columns=['U', 'T']), unpadded_y_pred_df], axis=1)

    # Add columns with differential values (how much measured values are above predicted ones)
    comparison_df['dif_U'] = comparison_df['U'] - comparison_df['U_pred']
    comparison_df['dif_T'] = comparison_df['T'] - comparison_df['T_pred']

    # Unpad the X data
    X_test_fragment = X_data[X_data != -9999.0].reshape(-1, X_data.shape[-1])

    # Rescale the X data
    X_test_fragment_rescaled = input_scaler.inverse_transform(X_test_fragment.reshape(-1, X_test_fragment.shape[-1]))

    # Convert the X_data to df
    time_fragment_df = pd.DataFrame(X_test_fragment_rescaled, columns=['Current [A]', 'Cycle time [h]',	'SoH'])

    # Convert hours to minutes
    time_fragment_df['Cycle time [h]'] = time_fragment_df['Cycle time [h]'] * 60 
    time_fragment_df.rename(columns={'Cycle time [h]': 'Cycle time [min]'}, inplace=True)


      
    # Combine the time fragment with the comparison DataFrame
    comparison_df = pd.concat([time_fragment_df, comparison_df], axis=1)


    return comparison_df





from sklearn.metrics import mean_absolute_error, mean_squared_error


def error_metrics(var_name, actual, predicted):
    ''' Return error metrics dataframe line for actual and predicted arrays'''
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    data = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }

    error_df = pd.DataFrame(data, index = [var_name])
    return error_df



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



def plot_features(features_to_plot, x_feature, df, plot_width=800, plot_height=400, mode='lines'):
    '''This function creates an interactive plot of specified size 
    of specified 'features_to_plot' vs specified x_feature.
    Select mode='lines', mode='markers' or mode='lines+markers'.
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
        
        # Determine x values based on whether x_feature is the index
        x_values = df.index if x_feature == 'index' else df[x_feature]
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df[feature],
            mode=mode,
            name=f'Actual {feature}',
            line=dict(color=color),
            hovertemplate=f'Cycle Number: %{{text}}{x_feature}: %{{x}}{feature}: %{{y}}',
            text=df['Cycle Number']  # Add the cycle number to the hover text
        ), row=i+1, col=1)
        
        # Update y-axis title for each subplot
        fig.update_yaxes(title_text=feature, row=i+1, col=1)
        
        # Update x-axis title for each subplot
        fig.update_xaxes(title_text=x_feature if x_feature != 'index' else 'Index', row=i+1, col=1)
    
    fig.update_layout(
        showlegend=False,  # Hide the legend
        width=plot_width,  # Set the width of the plot
        height=plot_height * len(features_to_plot)  # Adjust height based on the number of subplots
    )
    
    # Show the plot
    fig.show()



def overlap_plot(df, feature='Voltage (V)', cycles_to_hide=1):
    ''' 
    This function plots a feature for all cycles except for the last one and a specified number of the first ones.
    '''

    # Get unique cycle numbers sorted in ascending order
    cycle_numbers = sorted(df['Cycle Number'].unique())

    # Define cycles to exclude: first n cycles and the last one
    cycles_to_exclude = cycle_numbers[:cycles_to_hide] + [cycle_numbers[-1]]

    # Filter the dataframe to exclude the specified cycles
    filtered_df = df[~df['Cycle Number'].isin(cycles_to_exclude)]

    # Initialize the plotly figure
    fig = go.Figure()

    # Loop through each cycle and add traces
    for cycle, data in filtered_df.groupby('Cycle Number'):
        fig.add_trace(
            go.Scatter(
                x=data['Cycle Time(h)'],
                y=data[feature],
                mode='lines',
                name=f'Cycle {cycle}'
            )
        )

    # Customize the layout
    fig.update_layout(
        title='Cycles Overlapped',
        xaxis_title='Cycle Time (h)',
        yaxis_title=feature,
        legend_title='Cycles',
        width=900,  # Adjust width as needed
        height=500  # Adjust height as needed
    )

    fig.show()




def gen_padded_X_y(df, max_cycle_size, input_features, output_features):       

    X_unpaded, y_unpaded = [], []
    for cycle_number in df['Cycle number'].unique():
        
        # Get the data for each single cycle number
        single_cycle_data = df[df['Cycle number'] == cycle_number]

        # Extract input and output features for this cycle
        X_cycle = single_cycle_data[input_features].values
        y_cycle = single_cycle_data[output_features].values

        # Append scaled cycle data to lists
        X_unpaded.append(X_cycle)
        y_unpaded.append(y_cycle)

    # Pad sequences using the padding value
    X = pad_sequences(X_unpaded, maxlen=max_cycle_size, padding='post', dtype='float32', value=-9999.0)
    y = pad_sequences(y_unpaded, maxlen=max_cycle_size, padding='post', dtype='float32', value=-9999.0)

    return X, y


import pandas as pd

def gen_predicted_actual_dfs(model, X_test, y_test, output_features, y_scaler):

    predictions_scaled = model.predict(X_test)

    # Convert y_test and predictions_scaled to DataFrames
    y_test_df = pd.DataFrame(y_test.reshape(-1, len(output_features)))
    predictions_df = pd.DataFrame(predictions_scaled.reshape(-1, len(output_features)))

    # Identify the indices of the padded rows
    padded_indices = y_test_df[y_test_df[0] == -9999.0].index

    # Remove the padded rows from both DataFrames
    y_test_unpadded_df = y_test_df.drop(padded_indices)
    predictions_unpadded_df = predictions_df.drop(padded_indices)

    # Add the column names
    predictions_unpadded_df.columns = output_features
    y_test_unpadded_df.columns = output_features

    # Reset the indexes
    y_test_unpadded_df = y_test_unpadded_df.reset_index(drop=True)
    predictions_unpadded_df = predictions_unpadded_df.reset_index(drop=True)

    # Apply inverse transform to unpadded DataFrames
    y_test_unpadded_df = pd.DataFrame(y_scaler.inverse_transform(y_test_unpadded_df), columns=output_features)
    predictions_unpadded_df = pd.DataFrame(y_scaler.inverse_transform(predictions_unpadded_df), columns=output_features)

    return y_test_unpadded_df, predictions_unpadded_df