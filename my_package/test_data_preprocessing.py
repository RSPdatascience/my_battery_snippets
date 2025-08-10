

import pandas as pd
import numpy as np


def active_time_from_datetime(df):


    # Calculate total experiment time

    df['Date_Time'] = pd.to_datetime(df['Date_Time'])

    # Calculate the time difference in hours between consecutive rows
    df['Time_diff'] = df['Date_Time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds() / 3600  # convert to hours

    # Reset the time difference where there is a gap (for example, if the gap is more than 1 hour)
    # You can adjust the threshold based on what constitutes a "gap"
    df['Time_diff'] = df['Time_diff'].where(df['Time_diff'] <= 1, 0)

    # Accumulate the time differences to create a new active time column
    df['t (h)'] = df['Time_diff'].cumsum()

    df.drop(columns=['Time_diff'], inplace=True)

    # Drop the first row where there's no difference
    df.dropna(inplace=True)


    # Place the new column in the beginning
    df.insert(2, 't (h)', df.pop('t (h)'))

    return df




def cycle_number_from_index(df):
    ''' Obtain the cycle number'''
    cumulative_index = []
    current_cumulative = 0

    # Check if 'Cycle_Index' column exists
    if 'Cycle_Index' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Cycle_Index' column.")

    # Loop through the Cycle_Index column to create the cumulative index
    for i in range(len(df)):
        if i == 0:
            # First entry, set the initial cumulative index
            current_cumulative = 1
        else:
            # Check if Cycle_Index has increased or reset to 1
            if df.loc[i, 'Cycle_Index'] == 1 and df.loc[i-1, 'Cycle_Index'] != 1:
                current_cumulative += 1
            elif df.loc[i, 'Cycle_Index'] > df.loc[i-1, 'Cycle_Index']:
                current_cumulative += 1
        
        cumulative_index.append(current_cumulative)

    # Add the cycle number as a new column
    df.insert(4, 'Cycle number', cumulative_index)

    return df


def get_cycle_time(df):
    ''' Calculate cycle time from Cycle number and Time (h)'''

    df['Cycle time [h]'] = 0.0

    # Iterate through the dataframe and update Cycle time(h)
    cycle_time = 0.0
    for i in range(1, len(df)):
        if df.loc[i, 'Cycle number'] != df.loc[i-1, 'Cycle number']:
            cycle_time = 0.0
        else:
            cycle_time += df.loc[i, 't (h)'] - df.loc[i-1, 't (h)']
        df.loc[i, 'Cycle time [h]'] = cycle_time

    # Move Cycle_Time(h) to the beginning of the DataFrame
    df.insert(2, 'Cycle time [h]', df.pop('Cycle time [h]'))

    return df


def downsample(df,n=4):
    # Filter rows where Step_Index is 2 or 7
    step_index_2_or_7 = df[df['Step_Index'].isin([2, 7])]

    # Keep only every n-th row
    n=4
    step_index_2_or_7_reduced = step_index_2_or_7.iloc[::n]

    # Combine the reduced rows with the rest of the dataframe
    df_reduced = pd.concat([df[~df['Step_Index'].isin([2, 7])], step_index_2_or_7_reduced]).sort_index()

    return df_reduced

def get_cycle_time(df):
    ''' Calculate cycle time from Cycle number and Time (h)'''

    df['Cycle time [h]'] = 0.0

    # Iterate through the dataframe and update Cycle time(h)
    cycle_time = 0.0
    for i in range(1, len(df)):
        if df.loc[i, 'Cycle number'] != df.loc[i-1, 'Cycle number']:
            cycle_time = 0.0
        else:
            cycle_time += df.loc[i, 't (h)'] - df.loc[i-1, 't (h)']
        df.loc[i, 'Cycle time [h]'] = cycle_time

    # Move Cycle_Time(h) to the beginning of the DataFrame
    df.insert(2, 'Cycle time [h]', df.pop('Cycle time [h]'))

    return df






def hours_from_datetime (df):

    df['Date_Time'] = pd.to_datetime(df['Date_Time'])

    df.insert(1, 't (h)', (df['Date_Time'] - df['Date_Time'].iloc[0]).dt.total_seconds() / 3600)

    return df


def active_time_from_datetime(df):


    # Calculate total experiment time

    df['Date_Time'] = pd.to_datetime(df['Date_Time'])

    # Calculate the time difference in hours between consecutive rows
    df['Time_diff'] = df['Date_Time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds() / 3600  # convert to hours

    # Reset the time difference where there is a gap (for example, if the gap is more than 1 hour)
    # You can adjust the threshold based on what constitutes a "gap"
    df['Time_diff'] = df['Time_diff'].where(df['Time_diff'] <= 1, 0)

    # Accumulate the time differences to create a new active time column
    df['t (h)'] = df['Time_diff'].cumsum()

    df.drop(columns=['Time_diff'], inplace=True)

    # Drop the first row where there's no difference
    df.dropna(inplace=True)


    # Place the new column in the beginning
    df.insert(2, 't (h)', df.pop('t (h)'))

    return df




def cycle_number_from_index(df):
    ''' Obtain the cycle number'''
    cumulative_index = []
    current_cumulative = 0

    # Check if 'Cycle_Index' column exists
    if 'Cycle_Index' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Cycle_Index' column.")

    # Loop through the Cycle_Index column to create the cumulative index
    for i in range(len(df)):
        if i == 0:
            # First entry, set the initial cumulative index
            current_cumulative = 1
        else:
            # Check if Cycle_Index has increased or reset to 1
            if df.loc[i, 'Cycle_Index'] == 1 and df.loc[i-1, 'Cycle_Index'] != 1:
                current_cumulative += 1
            elif df.loc[i, 'Cycle_Index'] > df.loc[i-1, 'Cycle_Index']:
                current_cumulative += 1
        
        cumulative_index.append(current_cumulative)

    # Add the cycle number as a new column
    df.insert(4, 'Cycle number', cumulative_index)

    return df


def get_cycle_time(df):
    ''' Calculate cycle time from Cycle number and Time (h)'''

    df['Cycle time [h]'] = 0.0

    # Iterate through the dataframe and update Cycle time(h)
    cycle_time = 0.0
    for i in range(1, len(df)):
        if df.loc[i, 'Cycle number'] != df.loc[i-1, 'Cycle number']:
            cycle_time = 0.0
        else:
            cycle_time += df.loc[i, 't (h)'] - df.loc[i-1, 't (h)']
        df.loc[i, 'Cycle time [h]'] = cycle_time

    # Move Cycle_Time(h) to the beginning of the DataFrame
    df.insert(2, 'Cycle time [h]', df.pop('Cycle time [h]'))

    return df


def remove_short_cycles(df,threshold_percentage = 5):
    ''' Removes every cycle that is more than 5 % shorter than previous one '''
    # Find the maximum Cycle_Time for each cycle
    max_cycle_time = df.groupby('Cycle number')['Cycle time [h]'].max().reset_index()

    # Find cycles where the maximum Cycle_Time is more than 10% smaller than the previous cycle
    fraction = (100-threshold_percentage)/100
    bad_cycles1 = max_cycle_time[(max_cycle_time['Cycle time [h]'] < max_cycle_time['Cycle time [h]'].shift(1) * fraction )]

    # Merge with the original dataframe to get all data for the good cycles
    result_df = df[~df['Cycle number'].isin(bad_cycles1['Cycle number'])]

    return result_df




def cycle_data_gen(df, nom_capacity = 69.5, take_C_from_slow_cycles=True):

    '''Generate a dataframe with ΔU, Ri, C and SoH for either slow or fast cycles'''

    # Select the last voltage of each cycle's step 11
    df_before = df[df['Step Index'] == 11].groupby('Cycle Number')['Voltage (V)'].last().reset_index()
    df_before.rename(columns={'Voltage (V)': 'Voltage down (V)'}, inplace=True)

    # Select the n-th value of the voltage pulse
    n = 40 

    # Group by Cycle Number and get the n-th voltage value for each group where Step Index is 12
    df_after = df[df['Step Index'] == 12].groupby('Cycle Number')['Voltage (V)'].nth(n-1).reset_index()
    df_after.rename(columns={'Voltage (V)': 'Voltage up (V)'}, inplace=True)

    # Combine the DataFrames using the index
    df_cycle_data = pd.merge(df_before, df_after, left_index=True, right_index=True, how='left')

    #Remove the index column remaining from df_after
    df_cycle_data = df_cycle_data.drop(columns=['index'])

    # Calculate Voltage difference
    df_cycle_data['Voltage dif (V)'] = df_cycle_data['Voltage up (V)'] - df_cycle_data['Voltage down (V)']

    # Calculate the internal resistance
    df_cycle_data['Ri (mOhm)'] = df_cycle_data['Voltage dif (V)'] * 1000 / 100  # current difference is 100 A

    if take_C_from_slow_cycles==True:
        # Get the discharge capacity for each segment with ['Step Index'] == 21 corresponding to every third slow cycle
        df_dc = df[df['Step Index'] == 21].groupby('Cycle Number')['Discharge Capacity (Ah)'].max().iloc[2::3].reset_index()
        
    else:
        # Get the discharge capacty from every cycle from step 15
        df_dc = df[df['Step Index'] == 15].groupby('Cycle Number')['Discharge Capacity (Ah)'].max().reset_index()

    # Add the capacity column to the df
    df_cycle_data = pd.merge(df_cycle_data, df_dc, on = 'Cycle Number', how='outer')

    # Rename the discharge capacity column
    df_cycle_data.rename(columns={'Discharge Capacity (Ah)': 'DC (Ah)'}, inplace=True)


    # Add SoH column
    df_cycle_data['SoH'] = df_cycle_data['DC (Ah)'] / nom_capacity    

    # Calculate cycle duration
    cycle_max_time = df.groupby('Cycle Number')['Cycle Time (h)'].max().reset_index()

    # Rename the column
    cycle_max_time.rename(columns={'Cycle Time (h)': 'Cycle Duration (h)'}, inplace=True)

    df_cycle_data['Cycle Duration (h)'] = cycle_max_time['Cycle Duration (h)']

    # Move to the beginning of the df
    df_cycle_data.insert(1, 'Cycle Duration (h)', df_cycle_data.pop('Cycle Duration (h)'))

    # Add Max temperature per cycle
    df_cycle_data['Tmax (ºC)'] = df.groupby('Cycle Number')['Temperature (ºC)'].max()
    df_cycle_data['Tmin (ºC)'] = df.groupby('Cycle Number')['Temperature (ºC)'].min()
    df_cycle_data['Tavg (ºC)'] = df.groupby('Cycle Number')['Temperature (ºC)'].mean()
  
    return df_cycle_data



def select_slow_cycles(df, slow_cycle_steps=[19, 20, 21]):
    ''' Selects only slow cycles with the specified step indexes '''

    grouped = df.groupby('Cycle Number')['Step Index'].unique()
    valid_cycles = grouped[grouped.apply(lambda x: set(slow_cycle_steps).issubset(x))].index
    df = df[df['Cycle Number'].isin(valid_cycles)]
    
    print('Slow cycles: ', df['Cycle Number'].unique())

    return df




def select_every_nth_cycle(df, n = 3):

    cycles_to_select = df['Cycle Number'].unique()[n-1::n]

    print('Cycles selected: ', cycles_to_select)

    df = df[ df['Cycle Number'].isin(cycles_to_select) ]

    return df





def remove_end_n_points(df, cycle_col='Cycle Number', n = 2):
    def filter_group(group):
        return group.iloc[n:-n]  # Remove the first and last row

    df = df.groupby(cycle_col).apply(filter_group).reset_index(drop=True)
    return df


from scipy.signal import find_peaks

def find_strongest_peak(df, x_col, y_col):
    """
    Finds the strongest peak in the given dataframe columns and plots it.
   
    Returns:
    dict: A dictionary with peak position and amplitude.
    """
    x = df[x_col].values
    y = df[y_col].values
    
    # Find all peaks
    peaks, _ = find_peaks(y)
    
    if len(peaks) == 0:
        print("No peaks found.")
        return None
    
    # Find the strongest peak (highest amplitude)
    strongest_peak_idx = peaks[np.argmax(y[peaks])]
    peak_x = round(x[strongest_peak_idx],3)
    peak_y = round(y[strongest_peak_idx],3)
    
    
    return peak_x, peak_y

def slow_cycle_data(df_dis, df_cha, Qnom=69.5):
    ''' Generates a df with slow cycle data (maxima of the dQ/dV curves, CC, DC, SoH)
    based on dataframes containing filtered charge and discharge data'''
    data = []

    # Iterate over each unique cycle number
    for cn in df_dis['Cycle Number'].unique():
        # Filter the dataframe for the current cycle number
        single_cycle_df_dis = df_dis[df_dis['Cycle Number'] == cn]
        single_cycle_df_cha = df_cha[df_cha['Cycle Number'] == cn]

        # Find the strongest peak
        _, dqdv_max_dis = find_strongest_peak(single_cycle_df_dis, 'Voltage (V)','dQ/dU')
        _, dqdv_max_cha = find_strongest_peak(single_cycle_df_cha, 'Voltage (V)','dQ/dU')

        # Get the maximum capacity
        dc = single_cycle_df_dis['Capacity (Ah)'].max().round()
        cc = single_cycle_df_cha['Capacity (Ah)'].max().round()

        # Calculate SoH
        soh = round(dc / Qnom * 100, 2)

        # Append the data to the list
        data.append({
            'Cycle Number': int(cn),
            'dQ/dVmax dis (Ah/V)': dqdv_max_dis,
            'dQ/dVmax cha (Ah/V)': dqdv_max_cha,
            'CC (Ah)': cc,
            'DC (Ah)': dc,
            'SoH (%)': soh
        })

    # Create a dataframe from the data
    df_cycle_data = pd.DataFrame(data)

    return df_cycle_data


def unix_to_datetime(df, unix_time_col = 'Date_Time', datetime_col = 'Actual_date' ):
    '''Convert unix time into a standard date time'''

    df[datetime_col] = pd.to_datetime(df[unix_time_col].astype('int64') // 10_000_000, unit='s')

    df[datetime_col] = pd.to_datetime(df[datetime_col])

    df.insert(2, datetime_col, df.pop(datetime_col) )

    return df

def time_from_start(df, datetime_col='Actual_date', time_from_start_col='Time (h)'):
    '''Create a column with time from the beginning in hours'''
    if time_from_start_col not in df.columns:  # Fixing the column name check
        df.insert(1, time_from_start_col, (df[datetime_col] - df[datetime_col].iloc[0]).dt.total_seconds() / 3600)
    return df


# Function to add cumulative capacity
def add_cumulative_capacity(df, time_field, current_field):
    df = df.sort_values(by=time_field).reset_index(drop=True)
    cumulative_capacity = np.zeros(len(df))
    for i in range(1, len(df)):
        cumulative_capacity[i] = np.trapz(df[current_field].values[:i+1], x=df[time_field].values[:i+1])
    df['Capacity (Ah)'] = abs(cumulative_capacity)
    return df


def sos(dif, dzeta_100, dzeta_80, mode = 'over'):
    ''' Returns SoS for either 'under' or 'over' mode based on an aray or single value'''
    
    sos = 1 / (0.25 * ((dif - dzeta_100) / (dzeta_80 - dzeta_100))**2 + 1)

    if mode == 'over':  # positive deviation, e.g. overvoltage, overtemperature 
        return np.where(dif <= dzeta_100, 1, sos)  # SoS=1 if the difference measured-predicted is smaller than the warning threshold 
    
    else:  # e.g. undervoltage, SoH
        return np.where(dif >= dzeta_100, 1, sos) # SoS=1 if the difference measured-predicted is larger than the warning threshold 



def get_cn(df, current_col='Current (A)', cn_col='Cycle Number',
           start_transitions=[[0, 69.5], [0, 23], [-23, 69.5]],
           end_transitions=[[-69.5, 0], [-23, 0]],
           zero_thresh=0.02, tolerance=0.2):
    
    """ Get cycle number from curren transitions """

    df = df.copy()
    current = df[current_col].values

    # --- Detect cycle starts
    start_indices = []
    for from_val, to_val in start_transitions:
        if abs(from_val) < zero_thresh:
            mask = (np.abs(current[:-1]) < zero_thresh) & np.isclose(current[1:], to_val, atol=tolerance)
        else:
            mask = np.isclose(current[:-1], from_val, atol=tolerance) & np.isclose(current[1:], to_val, atol=tolerance)
        start_indices.append(np.where(mask)[0] + 1)
    
    starts = np.sort(np.unique(np.concatenate(start_indices)))

    # --- Detect cycle ends
    end_indices = []
    for from_val, to_val in end_transitions:
        if abs(to_val) < zero_thresh:
            mask = np.isclose(current[:-1], from_val, atol=tolerance) & (np.abs(current[1:]) < zero_thresh)
        else:
            mask = np.isclose(current[:-1], from_val, atol=tolerance) & np.isclose(current[1:], to_val, atol=tolerance)
        end_indices.append(np.where(mask)[0] + 1)
    
    ends = np.sort(np.unique(np.concatenate(end_indices)))

    # --- Pair starts and ends
    cycle_indices = []
    s = 0
    for start in starts:
        while s < len(ends) and ends[s] < start:
            s += 1
        if s < len(ends) and ends[s] > start:
            cycle_indices.append((start, ends[s]))
            s += 1

    # --- Assign cycle numbers
    df[cn_col] = np.nan
    for i, (start, end) in enumerate(cycle_indices):
        df.loc[start:end, cn_col] = i + 1

    # Forward fill within cycles
    df[cn_col] = df[cn_col].ffill()

    # Replace NaNs before first cycle with 0 and convert to int
    df[cn_col] = df[cn_col].fillna(0).astype(int)

    # Move the column to the first position
    col_data = df.pop(cn_col)
    df.insert(0, cn_col, col_data)

    return df