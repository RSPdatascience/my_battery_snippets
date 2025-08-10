


import pandas as pd
import numpy as np


def get_discharge_capacity(df):
    ''' Calculates the discharge capacity by by integrating the absolute value of current over time using trapezoidal rule'''

    # Initialize the Discharge_Capacity(A.h) column with zeros
    df['DC (A.h)'] = 0.0

    # Calculate the discharge capacity for each cycle by integrating the absolute value of current over time
    for cycle in df['Cycle number'].unique():
        cycle_df = df[(df['Cycle number'] == cycle) & (df['Step_Index'] == 7)]
        if not cycle_df.empty:
            # Calculate the discharge capacity using the trapezoidal rule
            discharge_capacity = np.trapz(cycle_df['I (A)'].abs(), cycle_df['Cycle time [h]'])
            df.loc[cycle_df.index, 'DC (A.h)'] = discharge_capacity
    
    return df



def max_dc(df):
    ''' Calculate the maximum discharge capacity for each cycle '''
    max_discharge_capacity = df.groupby('Cycle number')['DC (A.h)'].max().reset_index()
    max_discharge_capacity.rename(columns={'DC (A.h)': 'Capacity [Ah]'}, inplace=True)
    
    # Merge the max discharge capacity back to the original dataframe
    df = pd.merge(df, max_discharge_capacity, on='Cycle number', how='left')
    
   
    return df


# Function to add cumulative capacity
def add_cumulative_capacity(df, time_field, current_field):
    df = df.sort_values(by=time_field).reset_index(drop=True)
    cumulative_capacity = np.zeros(len(df))
    for i in range(1, len(df)):
        cumulative_capacity[i] = np.trapz(df[current_field].values[:i+1], x=df[time_field].values[:i+1])
    df['Capacity (Ah)'] = abs(cumulative_capacity)
    return df





def calculate_soc(df, nominal_capacity = 69.5):
    ''' Calculates cumulative charge and SoC for a cycle, starting from SoC = 0 '''
    
    # Compute time differences in hours (Δt)
    df['Δt'] = df['Cycle Time (h)'].diff().fillna(0)

    # Compute charge variation (ΔQ = I * Δt)
    df['ΔQ'] = df['Current (A)'] * df['Δt']

    # Integrate charge over time
    df['Cumulative Charge (Ah)'] = df['ΔQ'].cumsum()

    # Compute SoC (starting at 0)
    df['SoC'] = df['Q'] / nominal_capacity

    # Clip SoC between 0 and 1
    df['SoC'] = df['SoC'].clip(0, 1)

    df = df.drop(['Δt', 'ΔQ'], axis=1)
    return df



def get_soc_now(df, charge_col='Capacity (Ah)',soc_now_col='SoCnow'):
    df = df.copy()

    # Safe normalization for SoC within each cycle
    def safe_soc(x):
        delta = x.max() - x.min()
        return (x - x.min()) / delta if delta != 0 else pd.Series(0, index=x.index)

    # Apply SoC calculation
    df[soc_now_col] = df[charge_col].transform(safe_soc)

    return df



def get_real_soh(df, capacity_column='Capacity (Ah)',nominal_capacity=69.5):
    # Determine real SoH for comparison
    real_soh = df[capacity_column].max() / nominal_capacity  
    return real_soh