
__all__ = ['cumulative_trapezoid',
 'get_c_rate',
 'get_capacity',
 'get_energy',
 'get_energy_params',
 'get_power',
 'get_power_params',
 'get_resistance_per_cycle',
 'get_slow_cycles']

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


def get_power(df, current_col='Current (A)', voltage_col='Voltage (V)', power_col='P (W)'):
    df[power_col] = abs(df[current_col] * df[voltage_col])
    return df

def get_capacity(df, current_col='Current (A)', time_col='Time (h)', capacity_col='Q (Ah)'):
    df = df.copy()
    df[capacity_col] = cumulative_trapezoid(df[current_col], df[time_col], initial=0)
    return df

def get_energy(df, power_col='P (W)', time_col='Time (h)', energy_col='E (Wh)'):
    df = df.copy()
    df[energy_col] = abs(cumulative_trapezoid(df[power_col], df[time_col], initial=0))
    return df

def get_power_params(df):
    p_max = df['P (W)'].abs().max()
    return p_max

def get_energy_params(df_cycle_cha, df_cycle_dis, en_eff_ini=1):
    e_cha = df_cycle_cha['E (Wh)'].max()
    e_dis = df_cycle_dis['E (Wh)'].max()
    en_eff = 100 * e_dis / e_cha if e_cha != 0 else 0
    en_eff_fade = (1 - en_eff / en_eff_ini) * 100 if en_eff_ini != 0 else 0
    return en_eff, en_eff_fade

def get_resistance_per_cycle(df, peak_target=100, peak_margin=1, zero_target=0, zero_margin=0.1):
    """
    Applies peak and resistance calculation to each cycle in 'Cycle Number'.
    Returns a dataframe with dI, dU, and R (in mOhm).
    """
    results = []  # Store results for each cycle

    for cycle, cycle_df in df.groupby('Cycle Number'):
        
        # Find peak occurrence
        peak_df = cycle_df[(cycle_df['Current (A)'] >= peak_target - peak_margin) & 
                           (cycle_df['Current (A)'] <= peak_target + peak_margin)]
        peak_index = peak_df.index.min()  

        # Find peak current and voltage
        if pd.notna(peak_index):
            peak_current = cycle_df.loc[peak_index, 'Current (A)']
            peak_voltage = cycle_df.loc[peak_index, 'Voltage (V)']
        else:
            continue  # Skip this cycle if no peak found

        # Find last near-zero occurrence before peak
        zero_df = cycle_df.loc[:peak_index - 1]  # Filter only rows before peak
        zero_df = zero_df[(zero_df['Current (A)'] >= zero_target - zero_margin) & 
                          (zero_df['Current (A)'] <= zero_target + zero_margin)]
        last_zero_index = zero_df.index.max()  

        if pd.notna(last_zero_index):
            zero_current = cycle_df.loc[last_zero_index, 'Current (A)']
            zero_voltage = cycle_df.loc[last_zero_index, 'Voltage (V)']
        else:
            continue  # Skip this cycle if no near-zero found before peak

        # Calculate amplitude of jumps
        current_jump = abs(peak_current - zero_current)
        voltage_jump = abs(peak_voltage - zero_voltage)

        # Calculate resistance using Ohm's law (R = ΔV / ΔI) and convert to mOhm
        resistance_mOhm = (voltage_jump / current_jump) * 1000 if current_jump != 0 else None

        # Append results
        # results.append({'Cycle Number': cycle, 'dI (A)': current_jump, 'dU (V)': voltage_jump, 'Ri (mOhm)': round(resistance_mOhm, 3)})
        results.append({'Cycle Number': cycle, 'Ri (mOhm)': round(resistance_mOhm, 3)})

        resistance_df = pd.DataFrame(results)

        resistance_df['Ri incr (%)'] = (resistance_df['Ri (mOhm)'] - resistance_df['Ri (mOhm)'].iloc[0]) / resistance_df['Ri (mOhm)'].iloc[0] * 100
        resistance_df['Ri incr (%)'] = resistance_df['Ri incr (%)'].round(2)

    return resistance_df

def get_c_rate(df, rated_capacity = 69.5, max_voltage= 4.199):

    mean_current = df['Current (A)'] [df['Voltage (V)'] < max_voltage ].abs().mean() 
    c_rate = mean_current / rated_capacity
    return c_rate


def get_slow_cycles(df, current_col='Current (A)', cn_col='Cycle Number', C = 78, tolerance=0.2, num_slow_cycles=3):
    # Select slow cycles
    
    target_current = - C / 3  # ( only discharge )
    tolerance = 0.2

    # Filter for negative current (discharge) and within tolerance
    filtered_df = df[ (df['Current (A)'] < 0) &  (np.isclose(df['Current (A)'], target_current, atol=tolerance))]

    # Get the unique cycle numbers of these filtered rows
    slow_cycles = filtered_df['Cycle Number'].unique()
    selected_slow_cycles = slow_cycles[num_slow_cycles-1::num_slow_cycles] # select every third slow cycle starting from the 3rd one
    print(selected_slow_cycles)

    return selected_slow_cycles