
import numpy as np



def smooth_and_compute_derivatives(df, x_col='Cycle Time (h)', u_col='Voltage (V)', q_col='Capacity (Ah)', new_col='dQ/dU', window_ratio=0.07, polyorder=2):
                                   
    """
    Apply SG filtering and compute dQ/dU from time-series voltage and capacity data.
    
    Returns:
    - df with added columns: 'Ufit (V)', 'Qfit (Ah)', 'dQ/dU'
    """

    df = df.copy()
    n = len(df)
    window_length = max(5, int(window_ratio * n) | 1)
    if window_length <= polyorder:
        window_length = polyorder + 2 + (polyorder + 2) % 2

    dt = np.mean(np.diff(df[x_col]))

    # Smooth voltage and capacity
    df['Ufit (V)'] = savgol_filter(df[u_col], window_length, polyorder)
    df['Qfit (Ah)'] = savgol_filter(df[q_col], window_length, polyorder)

    # Derivatives
    dQ_dt = savgol_filter(df[q_col], window_length, polyorder, deriv=1, delta=dt)
    dU_dt = savgol_filter(df[u_col], window_length, polyorder, deriv=1, delta=dt)

    # Chain rule
    df[new_col] = abs( np.where(np.abs(dU_dt) > 1e-5, dQ_dt / dU_dt, np.nan) )
    df['dU/dQ'] = abs( np.where(np.abs(dU_dt) > 1e-5, dU_dt / dQ_dt, np.nan) )

    return df


def replace_outliers(df, column, threshold_percentage = 5):
    ''' Removes outliers by comparing each value with previous and next one. 
        If the difference is above specified percentage the value us substituted with previous one'''
    for i in range(1, len(df)-1):
        previous_value = df.at[i-1, column]
       
        next_value = df.at[i+1, column]
        avg_value = (previous_value + next_value) / 2

        current_value = df.at[i, column]

        difference_percentage = abs((current_value - avg_value) / avg_value) * 100
        
        if difference_percentage > threshold_percentage:
            df.at[i, column] = previous_value
        
    return df


import scipy.interpolate as interp

def generate_spline(df, field='Cycle Time (h)', target_field='Voltage (V)', spline_column_name='Uspl (V)', groupby_field='Cycle Number', s=0.0005, ):
    # Ensure we are working with a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Determine the name of the spline column
    if spline_column_name is None:
        spline_column_name = f'{target_field}fit'
    
    # Initialize a column for fitted values
    df[spline_column_name] = np.nan
    
    # Fit each discharge segment separately, grouped by the specified field
    for cn, group in df.groupby(groupby_field):
        time_segment = group[field].values
        target_segment = group[target_field].values
        
        # Fit a smoothing spline (adjust 's' for smoothness)
        spline = interp.UnivariateSpline(time_segment, target_segment, s=s)
        
        # Store fitted values only in the corresponding step
        df.loc[group.index, spline_column_name] = spline(time_segment)
    
    return df


from scipy.stats import median_abs_deviation

def hampel_filter(df, column, window_size=10, n_sigma=0.1):
    """
    Apply Hampel filter to a DataFrame column to remove outliers.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the derivative data.
        column (str): The name of the column with derivative values.
        window_size (int): The number of points to consider on each side of the current point.
        n_sigma (float): The threshold for outlier detection (in units of MAD).
    
    Returns:
        pd.Series: The filtered column with outliers replaced by local median.
    """
    filtered = df[column].copy()
    n = len(filtered)
    
    for i in range(window_size, n - window_size):
        window = filtered.iloc[i - window_size:i + window_size + 1]
        median = np.median(window)
        mad = median_abs_deviation(window)
        threshold = n_sigma * mad
        
        if abs(filtered.iloc[i] - median) > threshold:
            filtered.iloc[i] = median  # Replace outlier with median
    
    print('filter works')
    
    return filtered





def hampel_filter(df, col_name, window_size=31, n_sigma=3):
    """
    Apply Hampel filter to remove outliers from a specified column in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col_name (str): The name of the column to filter.
    window_size (int): The size of the window (number of observations) to use for the filter.
    n_sigma (int): The number of standard deviations to use as the threshold for identifying outliers.
    
    Returns:
    pd.DataFrame: The DataFrame with outliers removed from the specified column.
    """
    # Calculate rolling median and rolling standard deviation
    rolling_median = df[col_name].rolling(window=window_size, center=True).median()
    rolling_std = df[col_name].rolling(window=window_size, center=True).std()
    
    # Identify outliers
    diff = np.abs(df[col_name] - rolling_median)
    outlier_idx = diff > (n_sigma * rolling_std)
    
    # Replace outliers with rolling median
    df.loc[outlier_idx, col_name] = rolling_median[outlier_idx]
    
    return df



from scipy.signal import savgol_filter

def savgol_smoothing(df, field, window_size=5, poly_order=2):
    """
    Applies Savitzky-Golay smoothing to a specified column in a DataFrame
    and replaces the original column with the smoothed values.
    """
    # Ensure window size is valid
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")
    if window_size > len(df[field]):
        raise ValueError("Window size must not be larger than the number of data points.")
    if window_size <= poly_order:
        raise ValueError("Window size must be greater than polynomial order.")

    df_copy = df.copy()  # Avoid modifying the original DataFrame
    df_copy[field] = savgol_filter(df_copy[field], window_length=window_size, polyorder=poly_order)

    return df_copy