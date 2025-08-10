from scipy.signal import find_peaks
import numpy as np

__all__ = ["find_strongest_peak"]

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