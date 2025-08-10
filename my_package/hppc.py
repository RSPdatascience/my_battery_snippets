
from scipy.optimize import curve_fit
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fit_relaxations_all_cycles(df, step_indices=[11, 13, 15, 17],
                               time_col='Time (min)', voltage_col='Voltage (V)',
                               current_col='Current (A)', soc_col='SoC',
                               duration=0.15, plot=True):
    """
    Fit 2RC model for multiple relaxation steps across all cycles.
    Returns DataFrame with results for each Cycle Number and Step Index.
    If plot=True, shows a compact grid of voltage + fit subplots.
    """

    def double_exponential(t, A1, tau1, A2, tau2, V_inf):
        return V_inf + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)

    results = []

    cycle_numbers = df['Cycle Number'].unique()
    cycle_numbers.sort()

    for cycle_number in cycle_numbers:
        cycle_data = df[df['Cycle Number'] == cycle_number]
        max_soc = cycle_data[soc_col].max()

        for step_idx in step_indices:
            segment = cycle_data[cycle_data['Step Index'] == step_idx].copy()
            if segment.empty:
                print(f"No data for Cycle {cycle_number} Step {step_idx}, skipping.")
                continue

            segment = segment.sort_values(by=time_col).reset_index(drop=True)

            t = segment[time_col].values
            t = t - t[0]
            v = segment[voltage_col].values

            segment_duration = t[-1]
            duration_to_use = min(duration, segment_duration)

            mask = t <= duration_to_use
            t_fit = t[mask]
            v_fit = v[mask]

            A1_guess = v_fit[0] - v_fit[-1]
            A2_guess = A1_guess / 2
            tau1_guess = duration_to_use / 10 if duration_to_use > 0 else 0.01
            tau2_guess = duration_to_use / 2 if duration_to_use > 0 else 0.05
            V_inf_guess = v_fit[-1]

            p0 = [A1_guess, tau1_guess, A2_guess, tau2_guess, V_inf_guess]
            bounds = ([-np.inf, 0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])

            try:
                popt, _ = curve_fit(double_exponential, t_fit, v_fit, p0=p0, bounds=bounds, maxfev=10000)
                A1, tau1, A2, tau2, V_inf = popt

                I_step = np.mean(segment[current_col])
                I_step = I_step if abs(I_step) > 1e-6 else np.nan

                R1 = abs(A1 / I_step) if not np.isnan(I_step) else np.nan
                R2 = abs(A2 / I_step) if not np.isnan(I_step) else np.nan
                C1 = tau1 / R1 if R1 > 0 else np.nan
                C2 = tau2 / R2 if R2 > 0 else np.nan

                results.append({
                    'Cycle Number': cycle_number,
                    'Step Index': step_idx,
                    'Max SoC': max_soc,
                    'I_step [A]': round(I_step, 2),
                    'R1 [mΩ]': round(R1 * 1000, 3),
                    'C1 [F]': round(C1, 2),
                    'tau1 [s]': round(tau1 * 60, 2),
                    'R2 [mΩ]': round(R2 * 1000, 3),
                    'C2 [F]': round(C2, 2),
                    'tau2 [s]': round(tau2 * 60, 2),
                    'V_inf [V]': round(V_inf, 4),
                    'A1': A1,
                    'A2': A2,
                    'tau1_fit': tau1,
                    'tau2_fit': tau2
                })

            except Exception as e:
                print(f"Fit failed for Cycle {cycle_number} Step {step_idx}: {e}")

    results_df = pd.DataFrame(results)

    # === Grid plot after fitting all ===
    if plot and not results_df.empty:
        n_plots = len(results_df)
        plots_per_row = 5
        n_rows = math.ceil(n_plots / plots_per_row)

        fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(plots_per_row * 4, n_rows * 3))
        axes = axes.flatten()

        for i, row in results_df.iterrows():
            cycle_number = row['Cycle Number']
            step_idx = row['Step Index']

            ax = axes[i]

            segment = df[(df['Cycle Number'] == cycle_number) & (df['Step Index'] == step_idx)].copy()
            segment = segment.sort_values(by=time_col).reset_index(drop=True)
            t = segment[time_col].values
            t = t - t[0]
            v = segment[voltage_col].values

            duration_to_use = min(duration, t[-1])
            mask = t <= duration_to_use
            t_fit = t[mask]
            v_fit = v[mask]

            # Use stored parameters
            A1 = row['A1']
            tau1 = row['tau1_fit']
            A2 = row['A2']
            tau2 = row['tau2_fit']
            V_inf = row['V_inf [V]']

            v_model = V_inf + A1 * np.exp(-t_fit / tau1) + A2 * np.exp(-t_fit / tau2)

            ax.plot(t_fit, v_fit, 'o', label='Measured', markersize=3, alpha=0.6)
            ax.plot(t_fit, v_model, '-', label='Fit', alpha=0.8)
            ax.set_title(f'Cyc {cycle_number}, Step {step_idx}', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.3)

            if i % plots_per_row == 0:
                ax.set_ylabel('Voltage [V]')
            if i >= (n_rows - 1) * plots_per_row:
                ax.set_xlabel('Time [min]')
            ax.tick_params(labelsize=7)

        # Hide any unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle('2RC Fits Across Cycles and Steps', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    return results_df


def average_params_by_peak_type(df_params_all):
    # Define step indices
    discharge_steps = [11, 15]
    charge_steps = [13, 17]

    # Filter data
    discharge = df_params_all[df_params_all['Step Index'].isin(discharge_steps)]
    charge = df_params_all[df_params_all['Step Index'].isin(charge_steps)]

    # Aggregation (excluding Max SoC here)
    agg_dict = {
        'R1 [mΩ]': 'mean',
        'C1 [F]': 'mean',
        'tau1 [s]': 'mean',
        'R2 [mΩ]': 'mean',
        'C2 [F]': 'mean',
        'tau2 [s]': 'mean',
        'V_inf [V]': 'mean',
    
    }

    # Group by Cycle Number
    discharge_avg = discharge.groupby('Cycle Number').agg(agg_dict).add_suffix('_dis')
    charge_avg = charge.groupby('Cycle Number').agg(agg_dict).add_suffix('_cha')

    # Get Max SoC once (from either, e.g., charge)
    max_soc = charge.groupby('Cycle Number')['Max SoC'].max().rename('Max SoC')

    # Merge everything
    avg_params = pd.concat([max_soc, charge_avg, discharge_avg], axis=1).reset_index()

    # Optional: Clean column names (remove units)
    avg_params.columns = (
        avg_params.columns
        .str.replace(r' \[.*?\]', '', regex=True)  # Remove units like [mΩ], [F]
        
    )

    return avg_params



def compute_resistances(df, current_tolerance=0.02, n_points=5):
    """
    Compute instantaneous resistances R1–R4 for each cycle using voltage and current
    values from the n-th point after a current step, and the last point before it.

    Returns:
        pd.DataFrame with columns:
        ['Cycle Number', 'SoC', 'R1', 'R2', 'R3', 'R4', 'R0_dis', 'R0_cha', 'R_avg']
    """
    import numpy as np

    transitions = [-38, 38, -76, 75.9]  # Current steps for R1, R2, R3, R4
    results = []

    for cycle, group in df.groupby('Cycle Number'):
        group = group.reset_index(drop=True)
        soc = group['SoC'].max()
        resistances = []

        for target_current in transitions:
            condition = (
                (group['Current (A)'].shift(1).abs() < current_tolerance) &
                (group['Current (A)'].sub(target_current).abs() <= current_tolerance)
            )
            match_idx = group[condition].index

            if not match_idx.empty:
                i = match_idx[0]
                j = i + (n_points - 1)

                if i > 0 and j < len(group):
                    v_before = group.loc[i - 1, 'Voltage (V)']
                    i_before = group.loc[i - 1, 'Current (A)']

                    v_after = group.loc[j, 'Voltage (V)']
                    i_after = group.loc[j, 'Current (A)']

                    delta_v = v_before - v_after
                    delta_i = i_after - i_before

                    resistance = abs(1000 * delta_v / delta_i) if abs(delta_i) > 1e-6 else None
                else:
                    resistance = None
            else:
                resistance = None

            resistances.append(resistance)

        # Convert all None to np.nan
        resistances = [np.nan if r is None else r for r in resistances]
        R1, R2, R3, R4 = resistances
        R0_dis = np.nanmean([R1, R3])
        R0_cha = np.nanmean([R2, R4])
        R_avg = np.nanmean(resistances)

        results.append({
            'Cycle Number': cycle,
            'SoC': soc,
            'R1': R1,
            'R2': R2,
            'R3': R3,
            'R4': R4,
            'R0_dis': R0_dis,
            'R0_cha': R0_cha,
            'R_avg': R_avg
        })

    return pd.DataFrame(results)



def plot_cha_dis_params(df):
    """
    Plot paired charge/discharge parameters over cycle number.
    Ignores V_inf_dis. Pairs are matched by name prefix (R1, C1, etc.).
    """
    # Define param base names and their corresponding cha/dis column suffixes
    param_pairs = [
        ("R0", "R0_cha", "R0_dis"),
        ("R1", "R1_cha", "R1_dis"),
        ("R2", "R2_cha", "R2_dis"),
        ("C1", "C1_cha", "C1_dis"),
        ("C2", "C2_cha", "C2_dis"),
        ("tau1", "tau1_cha", "tau1_dis"),      
        ("tau2", "tau2_cha", "tau2_dis"),
        ("V_inf", "V_inf_cha", "V_inf_dis"),  # Will be skipped
        
    ]

    # Filter out V_inf
    #param_pairs = [pair for pair in param_pairs if pair[0] != "V_inf"]

    num_plots = len(param_pairs)
    fig, axs = plt.subplots(num_plots, 1, figsize=(8, 2.5 * num_plots), sharex=True)

    for i, (label, cha_col, dis_col) in enumerate(param_pairs):
        ax = axs[i]
        ax.plot(df["Max SoC"], df[cha_col], label=f"{label} (cha)", marker='o')
        ax.plot(df["Max SoC"], df[dis_col], label=f"{label} (dis)", marker='s')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)

    axs[-1].set_xlabel("Max SoC")
    plt.tight_layout()
    plt.show()