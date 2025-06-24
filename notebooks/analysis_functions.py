"""
Module: resource_depletion
Reusable functions for data loading, cleaning, transformation, plotting, and statistical analysis.
"""

import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, f_oneway, kruskal
from itertools import combinations
from typing import Dict
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


# Filename patterns for parsing metadata from filenames
# Define the pattern of filenames

GameTheoretic_filename_pattern_DQN =  re.compile(r"results_(?P<simulation_index>\d{3})_(?P<episodes>\d+)_DQN_"
                                                r"(?P<emotion>[^_]+)_(?P<see_emotions>[^_]+)_"
                                                r"(?P<alpha>[\d.]+)_(?P<beta>[\d.]+)_(?P<smoothing>[^_]+)_(?P<threshold>[\d.]+)_(?P<rounder>[\d.]+)_"
                                                r"(?P<learning_rate>[\d.]+)_(?P<gamma>[\d.]+)_(?P<epsilon>[\d.]+)_(?P<epsilon_decay>[\d.]+)_(?P<epsilon_min>[\d.]+)_"
                                                r"(?P<batch_size>[\d.]+)_(?P<hidden_size>[\d.]+)_(?P<update_target_every>[\d.]+)_"
                                                r"(?P<random_suffix>\d{6})_(?P<suffix>[a-zA-Z]+_[a-zA-Z]+)\.csv"
)

GameTheoretic_filename_pattern_QL = re.compile(r"results_(?P<simulation_index>\d{3})_(?P<episodes>\d+)_QLearning_"
                                              r"(?P<emotion>[^_]+)_(?P<see_emotions>[^_]+)_"
                                              r"(?P<alpha>[\d.]+)_(?P<beta>[\d.]+)_(?P<smoothing>[^_]+)_(?P<threshold>[\d.]+)_(?P<rounder>[\d.]+)_"
                                              r"(?P<learning_rate>[\d.]+)_(?P<gamma>[\d.]+)_(?P<epsilon>[\d.]+)_(?P<epsilon_decay>[\d.]+)_(?P<epsilon_min>[\d.]+)_"
                                              r"(?P<random_suffix>\d{6})_(?P<suffix>[a-zA-Z]+_[a-zA-Z]+)\.csv"
)


Maze2D_filename_order_QL = re.compile(
    r"maze2d_results_(?P<simulation_index>\d{3})_(?P<episodes>\d+)_QLearning_"
    r"(?P<emotion>[^_]+)_(?P<see_emotions>[^_]+)_"
    r"(?P<alpha>[\d.]+)_(?P<beta>[\d.]+)_(?P<smoothing>[^_]+)_(?P<threshold>[\d.]+)_(?P<rounder>[\d.]+)_"
    r"(?P<learning_rate>[\d.]+)_(?P<gamma>[\d.]+)_(?P<epsilon>[\d.]+)_(?P<epsilon_decay>[\d.]+)_(?P<epsilon_min>[\d.]+)_"
    r"(?P<random_suffix>\d{6})_(?P<suffix>[a-zA-Z]+_[a-zA-Z]+)\.csv"
)

Maze2D_filename_order_DQN = re.compile(
    r"maze2d_results_(?P<simulation_index>\d{3})_(?P<episodes>\d+)_DQN_"
    r"(?P<emotion>[^_]+)_(?P<see_emotions>[^_]+)_"
    r"(?P<alpha>[\d.]+)_(?P<beta>[\d.]+)_(?P<smoothing>[^_]+)_(?P<threshold>[\d.]+)_(?P<rounder>[\d.]+)_"
    r"(?P<learning_rate>[\d.]+)_(?P<gamma>[\d.]+)_(?P<epsilon>[\d.]+)_(?P<epsilon_decay>[\d.]+)_(?P<epsilon_min>[\d.]+)_"
    r"(?P<batch_size>[\d.]+)_(?P<hidden_size>[\d.]+)_(?P<update_target_every>[\d.]+)_"
    r"(?P<random_suffix>\d{6})_(?P<suffix>[a-zA-Z]+_[a-zA-Z]+)\.csv"
)

FILENAME_PATTERNS = [
    GameTheoretic_filename_pattern_DQN,
    GameTheoretic_filename_pattern_QL,
    Maze2D_filename_order_DQN,
    Maze2D_filename_order_QL
]

FILENAME_PATTERNS_PAIR = [
    ("Gametheoretic", GameTheoretic_filename_pattern_DQN),
    ("Gametheoretic", GameTheoretic_filename_pattern_QL),
    ("maze2d", Maze2D_filename_order_DQN),
    ("maze2d", Maze2D_filename_order_QL)
]

############################################################
##### PROCESSING OF FILES
############################################################

def parse_results_filenames(folder_path: str, filename_patterns=None) -> pd.DataFrame:
    """
    Scans a folder for result filenames and extracts simulation parameters into a DataFrame.

    Args:
        folder_path (str): Path to the folder containing result CSV files.
        filename_patterns (list): List of compiled regex patterns to match filenames.
                                  If None, uses global FILENAME_PATTERNS.

    Returns:
        pd.DataFrame: DataFrame containing parsed parameters from filenames.
    """
    if filename_patterns is None:
        filename_patterns = FILENAME_PATTERNS

    data = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        matched = False
        for pattern in filename_patterns:
            match = pattern.match(filename)
            if match:
                file_data = match.groupdict()
                file_data["filename"] = filename
                data.append(file_data)
                matched = True
                break  # Stop at the first match
        
        if not matched:
            print(f"Warning: filename did not match any pattern: {filename}")

    if not data:
        print("No matching filenames found.")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Convert numeric fields from str to numeric types if possible
    for col in df.columns:
        if col not in {"filename", "emotion", "see_emotions", "suffix"}:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass  # leave as is if conversion fails

    return df

def print_unique_parameter_values(df: pd.DataFrame, exclude: list = None):
    """
    Print a table with parameter names and their unique values.

    Args:
        df (pd.DataFrame): The input DataFrame with simulation parameters.
        exclude (list): Optional list of column names to exclude (e.g., ['filename', 'simulation_index']).
    """
    if exclude is None:
        exclude = ['filename', 'simulation_index']

    param_cols = [col for col in df.columns if col not in exclude]
    summary = {"parameter": [], "unique_values": []}

    for col in param_cols:
        summary["parameter"].append(col)
        unique_vals = sorted(df[col].dropna().unique().tolist())
        summary["unique_values"].append(unique_vals)

    summary_df = pd.DataFrame(summary)
    print(summary_df)

def compute_alpha_from_df(df: pd.DataFrame, nb_agents: int, verbose: bool = False) -> float:
    """
    Estimate alpha from a DataFrame containing total personal, empathic, and combined rewards
    for multiple agents. Alpha is estimated as:
        alpha = (combined - personal) / (empathic - personal)

    Args:
        df (pd.DataFrame): Input DataFrame containing the reward columns.
        nb_agents (int): Number of agents.
        verbose (bool): If True, print alpha stats per agent.

    Returns:
        float: Estimated average alpha.
    """
    alphas = []

    for agent_idx in range(nb_agents):
        personal_col = f"total_personal_reward_{agent_idx}"
        empathic_col = f"total_empathic_reward_{agent_idx}"
        combined_col = f"total_combined_reward_{agent_idx}"

        # Validate presence of required columns
        for col in [personal_col, empathic_col, combined_col]:
            if col not in df.columns:
                raise KeyError(f"Missing column: {col}")

        personal = df[personal_col].astype(float)
        empathic = df[empathic_col].astype(float)
        combined = df[combined_col].astype(float)

        denominator = empathic - personal
        valid_mask = denominator != 0

        if valid_mask.sum() == 0:
            if verbose:
                print(f"⚠ No valid rows for agent {agent_idx} (division by zero in alpha computation)")
            continue

        alpha_agent = (combined[valid_mask] - personal[valid_mask]) / denominator[valid_mask]
        # Filter between 0 and 1 (if needed)
        alpha_agent = alpha_agent[(alpha_agent >= 0) & (alpha_agent <= 1)]
        alphas.append(alpha_agent)

        if verbose and not alpha_agent.empty:
            print(f"[Agent {agent_idx}] alpha mean: {alpha_agent.mean():.4f}, samples: {len(alpha_agent)}")

    if not alphas:
        raise ValueError("No valid alpha values could be computed.")

    all_alphas = pd.concat(alphas)
    mean_alpha = all_alphas.mean()
    if verbose:
        print(f"Estimated average alpha from DataFrame: {mean_alpha:.4f}")
    return mean_alpha

def aggregate_results_by_suffix(folder_path: str, target_suffix: str, environment_type: str = None) -> pd.DataFrame:
    """
    Aggregates multiple CSV files in a folder by a given suffix and environment type into one DataFrame.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        target_suffix (str): Suffix indicating target CSV files (e.g., "episode_summary" or "step_data").
        environment_type (str): Type of environment ("gametheoretic" or "maze2d"). If provided, only files of that type are aggregated.

    Returns:
        pd.DataFrame: Aggregated DataFrame of all matching CSV files.
    """
    all_data = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        for source_type, pattern in FILENAME_PATTERNS_PAIR:
            if environment_type and source_type.lower() != environment_type.lower():
                continue

            match = pattern.match(filename)
            if match:
                metadata = match.groupdict()
                if metadata.get("suffix", "").strip() == target_suffix.strip():
                    file_path = os.path.join(folder_path, filename)
                    try:
                        df = pd.read_csv(file_path)
                        for key, value in metadata.items():
                            df[key] = value
                        df["source"] = source_type.lower()  # normalize environment type
                        all_data.append(df)
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
                break

    if not all_data:
        print(f"No matching files found for suffix '{target_suffix}' and source '{environment_type}'.")
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)

    # Convert numeric-like columns to numeric types
    for col in final_df.columns:
        if col not in {"emotion", "see_emotions", "suffix", "filename", "source"}:
            try:
                final_df[col] = pd.to_numeric(final_df[col])
            except Exception:
                pass

    # Save aggregated data to CSV in the same folder
    filtered_tag = f"_{environment_type.lower()}" if environment_type else ""
    output_filename = f"aggregated_{target_suffix}{filtered_tag}.csv"
    output_path = os.path.join(folder_path, output_filename)
    final_df.to_csv(output_path, index=False)
    print(f"Saved aggregated data to: {output_path}")
    return final_df

def sum_columns_by_format(df: pd.DataFrame, prefix: str, new_column_name: str = None) -> pd.DataFrame:
    """
    Adds a new column to the DataFrame that contains the row-wise sum of all columns
    matching the pattern '{prefix}_<i>', where 'i' is an integer.

    Args:
        df (pd.DataFrame): The DataFrame to modify (operation is done in place).
        prefix (str): The prefix of the target columns (e.g., 'total_combined_reward').
        new_column_name (str, optional): The name of the new column to add.
                                         Defaults to 'Sum_<prefix>' if not specified.

    Returns:
        pd.DataFrame: The DataFrame with the new summed column.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    matching_cols = [col for col in df.columns if pattern.match(col)]

    if not matching_cols:
        raise ValueError(f"No columns found with prefix '{prefix}_<i>'")

    if new_column_name is None:
        new_column_name = f"Sum_{prefix}"

    df[new_column_name] = df[matching_cols].sum(axis=1)
    return df

def summarize_simulations(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes per-episode simulation data into one row per simulation with computed metrics.

    Args:
        df_summary (pd.DataFrame): DataFrame with per-episode simulation results.

    Returns:
        pd.DataFrame: Summary DataFrame, one row per simulation with aggregated metrics.
    """
    required_columns = [
        'simulation_index', 'alpha', 'see_emotions', 'total_steps',
        'initial_resources', 'max_steps', 'resource_remaining',
        'total_personal_reward__averaged_efficiency', 'gini_coef'
    ]
    missing = [col for col in required_columns if col not in df_summary.columns]
    if missing:
        raise ValueError(f"Missing columns in input DataFrame: {missing}")

    df_sim = (
        df_summary
        .groupby(['simulation_index', 'alpha', 'see_emotions'])
        .agg(
            total_steps=('total_steps', 'sum'),
            initial_resources=('initial_resources', 'sum'),
            max_steps=('max_steps', 'sum'),
            resource_remaining=('resource_remaining', 'sum'),
            total_efficiency=('total_personal_reward__averaged_efficiency', 'sum'),
            gini_coef=('gini_coef', 'mean'),
        )
        .assign(
            step_number_depletion=lambda x: 1 - (x['total_steps'] / x['max_steps']),
            resource_quantity_depletion=lambda x: 1 - (x['resource_remaining'] / x['initial_resources']),
            efficiency_metric=lambda x: x['total_efficiency'] / x['total_steps'],
        )
        .reset_index()
        .rename(columns={'simulation_index': 'simulation_number'})
        [['simulation_number', 'alpha', 'see_emotions', 'total_steps', 'initial_resources',
          'step_number_depletion', 'resource_quantity_depletion', 'efficiency_metric', 'gini_coef']]
    )

    return df_sim

def windowed_avg_combined_reward(
    df: pd.DataFrame,
    reward_prefix: str = "total_combined_reward_",
    episode_column: str = "episode",
    simulation_id_column: str = "simulation_index",
    window_size: int = 5,
    aggregation_mode: str = "mean",  # or "best"
    plot: bool = False,
    title: str = None,
    legend: str = None
) -> pd.DataFrame:
    """
    Computes a windowed moving average of a reward (or any scalar metric) per episode across simulations.

    Args:
        df (pd.DataFrame): Input DataFrame.
        reward_prefix (str): Prefix of the reward column(s) to aggregate. If it matches a single column exactly, it's used directly.
        episode_column (str): Column name for episodes.
        simulation_id_column (str): Column indicating different simulations.
        window_size (int): Window size for moving average.
        aggregation_mode (str): 'mean' or 'best' for multi-agent columns (default 'mean').
        plot (bool): Whether to plot the resulting moving average.
        title (str): Plot title (optional).
        legend (str): Legend label for the plot (optional).

    Returns:
        pd.DataFrame: DataFrame with columns ['episode', 'mean_reward', 'moving_avg'].
    """
    if title is None:
        title = f'Windowed Avg: {reward_prefix}{aggregation_mode}'
    if legend is None:
        legend = f"Windowed Avg Smoothing: {window_size} episodes"

    # Determine reward aggregation
    if reward_prefix in df.columns:
        df["aggregated_reward"] = df[reward_prefix]
    else:
        reward_cols = [col for col in df.columns if col.startswith(reward_prefix)]
        if not reward_cols:
            raise ValueError(f"No reward columns found with prefix '{reward_prefix}'")
        if aggregation_mode == "mean":
            df["aggregated_reward"] = df[reward_cols].mean(axis=1)
        elif aggregation_mode == "best":
            df["aggregated_reward"] = df[reward_cols].max(axis=1)
        else:
            raise ValueError("aggregation_mode must be 'mean' or 'best'")

    # Group by episode to average across simulations
    episode_avg = (
        df.groupby(episode_column)["aggregated_reward"]
        .mean()
        .reset_index()
        .rename(columns={"aggregated_reward": "mean_reward"})
    )

    # Apply centered moving average smoothing
    episode_avg["moving_avg"] = episode_avg["mean_reward"].rolling(window=window_size, min_periods=1, center=True).mean()

    # Optional plotting
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(episode_avg[episode_column], episode_avg["moving_avg"], label=legend)
        plt.xlabel("Episode")
        ylabel = "Reward" if "reward" in reward_prefix else reward_prefix.replace('_', ' ').title()
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return episode_avg

def compute_episode_reward_correlation(
    df: pd.DataFrame,
    reward_column: str = 'total_personal_reward',
    simulation_column: str = 'simulation_index',
    type_column: str = 'alpha'
) -> pd.DataFrame:
    """
    Computes correlation between episode number and a reward metric across simulations, split by a category.
    If residuals are normally distributed (Shapiro-Wilk p >= 0.05), Pearson correlation is used;
    otherwise Spearman correlation is used.

    Args:
        df (pd.DataFrame): Input DataFrame containing simulation results (must have 'episode' column).
        reward_column (str): Column with total reward per episode.
        simulation_column (str): Column identifying the simulation.
        type_column (str): Column distinguishing between categories (e.g., 'alpha').

    Returns:
        pd.DataFrame: DataFrame with columns [simulation, type, correlation, p_value, test_used, shapiro_stat, shapiro_p, n_episodes].
    """
    results = []
    if 'episode' not in df.columns:
        raise ValueError("DataFrame must include an 'episode' column.")

    grouped = df.groupby([simulation_column, type_column])
    for (sim_id, sim_type), group in grouped:
        if group['episode'].nunique() > 1:
            # Fit linear regression to compute residuals
            slope, intercept, _, _, _ = stats.linregress(group['episode'], group[reward_column])
            predicted = intercept + slope * group['episode']
            residuals = group[reward_column] - predicted

            # Test residuals normality
            shapiro_stat, shapiro_p = stats.shapiro(residuals)

            if shapiro_p >= 0.05:
                # Residuals normal => Pearson correlation
                corr, p_value = stats.pearsonr(group['episode'], group[reward_column])
                test_used = 'Pearson'
            else:
                # Residuals non-normal => Spearman correlation
                corr, p_value = stats.spearmanr(group['episode'], group[reward_column])
                test_used = 'Spearman'

            results.append({
                'simulation': sim_id,
                'type': sim_type,
                'correlation': corr,
                'p_value': p_value,
                'test_used': test_used,
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'n_episodes': len(group)
            })

    return pd.DataFrame(results)

def gini_coefficient(arr: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1D numpy array.
    """
    arr = arr.flatten()
    if np.amin(arr) < 0:
        arr = arr - np.amin(arr)  # Shift to non-negative
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    n = len(arr)
    diff_sum = np.sum(np.abs(np.subtract.outer(arr, arr)))
    return diff_sum / (2 * n**2 * mean)

def parse_value(val):
    """
    Convert a value to a numpy array, handling string representations of lists.
    """
    if pd.isnull(val):
        return np.array([0.0])
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return np.array(parsed, dtype=float)
            return np.array([float(parsed)])
        except (ValueError, SyntaxError):
            return np.array([0.0])
    elif isinstance(val, (list, tuple, np.ndarray)):
        return np.array(val, dtype=float)
    else:
        return np.array([float(val)])

def compute_gini_for_df(df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    Compute Gini coefficient for each row across columns starting with a prefix.

    Args:
        df (pd.DataFrame): Input DataFrame.
        prefix (str): Column prefix to select.

    Returns:
        pd.Series: Gini coefficient per row.
    """
    cols = [col for col in df.columns if col.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found starting with prefix '{prefix}'")

    def row_gini(row):
        values = []
        for val in row[cols]:
            parsed_val = parse_value(val)
            values.extend(parsed_val.flatten())
        return gini_coefficient(np.array(values, dtype=float))

    return df.apply(row_gini, axis=1)

def compute_efficiency_for_df(df: pd.DataFrame, prefix: str, new_column_name: str) -> pd.DataFrame:
    """
    Compute the average across columns starting with a given prefix for each row in df.
    Handles string representations of single-element lists (e.g., '[-1.]').

    Args:
        df (pd.DataFrame): Input DataFrame.
        prefix (str): Prefix for selecting target columns.
        new_column_name (str): Name of the new column to store the computed average.

    Returns:
        pd.DataFrame: DataFrame with the new column added.
    """
    cols = [col for col in df.columns if col.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found starting with prefix '{prefix}'")

    for col in cols:
        # Clean column: parse strings like '[-1.]' into float -1.0
        def parse_cell(val):
            if isinstance(val, (float, int)):
                return float(val)
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list) and len(parsed) == 1:
                        return float(parsed[0])
                    return float(parsed)
                except Exception:
                    raise ValueError(f"Value '{val}' in column '{col}' could not be parsed to float.")
            raise ValueError(f"Unsupported value type {type(val)} in column '{col}'.")
        df[col] = df[col].apply(parse_cell)

    df[new_column_name] = df[cols].mean(axis=1)
    return df

def compute_normalized_efficiency_for_df(
    df: pd.DataFrame,
    prefix: str,
    new_column_name: str,
    steps_column: str = 'total_steps'
) -> pd.DataFrame:
    """
    Compute the average across columns starting with a given prefix for each row,
    then normalize by the number of steps.

    Args:
        df (pd.DataFrame): Input DataFrame.
        prefix (str): Prefix to select columns (e.g., 'total_personal_reward_').
        new_column_name (str): Name of the new column to add.
        steps_column (str): Column indicating number of steps (default 'total_steps').

    Returns:
        pd.DataFrame: DataFrame with the new column added.
    """
    cols = [col for col in df.columns if col.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found starting with prefix '{prefix}'")

    # Parse values like '[-1.]' or '[0.]'
    def parse_cell(val):
        if isinstance(val, (float, int)):
            return float(val)
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list) and len(parsed) == 1:
                    return float(parsed[0])
                return float(parsed)
            except Exception:
                raise ValueError(f"Value '{val}' could not be parsed to float.")
        raise ValueError(f"Unsupported value type: {type(val)}")

    for col in cols:
        df[col] = df[col].apply(parse_cell)

    # Compute mean reward across selected columns
    df["_temp_mean"] = df[cols].mean(axis=1)
    # Normalize by steps
    df[new_column_name] = df["_temp_mean"] / df[steps_column]
    df.drop(columns=["_temp_mean"], inplace=True)

    return df

def clean_initial_resources_column(df: pd.DataFrame, column: str = 'initial_resources') -> None:
    """
    Ensure the initial resource column contains numeric values.

    Args:
        df (pd.DataFrame): DataFrame containing the column.
        column (str): Column name of initial resources.
    """
    def parse_val(x):
        if pd.isna(x):
            return np.nan
        try:
            val = ast.literal_eval(x) if isinstance(x, str) else x
            if isinstance(val, (list, tuple)) and val:
                return float(val[0])
            return float(val)
        except:
            return np.nan

    if column in df.columns and df[column].dtype == object:
        df[column] = df[column].apply(parse_val)
        df[column] = pd.to_numeric(df[column], errors='coerce')

def compute_resource_quantity_depletion(df: pd.DataFrame) -> None:
    """
    Compute the proportion of resource quantity depleted at the end of each episode.
    Adds a column 'resource_quantity_depletion' to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns 'initial_resources' and 'resource_remaining'.
    """
    clean_initial_resources_column(df, 'initial_resources')
    group_keys = ['simulation_index', 'episode']
    depletion_df = df.groupby(group_keys).agg(
        ending_resource=('resource_remaining', 'last'),
        starting_resource=('initial_resources', 'first')
    ).reset_index()
    depletion_df['resource_quantity_depletion'] = 1 - depletion_df['ending_resource'] / depletion_df['starting_resource']
    df_merged = df.merge(depletion_df[group_keys + ['resource_quantity_depletion']], on=group_keys, how='left')
    df['resource_quantity_depletion'] = df_merged['resource_quantity_depletion']

def compute_step_number_depletion(df: pd.DataFrame) -> None:
    """
    Compute how early each episode ended due to resource depletion.
    Adds a column 'step_number_depletion' to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns 'total_steps' and 'max_steps'.
    """
    clean_initial_resources_column(df, 'initial_resources')
    df['resource_quantity_depletion'] = 1 - df['resource_remaining'] / df['initial_resources']
    df['step_number_depletion'] = 1 - df['total_steps'] / df['max_steps']

############################################################
##### Plotting
############################################################

def plot_mean_and_range_across_simulations(
    df: pd.DataFrame,
    value_col: str,
    simulation_col: str = 'seed',
    episode_col: str = 'episode',
    step_col: str = 'step',
    is_step_csv: bool = False,
    title: str = None,
    ylabel: str = None,
    rolling_window: int = None,
    plot_individual: bool = False
):
    """
    Plot the mean and range (min-max) of a value across simulations over episodes or steps.
    Optionally overlays individual simulation traces.

    Args:
        df (pd.DataFrame): DataFrame with simulation data.
        value_col (str): Column name of the value to plot.
        simulation_col (str): Column identifying each simulation (default 'seed').
        episode_col (str): Column name for episodes (default 'episode').
        step_col (str): Column name for steps (if step-level data).
        is_step_csv (bool): Whether the data is step-level (True) or episode-level (False).
        title (str): Plot title.
        ylabel (str): Y-axis label.
        rolling_window (int): Window size for smoothing (None or >1).
        plot_individual (bool): If True, plot each simulation individually in background.
    """
    if title is None:
        title = f"{value_col} (mean and range over simulations)"

    if is_step_csv:
        group_cols = [simulation_col, episode_col, step_col]
        avg_group_cols = [episode_col, step_col]
    else:
        group_cols = [simulation_col, episode_col]
        avg_group_cols = [episode_col]

    sim_stats = df.groupby(group_cols)[value_col].mean().reset_index()
    summary = sim_stats.groupby(avg_group_cols)[value_col].agg(['mean', 'min', 'max']).reset_index()

    if rolling_window and rolling_window > 1:
        summary['mean'] = summary['mean'].rolling(rolling_window, min_periods=1, center=True).mean()
        summary['min'] = summary['min'].rolling(rolling_window, min_periods=1, center=True).mean()
        summary['max'] = summary['max'].rolling(rolling_window, min_periods=1, center=True).mean()

    plt.figure(figsize=(10, 6))
    if plot_individual:
        for sim_id, sim_df in sim_stats.groupby(simulation_col):
            x = sim_df[step_col] if is_step_csv else sim_df[episode_col]
            y = sim_df[value_col]
            if rolling_window:
                y = y.rolling(rolling_window, min_periods=1).mean()
            plt.plot(x, y, color='gray', alpha=0.3)

    x_vals = summary[step_col] if is_step_csv else summary[episode_col]
    plt.fill_between(
        x_vals,
        summary['min'],
        summary['max'],
        color='blue',
        alpha=0.2,
        label='Range (min-max)'
    )
    plt.plot(x_vals, summary['mean'], color='blue', linewidth=2, label='Mean')
    plt.xlabel('Step' if is_step_csv else 'Episode')
    plt.ylabel(ylabel if ylabel else value_col)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_avg_steps_to_depletion(
    df: pd.DataFrame,
    steps_col: str = 'steps_to_depletion',
    episode_col: str = 'episode',
    rolling_window: int = 5,
    title: str = "Fluctuation of Average Steps to Depletion",
    ylabel: str = "Average Steps to Depletion",
    figsize: tuple = (10, 6)
):
    """
    Plot the fluctuation of average steps to depletion over episodes.

    Args:
        df (pd.DataFrame): DataFrame containing steps to depletion data.
        steps_col (str): Column with steps until depletion (default 'steps_to_depletion').
        episode_col (str): Episode column (default 'episode').
        rolling_window (int): Window size for smoothing (default 5).
        title (str): Plot title.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size.
    """
    summary = df.groupby(episode_col)[steps_col].mean().reset_index()
    if rolling_window > 1:
        summary['steps_smoothed'] = summary[steps_col].rolling(rolling_window, min_periods=1, center=True).mean()
    else:
        summary['steps_smoothed'] = summary[steps_col]

    plt.figure(figsize=figsize)
    plt.plot(summary[episode_col], summary['steps_smoothed'], marker='o', linestyle='-', color='tab:green', label='Smoothed avg steps')
    plt.scatter(summary[episode_col], summary[steps_col], color='tab:gray', alpha=0.6, label='Raw avg steps')
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

############################################################
##### STATISTICAL TESTS
############################################################


def cohen_d(x, y):
    """Calculate Cohen's d effect size for two independent samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0


def rank_biserial_u(u, nx, ny):
    """Calculate rank-biserial correlation effect size from Mann-Whitney U."""
    return 1 - (2 * u) / (nx * ny)


def statistical_test(df, group_col, value_col, alpha=0.05):
    """
    Perform statistical comparison across two or more groups.

    For each group, it computes descriptive statistics, tests for normality (Shapiro-Wilk),
    and tests for equal variances (Levene's test). Based on these assumptions:
      - If normality and equal variances hold:
          - Two groups: Student's or Welch's t-test (with Cohen's d)
          - >2 groups: One-way ANOVA with Tukey HSD post-hoc (if significant)
      - If assumptions are violated:
          - Two groups: Mann-Whitney U test (with rank-biserial r)
          - >2 groups: Kruskal-Wallis with Mann-Whitney pairwise comparisons,
                      adjusted with Holm correction

    Parameters:
        df (pd.DataFrame): The dataset.
        group_col (str): Name of the column indicating group membership.
        value_col (str): Name of the numeric variable to compare.
        alpha (float): Significance level (default: 0.05).

    Returns:
        dict with keys:
            - 'groups': sorted list of group labels
            - 'descriptive': dict {group: {mean, std, median, n}}
            - 'normal': bool (True if all groups are normally distributed)
            - 'normality_test': dict {group: Shapiro-Wilk p-value or None}
            - 'equal_variance': bool (True if Levene’s test is non-significant)
            - 'levene_test': dict with 'statistic' and 'p_value'
            - 'test': name of the main test used
            - 'anova_stat' or 'statistic': test statistic
            - 'anova_p' or 'p_value': p-value of the main test
            - 'effect_size': float, if applicable
            - 'apa': APA-style summary string of the main test
            - 'posthoc': list of dicts for pairwise comparisons (if applicable)
            - 'posthoc_table': summary table (only for Tukey HSD)
            - 'pairwise': list of dicts for Mann-Whitney pairwise (if applicable)
    """
    results = {}
    groups = sorted(df[group_col].dropna().unique())  # Sort group labels
    data = [df[df[group_col] == grp][value_col].dropna() for grp in groups]
    results['groups'] = list(groups)

    # Descriptive statistics
    descriptives = {}
    for grp, values in zip(groups, data):
        descriptives[grp] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),
            'median': np.median(values),
            'n': len(values)
        }
    results['descriptive'] = descriptives

    # Shapiro-Wilk normality test
    normality_test = {}
    normal_flags = []
    for grp, values in zip(groups, data):
        if len(values) < 3:
            normality_test[grp] = None
            normal_flags.append(False)
        else:
            _, p = shapiro(values)
            normality_test[grp] = p
            normal_flags.append(p >= alpha)
    results['normality_test'] = normality_test
    results['normal'] = all(normal_flags)

    # Levene’s test for equal variances
    try:
        stat_levene, p_levene = levene(*data)
        results['levene_test'] = {'statistic': stat_levene, 'p_value': p_levene}
        results['equal_variance'] = p_levene >= alpha
    except Exception:
        results['levene_test'] = {'statistic': None, 'p_value': None}
        results['equal_variance'] = False

    # Main test logic
    if len(groups) == 2:
        g1, g2 = data
        name1, name2 = groups
        if results['normal']:
            if results['equal_variance']:
                stat, pval = ttest_ind(g1, g2, equal_var=True)
                test_name = "Student's t-test"
            else:
                stat, pval = ttest_ind(g1, g2, equal_var=False)
                test_name = "Welch's t-test"
            effect = cohen_d(g1, g2)
            apa = f"{test_name}: t({len(g1)+len(g2)-2}) = {stat:.2f}, p = {pval:.3f}, d = {effect:.2f}"
        else:
            u_stat, pval = mannwhitneyu(g1, g2, alternative='two-sided')
            effect = rank_biserial_u(u_stat, len(g1), len(g2))
            test_name = "Mann-Whitney U"
            apa = f"{test_name}: U = {u_stat:.2f}, p = {pval:.3f}, r = {effect:.2f}"

        results.update({
            'test': test_name,
            'statistic': stat if results['normal'] else u_stat,
            'p_value': pval,
            'effect_size': effect,
            'apa': apa
        })

    else:
        if results['normal'] and results['equal_variance']:
            stat, pval = f_oneway(*data)
            results.update({
                'test': "ANOVA",
                'anova_stat': stat,
                'anova_p': pval,
                'apa': f"ANOVA: F({len(groups)-1}, {len(df)-len(groups)}) = {stat:.2f}, p = {pval:.3f}"
            })
            if pval < alpha:
                try:
                    df_clean = df.dropna(subset=[group_col, value_col])
                    tukey = pairwise_tukeyhsd(endog=df_clean[value_col], groups=df_clean[group_col], alpha=alpha)
                    results['posthoc_table'] = tukey.summary()
                    comparisons = []
                    for (grp1, grp2), diff, p_adj in zip(combinations(tukey.groupsunique, 2), tukey.meandiffs, tukey.pvalues):
                        apa = f"Tukey HSD: {grp1} vs {grp2}, mean diff = {diff:.2f}, p_adj = {p_adj:.3f}"
                        comparisons.append({
                            'group1': grp1,
                            'group2': grp2,
                            'mean_diff': diff,
                            'p_adj': p_adj,
                            'apa': apa
                        })
                    results['posthoc'] = comparisons
                except ImportError:
                    results['posthoc'] = "statsmodels not installed"

        else:
            stat, pval = kruskal(*data)
            results.update({
                'test': "Kruskal-Wallis",
                'anova_stat': stat,
                'anova_p': pval,
                'apa': f"Kruskal-Wallis: H = {stat:.2f}, p = {pval:.3f}"
            })
            if pval < alpha:
                pairwise = []
                p_values = []
                comparisons = []
                for (i, grp1), (j, grp2) in combinations(enumerate(groups), 2):
                    g1, g2 = data[i], data[j]
                    print(f"\nComparing {grp1} (n={len(g1)}) with {grp2} (n={len(g2)})")
                    print("g1:", g1.tolist())
                    print("g2:", g2.tolist())
                    print(f"Means: g1 = {np.mean(g1):.6f}, g2 = {np.mean(g2):.6f}")
                    print(f"Std devs: g1 = {np.std(g1, ddof=1):.6f}, g2 = {np.std(g2, ddof=1):.6f}")
                    print(f"Same mean (tol=1e-6)?", np.isclose(np.mean(g1), np.mean(g2), atol=1e-6))

                for (i, grp1), (j, grp2) in combinations(enumerate(groups), 2):
                    g1, g2 = data[i], data[j]
                    u_stat, p_pair = mannwhitneyu(g1, g2, alternative='two-sided')
                    p_values.append(p_pair)
                    comparisons.append((grp1, grp2, u_stat))

                reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method='holm')

                for ((grp1, grp2, u_stat), p_raw, p_adj, rej) in zip(comparisons, p_values, p_adjusted, reject):
                    n1 = len(df[df[group_col] == grp1][value_col].dropna())
                    n2 = len(df[df[group_col] == grp2][value_col].dropna())
                    effect = rank_biserial_u(u_stat, n1, n2)
                    apa = (f"Mann-Whitney U: {grp1} vs {grp2}, U = {u_stat:.2f}, "
                           f"p_raw = {p_raw:.3f}, p_adj = {p_adj:.3f}, reject = {rej}, r = {effect:.2f}")
                    pairwise.append({
                        'group1': grp1,
                        'group2': grp2,
                        'U': u_stat,
                        'raw_p': p_raw,
                        'p_adj': p_adj,
                        'reject_null': rej,
                        'effect_size': effect,
                        'apa': apa
                    })
                results['pairwise'] = pairwise

    return results
