import joblib
from utils.eval import intersection_over_union
from utils.formatAndPreprocessNewPatterns import get_patetrn_name_by_encoding, get_pattern_encoding_by_name, get_reverse_pattern_encoding
import pandas as pd
import numpy as np
import math
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed
# Remove matplotlib imports and plotting function import
# import matplotlib.pyplot as plt
# from utils.functionalPatternLocateAndPlot import plot_pattern_groups_and_finalized_sections

# --- Global Configuration & Model Loading ---
# Load the pre-trained model and pattern encodings
# It's assumed 'Models/Width Aug OHLC_mini_rocket_xgb.joblib' is in the correct path
MODEL_PATH = 'Models/Width Aug OHLC_mini_rocket_xgb.joblib'
try:
    rocket_model_global = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the path is correct.")
    # You might want to exit or raise an exception here if the model is critical
    rocket_model_global = None 

pattern_encoding_reversed_global = get_reverse_pattern_encoding()

# Default parameters for the pattern location logic
WIN_SIZE_PROPORTIONS = np.round(np.logspace(0, np.log10(20), num=10), 2).tolist()
PADDING_PROPORTION = 0.6
STRIDE = 1
# Default probability thresholds for pattern identification.
PROBABILITY_THRESHOLD_LIST = [0.8884, 0.8676, 0.5620, 0.5596, 0.5132, 0.8367, 0.7635]
PROB_THRESHOLD_NO_PATTERN = 0.5 # Threshold to mark as 'No Pattern'

# DBSCAN Clustering parameters
DBSCAN_EPS = 0.04
DBSCAN_MIN_SAMPLES = 3

# --- Private Helper Functions ---

def _process_window(i, ohlc_data_segment, rocket_model, probability_threshold, pattern_encoding_reversed, seg_start, seg_end, window_size, padding_proportion, prob_threshold_of_no_pattern_to_mark_as_no_pattern=1):
    """Processes a single window of OHLC data to predict patterns."""
    start_index = i - math.ceil(window_size * padding_proportion)
    end_index = start_index + window_size

    start_index = max(start_index, 0)
    end_index = min(end_index, len(ohlc_data_segment))

    ohlc_segment = ohlc_data_segment[start_index:end_index]
    if len(ohlc_segment) == 0:
        return None

    win_start_date = ohlc_segment['Date'].iloc[0]
    win_end_date = ohlc_segment['Date'].iloc[-1]

    # Prepare data for Rocket model (reshape and transpose)
    ohlc_array_for_rocket = ohlc_segment[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy().reshape(1, len(ohlc_segment), 5)
    ohlc_array_for_rocket = np.transpose(ohlc_array_for_rocket, (0, 2, 1))

    try:
        pattern_probabilities = rocket_model.predict_proba(ohlc_array_for_rocket)
    except Exception as e:
        # print(f"Error in prediction for window {i}: {e}") # Optional: for debugging
        return None

    max_probability = np.max(pattern_probabilities)
    # Assuming get_pattern_encoding_by_name returns a valid index or handles errors
    no_pattern_encoding = get_pattern_encoding_by_name('No Pattern')
    if no_pattern_encoding is None: # Handle case where 'No Pattern' is not in encoding
        # print("Warning: 'No Pattern' encoding not found.") # Optional warning
        no_pattern_proba = 0 
    else:
        no_pattern_proba = pattern_probabilities[0][no_pattern_encoding]
        
    pattern_index = np.argmax(pattern_probabilities)

    pred_proba = max_probability
    pred_pattern = get_patetrn_name_by_encoding(pattern_index)

    if no_pattern_proba >= prob_threshold_of_no_pattern_to_mark_as_no_pattern: # Use >= for consistency
        pred_proba = no_pattern_proba
        pred_pattern = 'No Pattern'

    return {
        'Start': win_start_date, 'End': win_end_date, 'Chart Pattern': pred_pattern,
        'Seg_Start': seg_start, 'Seg_End': seg_end, 'Probability': pred_proba
    }

def _parallel_process_sliding_window(ohlc_data_segment, rocket_model, probability_threshold, stride, pattern_encoding_reversed, window_size, padding_proportion, prob_threshold_of_no_pattern_to_mark_as_no_pattern=1, parallel=True, num_cores=16, verbose_level=1):
    """Applies sliding window pattern detection in parallel or sequentially."""
    seg_start = ohlc_data_segment['Date'].iloc[0]
    seg_end = ohlc_data_segment['Date'].iloc[-1]

    common_args = {
        'ohlc_data_segment': ohlc_data_segment,
        'rocket_model': rocket_model,
        'probability_threshold': probability_threshold,
        'pattern_encoding_reversed': pattern_encoding_reversed,
        'window_size': window_size,
        'seg_start': seg_start,
        'seg_end': seg_end,
        'padding_proportion': padding_proportion,
        'prob_threshold_of_no_pattern_to_mark_as_no_pattern': prob_threshold_of_no_pattern_to_mark_as_no_pattern
    }

    if parallel:
        with Parallel(n_jobs=num_cores, verbose=verbose_level) as parallel_executor: # User requested verbose
            results = parallel_executor(
                delayed(_process_window)(i=i, **common_args)
                for i in range(0, len(ohlc_data_segment), stride)
            )
    else:
        results = []
        total_iterations = len(range(0, len(ohlc_data_segment), stride)) # Optional: for progress
        for i_idx, i in enumerate(range(0, len(ohlc_data_segment), stride)):
            res = _process_window(i=i, **common_args)
            if res is not None:
                results.append(res)
            if verbose_level > 0: # Basic progress for sequential
                 print(f"Processing window {i_idx + 1} of {total_iterations}...")

    return pd.DataFrame([res for res in results if res is not None])

def _prepare_dataset_for_cluster(ohlc_data_segment, win_results_df):
    """Adds position-based features to window results for clustering."""
    predicted_patterns = win_results_df.copy()

    for index, row in predicted_patterns.iterrows():
        pattern_start_date = row['Start']
        pattern_end_date = row['End']

        start_point_index = len(ohlc_data_segment[ohlc_data_segment['Date'] < pattern_start_date])
        pattern_len = len(ohlc_data_segment[(ohlc_data_segment['Date'] >= pattern_start_date) & (ohlc_data_segment['Date'] <= pattern_end_date)])
        
        pattern_mid_index = start_point_index + (pattern_len / 2.0) # Use float division
        
        predicted_patterns.at[index, 'Center'] = pattern_mid_index
        predicted_patterns.at[index, 'Pattern_Start_pos'] = start_point_index
        predicted_patterns.at[index, 'Pattern_End_pos'] = start_point_index + pattern_len
    return predicted_patterns

def _cluster_windows(predicted_patterns, probability_threshold, eps=0.05, min_samples_dbscan=2):
    """Clusters detected pattern windows using DBSCAN.
       min_samples_dbscan is the min_samples for DBSCAN algorithm itself.
       The overlap check for intersected_clusters will also use this value.
    """
    df = predicted_patterns.copy()

    if isinstance(probability_threshold, list):
        temp_dfs = []
        # Ensure probability_threshold list length matches number of encodable patterns if used directly with get_patetrn_name_by_encoding(i)
        # Or, better, iterate through unique patterns present in df if threshold list is a dict or structured differently.
        # Assuming probability_threshold list is indexed corresponding to pattern encodings from 0 to N-1
        for i, p_thresh in enumerate(probability_threshold):
            pattern_name = get_patetrn_name_by_encoding(i)
            if pattern_name: 
                 temp_dfs.append(df[(df['Chart Pattern'] == pattern_name) & (df['Probability'] >= p_thresh)])
        if temp_dfs:
            df = pd.concat(temp_dfs) if temp_dfs else pd.DataFrame(columns=df.columns)
        else: 
            df = pd.DataFrame(columns=df.columns) 
    else: # single float threshold
        df = df[df['Probability'] >= probability_threshold] # Changed > to >=

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    cluster_labled_windows_list = []
    interseced_clusters_list = []
    
    # Normalize 'Center' for DBSCAN if there's variance
    min_center_val = df['Center'].min()
    max_center_val = df['Center'].max()

    for pattern, group in df.groupby('Chart Pattern'):
        if group.empty:
            continue
            
        centers = group['Center'].values.reshape(-1, 1)
        
        if min_center_val < max_center_val: # Avoid division by zero if all centers are same
            norm_centers = (centers - min_center_val) / (max_center_val - min_center_val)
        elif len(centers) > 0 : # All centers are the same, no real distance variance
            norm_centers = np.zeros_like(centers) # Treat as single point for clustering
        else: # Empty group after filtering, should not happen if group.empty() check passed
            norm_centers = np.array([])

        if len(norm_centers) == 0: 
            group['Cluster'] = -1 
            cluster_labled_windows_list.append(group)
            continue

        current_min_samples_for_dbscan = min(min_samples_dbscan, len(norm_centers))
        if current_min_samples_for_dbscan < 1 and len(norm_centers) > 0 : 
             current_min_samples_for_dbscan = 1 
        elif len(norm_centers) == 0:
            group['Cluster'] = -1
            cluster_labled_windows_list.append(group)
            continue

        db = DBSCAN(eps=eps, min_samples=current_min_samples_for_dbscan).fit(norm_centers)
        group['Cluster'] = db.labels_
        cluster_labled_windows_list.append(group)
        
        for cluster_id, cluster_group in group[group['Cluster'] != -1].groupby('Cluster'):
            expanded_dates = []
            for _, row_cg in cluster_group.iterrows(): # Renamed 'row' to 'row_cg' to avoid conflict
                # Ensure Start and End are valid datetime objects
                try:
                    dates = pd.date_range(start=pd.to_datetime(row_cg["Start"]), end=pd.to_datetime(row_cg["End"]))
                    expanded_dates.extend(dates)
                except Exception as e:
                    # print(f"Warning: Could not create date range for row: {row_cg}. Error: {e}") # Optional
                    continue


            if not expanded_dates:
                continue

            date_counts = pd.Series(expanded_dates).value_counts().sort_index()
            
            # Use min_samples_dbscan for defining a significant overlap
            overlapping_dates = date_counts[date_counts >= min_samples_dbscan] 
            if overlapping_dates.empty:
                continue

            cluster_start = overlapping_dates.index.min()
            cluster_end = overlapping_dates.index.max()
            
            interseced_clusters_list.append({
                'Chart Pattern': pattern,
                'Cluster': cluster_id, # This ID is local to the (pattern, window_size) batch
                'Start': cluster_start,
                'End': cluster_end,
                'Seg_Start': cluster_group['Seg_Start'].iloc[0],
                'Seg_End': cluster_group['Seg_End'].iloc[0],
                'Avg_Probability': cluster_group['Probability'].mean(),
            })

    final_cluster_labled_df = pd.concat(cluster_labled_windows_list) if cluster_labled_windows_list else pd.DataFrame(columns=df.columns if not df.empty else [])
    if 'Cluster' not in final_cluster_labled_df.columns and not final_cluster_labled_df.empty:
        final_cluster_labled_df['Cluster'] = -1 # Default if no clusters formed but df had data

    final_interseced_df = pd.DataFrame(interseced_clusters_list)

    return final_cluster_labled_df, final_interseced_df

# --- Public API Function ---

def locate_patterns(ohlc_data: pd.DataFrame,
                    patterns_to_return: list = None,
                    model=None, 
                    pattern_encoding_reversed=None, 
                    win_size_proportions: list = None,
                    padding_proportion: float = PADDING_PROPORTION,
                    stride: int = STRIDE,
                    probability_threshold = None, 
                    prob_threshold_of_no_pattern_to_mark_as_no_pattern: float = PROB_THRESHOLD_NO_PATTERN,
                    dbscan_eps: float = DBSCAN_EPS,
                    dbscan_min_samples: int = DBSCAN_MIN_SAMPLES,
                    enable_plotting: bool = False,  # Keep parameter but ignore it
                    parallel_processing: bool = True,
                    num_cores_parallel: int = 16,
                    parallel_verbose_level: int = 1
                    ):
    """
    Locates financial chart patterns in OHLC data using a sliding window approach and clustering.
    """
    active_model = model if model is not None else rocket_model_global
    active_pattern_encoding_rev = pattern_encoding_reversed if pattern_encoding_reversed is not None else pattern_encoding_reversed_global
    active_win_size_proportions = win_size_proportions if win_size_proportions is not None else WIN_SIZE_PROPORTIONS
    active_probability_threshold = probability_threshold if probability_threshold is not None else PROBABILITY_THRESHOLD_LIST

    if active_model is None:
        print("Error: Pattern detection model is not loaded. Cannot proceed.")
        return pd.DataFrame()

    ohlc_data_segment = ohlc_data.copy()
    ohlc_data_segment['Date'] = pd.to_datetime(ohlc_data_segment['Date'])
    seg_len = len(ohlc_data_segment)

    if ohlc_data_segment.empty:
        return pd.DataFrame()

    win_results_for_each_size = []
    located_patterns_and_other_info_for_each_size = []
    cluster_labled_windows_list = [] # Stores all clustered windows from all iterations
    used_win_sizes = []
    global_cluster_id_offset = 0 # To ensure cluster IDs are unique across all window sizes and patterns

    for win_prop in active_win_size_proportions:
        window_size = seg_len // win_prop if win_prop > 0 else seg_len # Avoid division by zero
        window_size = int(max(10, window_size)) 

        if window_size in used_win_sizes:
            continue
        used_win_sizes.append(window_size)

        win_results_df = _parallel_process_sliding_window(
            ohlc_data_segment, active_model, active_probability_threshold, stride,
            active_pattern_encoding_rev, window_size, padding_proportion,
            prob_threshold_of_no_pattern_to_mark_as_no_pattern,
            parallel=parallel_processing, num_cores=num_cores_parallel,
            verbose_level=parallel_verbose_level # Pass verbosity
        )
        
        if win_results_df.empty:
            continue
        win_results_df['Window_Size'] = window_size
        # win_results_for_each_size.append(win_results_df) # Not directly used later, can be omitted if not needed for debugging

        predicted_patterns_for_cluster = _prepare_dataset_for_cluster(ohlc_data_segment, win_results_df)
        if predicted_patterns_for_cluster.empty:
            continue

        # Pass dbscan_min_samples to _cluster_windows
        temp_cluster_labled_windows_df, temp_interseced_clusters_df = _cluster_windows(
            predicted_patterns_for_cluster, active_probability_threshold,
            eps=dbscan_eps, min_samples_dbscan=dbscan_min_samples # Pass the parameter
        )

        if temp_cluster_labled_windows_df.empty or temp_interseced_clusters_df.empty:
            continue
        
        # Adjust cluster IDs to be globally unique before appending
        # For temp_cluster_labled_windows_df
        non_noise_clusters_mask_labeled = temp_cluster_labled_windows_df['Cluster'] != -1
        if non_noise_clusters_mask_labeled.any():
            temp_cluster_labled_windows_df.loc[non_noise_clusters_mask_labeled, 'Cluster'] = \
                temp_cluster_labled_windows_df.loc[non_noise_clusters_mask_labeled, 'Cluster'].astype(int) + global_cluster_id_offset
        
        # For temp_interseced_clusters_df
        # Note: 'Cluster' in temp_interseced_clusters_df is already filtered for non-noise by its creation logic
        if not temp_interseced_clusters_df.empty:
             temp_interseced_clusters_df['Cluster'] = temp_interseced_clusters_df['Cluster'].astype(int) + global_cluster_id_offset
        
        current_max_cluster_id_in_batch = -1
        if not temp_interseced_clusters_df.empty and 'Cluster' in temp_interseced_clusters_df.columns:
            valid_clusters = temp_interseced_clusters_df[temp_interseced_clusters_df['Cluster'] != -1]['Cluster']
            if not valid_clusters.empty:
                 current_max_cluster_id_in_batch = valid_clusters.max()
        
        cluster_labled_windows_list.append(temp_cluster_labled_windows_df)
        
        temp_interseced_clusters_df['Calc_Start'] = temp_interseced_clusters_df['Start']
        temp_interseced_clusters_df['Calc_End'] = temp_interseced_clusters_df['End']
        located_patterns_info = temp_interseced_clusters_df.copy()
        located_patterns_info['Window_Size'] = window_size
        located_patterns_and_other_info_for_each_size.append(located_patterns_info)

        if current_max_cluster_id_in_batch > -1 :
            global_cluster_id_offset = current_max_cluster_id_in_batch + 1
        elif non_noise_clusters_mask_labeled.any(): # If intersected was empty but labeled had clusters
            max_labeled_cluster = temp_cluster_labled_windows_df.loc[non_noise_clusters_mask_labeled, 'Cluster'].max()
            global_cluster_id_offset = max_labeled_cluster + 1


    if not located_patterns_and_other_info_for_each_size:
        return pd.DataFrame()

    all_located_patterns_df = pd.concat(located_patterns_and_other_info_for_each_size, ignore_index=True)
    if all_located_patterns_df.empty:
        return pd.DataFrame()

    # Filter overlapping patterns (logic remains similar to previous version)
    unique_chart_patterns = all_located_patterns_df['Chart Pattern'].unique()
    # Sort window sizes descending to prioritize larger windows
    sorted_unique_window_sizes = np.sort(all_located_patterns_df['Window_Size'].unique())[::-1] 

    final_filtered_patterns_list = []
    # Use a copy and mark 'taken' to handle overlaps systematically
    candidate_patterns_df = all_located_patterns_df.copy()
    # Ensure 'taken' column exists, default to False
    if 'taken' not in candidate_patterns_df.columns:
        candidate_patterns_df['taken'] = False
    else: # if it somehow exists from a previous run (unlikely with .copy()), reset it
        candidate_patterns_df['taken'] = False


    for cp_val in unique_chart_patterns:
        for ws_val in sorted_unique_window_sizes:
            # Select current batch of patterns to consider
            current_batch_indices = candidate_patterns_df[
                (candidate_patterns_df['Chart Pattern'] == cp_val) &
                (candidate_patterns_df['Window_Size'] == ws_val) &
                (~candidate_patterns_df['taken'])
            ].index

            for current_idx in current_batch_indices:
                if candidate_patterns_df.loc[current_idx, 'taken']: # Already claimed by a higher priority pattern
                    continue

                current_row_data = candidate_patterns_df.loc[current_idx]
                final_filtered_patterns_list.append(current_row_data.drop('taken')) # Add to final list
                candidate_patterns_df.loc[current_idx, 'taken'] = True # Mark as taken

                # Now, check for overlaps with other non-taken patterns and invalidate lower-priority ones
                # Lower priority: smaller window, or same window but processed later (which this loop structure handles),
                # or significantly lower probability.
                overlapping_candidates_indices = candidate_patterns_df[
                    (candidate_patterns_df.index != current_idx) & # Don't compare with itself
                    (candidate_patterns_df['Chart Pattern'] == cp_val) &
                    (~candidate_patterns_df['taken']) &
                    (candidate_patterns_df['Calc_Start'] <= current_row_data['Calc_End']) &
                    (candidate_patterns_df['Calc_End'] >= current_row_data['Calc_Start'])
                ].index

                for ov_idx in overlapping_candidates_indices:
                    ov_row_data = candidate_patterns_df.loc[ov_idx]
                    iou = intersection_over_union(current_row_data['Calc_Start'], current_row_data['Calc_End'],
                                                  ov_row_data['Calc_Start'], ov_row_data['Calc_End'])
                    if iou > 0.6: # Significant overlap
                        # current_row_data (from larger/earlier window) is preferred by default.
                        # ov_row_data (overlapping candidate) is discarded UNLESS:
                        # it's from a smaller window AND has significantly higher probability.
                        is_ov_preferred = (ov_row_data['Window_Size'] < current_row_data['Window_Size']) and \
                                          ((ov_row_data['Avg_Probability'] - current_row_data['Avg_Probability']) > 0.1)
                        
                        if not is_ov_preferred:
                            candidate_patterns_df.loc[ov_idx, 'taken'] = True
                        # If ov_preferred, current_row_data was already added. The ov_row will be considered
                        # when its (smaller) window size turn comes, if not already taken.
                        # This implies a potential issue: if current_row is added, and a smaller, much better ov_row exists,
                        # current_row should ideally be removed. The current logic adds current_row first.
                        # For a more robust selection, decisions might need to be deferred or involve pairwise ranking.
                        # However, given the descending window size iteration, this greedy choice is often sufficient.
                        # Re-evaluating this complex interaction:
                        # If current_row (larger window) is chosen, and an ov_row (smaller window, much higher prob) exists,
                        # the current logic keeps current_row and marks ov_row as NOT taken, so ov_row can be picked later.
                        # This might lead to both being in the list if their IoU with *other* patterns doesn't disqualify them.
                        # The final drop_duplicates will handle exact overlaps.

    filtered_loc_pat_and_info_df = pd.DataFrame(final_filtered_patterns_list)
    if not filtered_loc_pat_and_info_df.empty:
         # Drop duplicates based on the defining characteristics of a pattern instance
         filtered_loc_pat_and_info_df = filtered_loc_pat_and_info_df.sort_values(
             by=['Chart Pattern', 'Calc_Start', 'Window_Size', 'Avg_Probability'], 
             ascending=[True, True, False, False] # Prioritize larger window, then higher prob for duplicates
         ).drop_duplicates(
             subset=['Chart Pattern', 'Calc_Start', 'Calc_End'], 
             keep='first' # Keep the one that came first after sorting (best according to sort)
         ).sort_values(by='Calc_Start').reset_index(drop=True)


    if enable_plotting and not filtered_loc_pat_and_info_df.empty and cluster_labled_windows_list:
        # Remove plotting code
        pass

    if patterns_to_return and not filtered_loc_pat_and_info_df.empty:
        return filtered_loc_pat_and_info_df[filtered_loc_pat_and_info_df['Chart Pattern'].isin(patterns_to_return)]
    
    return filtered_loc_pat_and_info_df

