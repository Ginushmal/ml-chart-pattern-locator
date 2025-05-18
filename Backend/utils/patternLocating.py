import joblib
from tqdm import tqdm
from utils.eval import intersection_over_union
from utils.formatAndPreprocessNewPatterns import get_patetrn_name_by_encoding, get_pattern_encoding_by_name, get_reverse_pattern_encoding
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import math
from sklearn.cluster import DBSCAN

from utils.functionalPatternLocateAndPlot import plot_pattern_groups_and_finalized_sections

path = 'Datasets/OHLC data'



def process_window(i, ohlc_data_segment, rocket_model, probability_threshold, pattern_encoding_reversed,seg_start, seg_end, window_size, padding_proportion,prob_threshold_of_no_pattern_to_mark_as_no_pattern=1):
    start_index = i - math.ceil(window_size * padding_proportion)
    end_index = start_index + window_size

    start_index = max(start_index, 0)
    end_index = min(end_index, len(ohlc_data_segment))

    ohlc_segment = ohlc_data_segment[start_index:end_index]
    if len(ohlc_segment) == 0:
        return None  # Skip empty segments
    win_start_date = ohlc_segment['Date'].iloc[0]
    win_end_date = ohlc_segment['Date'].iloc[-1]
    
    # print("ohlc befor :" , ohlc_segment)
    ohlc_array_for_rocket = ohlc_segment[['Open', 'High', 'Low', 'Close','Volume']].to_numpy().reshape(1, len(ohlc_segment), 5)
    ohlc_array_for_rocket = np.transpose(ohlc_array_for_rocket, (0, 2, 1))
    # print( "ohlc for rocket :" , ohlc_array_for_rocket)
    try:
        pattern_probabilities = rocket_model.predict_proba(ohlc_array_for_rocket)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
    max_probability = np.max(pattern_probabilities)
    # print(pattern_probabilities)
    # print(f"Predicted Pattern: {pattern_encoding_reversed[np.argmax(pattern_probabilities)]} with probability: {max_probability} in num {i} window")
    # if max_probability > probability_threshold:
    no_pattern_proba = pattern_probabilities[0][get_pattern_encoding_by_name ('No Pattern')]
    pattern_index = np.argmax(pattern_probabilities)
    
    pred_proba = max_probability
    pred_pattern = get_patetrn_name_by_encoding(pattern_index)
    if no_pattern_proba > prob_threshold_of_no_pattern_to_mark_as_no_pattern:
        pred_proba = no_pattern_proba
        pred_pattern = 'No Pattern'
    
    new_row = {
        'Start': win_start_date, 'End': win_end_date,  'Chart Pattern': pred_pattern,  'Seg_Start': seg_start, 'Seg_End': seg_end ,
        'Probability': pred_proba
    }
    # plot_patterns_for_segment(test_seg_id, pd.DataFrame([new_row]), ohlc_data_segment)
    return new_row
    # return None



def parallel_process_sliding_window(ohlc_data_segment, rocket_model, probability_threshold, stride, pattern_encoding_reversed, window_size, padding_proportion,prob_threshold_of_no_pattern_to_mark_as_no_pattern=1,parallel=True,num_cores=-1):
    # get the start and end dates of the ohlc data 
    seg_start = ohlc_data_segment['Date'].iloc[0]
    seg_end = ohlc_data_segment['Date'].iloc[-1]

    if parallel:
        # Use Parallel as a context manager to ensure cleanup
        with Parallel(n_jobs=num_cores,verbose = 1) as parallel:
            results = parallel(
                delayed(process_window)(
                    i=i,
                    ohlc_data_segment=ohlc_data_segment,
                    rocket_model=rocket_model,
                    probability_threshold=probability_threshold,
                    pattern_encoding_reversed=pattern_encoding_reversed,
                    window_size=window_size,
                    seg_start=seg_start,
                    seg_end=seg_end,
                    padding_proportion=padding_proportion,
                    prob_threshold_of_no_pattern_to_mark_as_no_pattern=prob_threshold_of_no_pattern_to_mark_as_no_pattern
                )

                for i in range(0, len(ohlc_data_segment), stride)
            )

        # print(f"Finished processing segment {seg_id} for symbol {symbol}")
        # print(results)
        # Filter out None values and create DataFrame
        return pd.DataFrame([res for res in results if res is not None])
    else:
    
        #  do the sam e thing without parrellel processing
        results = []
        total_iterations = len(range(0, len(ohlc_data_segment), stride))
        for i_idx, i in enumerate(range(0, len(ohlc_data_segment), stride)):
            res = process_window(i, ohlc_data_segment, rocket_model, probability_threshold, pattern_encoding_reversed, seg_start, seg_end, window_size, padding_proportion)
            if res is not None:
                results.append(res)
            # Progress print statement
            print(f"Processing window {i_idx + 1} of {total_iterations}...")
        return pd.DataFrame(results)

            
def prepare_dataset_for_cluster(ohlc_data_segment, win_results_df):

    predicted_patterns = win_results_df.copy()
    origin_date = ohlc_data_segment['Date'].min()
    for index, row in predicted_patterns.iterrows():
        pattern_start = row['Start']
        pattern_end = row['End']
        
        #  get the number of OHLC data points from the origin date to the pattern start date
        start_point_index = len(ohlc_data_segment[ohlc_data_segment['Date'] < pattern_start])
        pattern_len = len(ohlc_data_segment[(ohlc_data_segment['Date'] >= pattern_start) & (ohlc_data_segment['Date'] <= pattern_end)])
        
        pattern_mid_index = start_point_index + (pattern_len / 2)
        
        # add the center index to a new column Center in the predicted_patterns current row
        predicted_patterns.at[index, 'Center'] = pattern_mid_index
        predicted_patterns.at[index, 'Pattern_Start_pos'] = start_point_index
        predicted_patterns.at[index, 'Pattern_End_pos'] = start_point_index + pattern_len

    return predicted_patterns
     
def cluster_windows(predicted_patterns , probability_threshold, window_size,eps = 0.05 , min_samples = 2):
    df = predicted_patterns.copy()

    # check if the probability_threshold is a list or a float
    if isinstance(probability_threshold, list):
        # the list contain the probability thresholds for each chart pattern 
        # filter the dataframe for each probability threshold
        for i in range(len(probability_threshold)):
            pattern_name = get_patetrn_name_by_encoding(i)
            df.drop(df[(df['Chart Pattern'] == pattern_name) & (df['Probability'] < probability_threshold[i])].index, inplace=True)
            # print(f"Filtered {pattern_name} with probability < {probability_threshold[i]}")

            
    else:
        # only get the rows that has a probability greater than the probability threshold
        df = df[df['Probability'] > probability_threshold]

    # Initialize a list to store merged clusters from all groups
    cluster_labled_windows = []
    interseced_clusters = []
    
    min_center = df['Center'].min()
    max_center = df['Center'].max()

    # Group by 'Chart Pattern' and apply clustering to each group
    for pattern, group in df.groupby('Chart Pattern'):
        # print (pattern)
        # print(group)
        # Clustering
        centers = group['Center'].values.reshape(-1, 1)
        
        # centers normalization
        if min_center < max_center:  # Avoid division by zero
            norm_centers = (centers - min_center) / (max_center - min_center)
        else:
            # If all values are the same, set to constant (e.g., 0 or 1)
            norm_centers = np.ones_like(centers)
        
        # eps  =window_size/2 + 4
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(norm_centers)
        group['Cluster'] = db.labels_
        
        cluster_labled_windows.append(group)
        
        # Filter out noise (-1) and group by Cluster
        for cluster_id, cluster_group in group[group['Cluster'] != -1].groupby('Cluster'):

            
            expanded_dates = []
            for _, row in cluster_group.iterrows():
                # Print the start and end dates for debugging
                dates = pd.date_range(row["Start"], row["End"])
                expanded_dates.extend(dates)

            # print("Total expanded dates:", len(expanded_dates))


            # Step 2: Count occurrences of each date
            date_counts = pd.Series(expanded_dates).value_counts().sort_index()

            # Step 3: Identify cluster start and end (where at least 2 windows overlap)
            cluster_start = date_counts[date_counts >= 2].index.min()
            cluster_end = date_counts[date_counts >= 2].index.max()
            
            interseced_clusters.append({
                # 'Seg_ID' : cluster_group['Seg_ID'].iloc[0],
                # 'Symbol' : cluster_group['Symbol'].iloc[0],
                'Chart Pattern': pattern,
                'Cluster': cluster_id,
                'Start': cluster_start,
                'End': cluster_end,
                'Seg_Start': cluster_group['Seg_Start'].iloc[0],
                'Seg_End': cluster_group['Seg_End'].iloc[0],
                'Avg_Probability': cluster_group['Probability'].mean(),
            })

    if len(cluster_labled_windows) == 0 or len(interseced_clusters) == 0:
        return None,None
    # # Combine all merged clusters into a final DataFrame
    cluster_labled_windows_df = pd.concat(cluster_labled_windows)
    interseced_clusters_df = pd.DataFrame(interseced_clusters)

    # sort by the index 
    cluster_labled_windows_df = cluster_labled_windows_df.sort_index()
    # print(cluster_labled_windows_df)
    # Display the result
    # print(merged_df)
    return cluster_labled_windows_df,interseced_clusters_df


# =========================Advance Locator ==========================

pattern_encoding_reversed = get_reverse_pattern_encoding()
# load the joblib model at Models\Width Aug OHLC_mini_rocket_xgb.joblib to use 
model =  joblib.load('Models/Width Aug OHLC_mini_rocket_xgb.joblib')
plot_count = 0


win_size_proportions = np.round(np.logspace(0, np.log10(20), num=10), 2).tolist()
padding_proportion = 0.6
stride = 1
probability_threshold = 0.4
# probab_threshold_list =[0.8, 0.8, 0.5, 0.5, 0.5, 0.8, 0.7]
probab_threshold_list =[0.6, 0.6, 0.4, 0.4, 0.4, 0.6, 0.5]

prob_threshold_of_no_pattern_to_mark_as_no_pattern = 0.5
target_len = 30

eps=0.04 # in the dbscan clustering
min_samples=3 # in the dbscan clustering
win_width_proportion=10 # in the dbscan clustering from what amount to divide the width related feature


def locate_patterns(ohlc_data, patterns_to_return= None,model = model , pattern_encoding_reversed= pattern_encoding_reversed,plot_count = 10):
    ohlc_data_segment = ohlc_data.copy()
    # convert date to datetime
    ohlc_data_segment['Date'] = pd.to_datetime(ohlc_data_segment['Date'])
    seg_len = len(ohlc_data_segment)
    
    if ohlc_data_segment is None or len(ohlc_data_segment) == 0:
        print("OHLC Data segment is empty")
        raise Exception("OHLC Data segment is empty")  

    win_results_for_each_size = []
    located_patterns_and_other_info_for_each_size = []
    cluster_labled_windows_list = []

    used_win_sizes = []
    win_iteration = 0

    for win_size_proportion in win_size_proportions:

        window_size = seg_len // win_size_proportion
        # print(f"Win size : {window_size}")
        if window_size < 10:
            window_size = 10
        # elif window_size > 30:
        #     window_size = 30
            
        # convert to int 
        window_size = int(window_size)
        if window_size in used_win_sizes:
            continue
        used_win_sizes.append(window_size)
   
        # win_results_df = parallel_process_sliding_window(ohlc_data_segment, model, probability_threshold,stride, pattern_encoding_reversed,group,test_seg_id,window_size, padding_proportion, len_norm, target_len)
        win_results_df = parallel_process_sliding_window(ohlc_data_segment, model, probab_threshold_list,stride, pattern_encoding_reversed,window_size, padding_proportion,prob_threshold_of_no_pattern_to_mark_as_no_pattern,parallel=True)
        
        if win_results_df is None or len(win_results_df) == 0:
            print("Window results dataframe is empty")
            continue
        win_results_df['Window_Size'] = window_size
        win_results_for_each_size.append(win_results_df)
        # plot_sliding_steps(win_results_df ,ohlc_data_segment,probability_threshold ,test_seg_id)
        predicted_patterns = prepare_dataset_for_cluster(ohlc_data_segment, win_results_df)
        if predicted_patterns is None or len(predicted_patterns) == 0:
            print("Predicted patterns dataframe is empty")
        # print("Predicted Patterns :",predicted_patterns)
        # cluster_labled_windows_df , interseced_clusters_df = cluster_windows(predicted_patterns, probability_threshold, window_size)
        cluster_labled_windows_df , interseced_clusters_df = cluster_windows(predicted_patterns, probab_threshold_list, window_size)
        if cluster_labled_windows_df is None or interseced_clusters_df is None or len(cluster_labled_windows_df) == 0 or len(interseced_clusters_df) == 0:
            print("Clustered windows dataframe is empty")
            continue
        mask = cluster_labled_windows_df['Cluster'] != -1
        cluster_labled_windows_df.loc[mask, 'Cluster'] = cluster_labled_windows_df.loc[mask, 'Cluster'].astype(int) + win_iteration
        # mask2 = interseced_clusters_df['Cluster'] != -1
        interseced_clusters_df['Cluster'] = interseced_clusters_df['Cluster'].astype(int) + win_iteration
        num_of_unique_clusters = interseced_clusters_df[interseced_clusters_df['Cluster']!=-1]['Cluster'].nunique()
        win_iteration += num_of_unique_clusters 
        cluster_labled_windows_list.append(cluster_labled_windows_df)
        # located_patterns_and_other_info = functional_pattern_filter_and_point_recognition(interseced_clusters_df)
        interseced_clusters_df['Calc_Start'] = interseced_clusters_df['Start']
        interseced_clusters_df['Calc_End'] = interseced_clusters_df['End']
        located_patterns_and_other_info = interseced_clusters_df.copy()

        if located_patterns_and_other_info is None or len(located_patterns_and_other_info) == 0:
            print("]Located patterns and other info dataframe is empty")
            continue
        # plot_pattern_groups_and_finalized_sections(located_patterns_and_other_info, cluster_labled_windows_df, test_seg_id)
        located_patterns_and_other_info['Window_Size'] = window_size
        
        located_patterns_and_other_info_for_each_size.append(located_patterns_and_other_info)
        
    if located_patterns_and_other_info_for_each_size is None or len(located_patterns_and_other_info_for_each_size) == 0 or win_results_for_each_size is None or len(win_results_for_each_size) == 0:
        print("Located patterns and other info for each size is empty")
        return None
    located_patterns_and_other_info_for_each_size_df = pd.concat(located_patterns_and_other_info_for_each_size)
    win_results_for_each_size_df = pd.concat(win_results_for_each_size, ignore_index=True)
    # window_results_list.append(win_results_for_each_size_df)

    # get the set of unique window sizes from located_patterns_and_other_info_for_each_size_df
    unique_window_sizes = located_patterns_and_other_info_for_each_size_df['Window_Size'].unique()
    unique_patterns = located_patterns_and_other_info_for_each_size_df['Chart Pattern'].unique()    

    # sort the unique_window_sizes descending order
    unique_window_sizes = np.sort(unique_window_sizes)[::-1]

    filtered_loc_pat_and_info_rows_list = []

    for chart_pattern in unique_patterns:    
        located_patterns_and_other_info_for_each_size_df_chart_pattern = located_patterns_and_other_info_for_each_size_df[located_patterns_and_other_info_for_each_size_df['Chart Pattern'] == chart_pattern]
        for win_size in unique_window_sizes:
            located_patterns_and_other_info_for_each_size_df_win_size_chart_pattern = located_patterns_and_other_info_for_each_size_df_chart_pattern[located_patterns_and_other_info_for_each_size_df_chart_pattern['Window_Size'] == win_size]
            for idx , row in located_patterns_and_other_info_for_each_size_df_win_size_chart_pattern.iterrows():
                start_date = row['Calc_Start']
                end_date = row['Calc_End']
                is_already_included = False
                # check if there are any other rows that intersect with the start and end dates with the same chart pattern
                intersecting_rows = located_patterns_and_other_info_for_each_size_df_chart_pattern[
                                                    (located_patterns_and_other_info_for_each_size_df_chart_pattern['Calc_Start'] <= end_date) &
                                                    (located_patterns_and_other_info_for_each_size_df_chart_pattern['Calc_End'] >= start_date)
                                                ]
                is_already_included = False
                for idx2, row2 in intersecting_rows.iterrows():
                    iou = intersection_over_union(start_date, end_date, row2['Calc_Start'], row2['Calc_End'])

                    if iou > 0.6:
                        # Case 1: Larger window already exists
                        if row2['Window_Size'] > row['Window_Size']:
                            # Case 1A: But smaller one has significantly higher probability, keep it instead
                            if (row['Avg_Probability'] - row2['Avg_Probability']) > 0.1:
                                is_already_included = False
                            else:
                                is_already_included = True
                                break  # Keep large, skip current(small)

                        # Case 2: Equal or smaller window exists, possibly overlapping
                        elif row['Window_Size'] >= row2['Window_Size']:
                            # If current row has significantly better probability, replace existing
                            if (row2['Avg_Probability'] - row['Avg_Probability']) > 0.1:
                                is_already_included = True
                                break  # remove current (large) , keep small
                            else:
                                is_already_included = False
                                # break

                if not is_already_included:
                    filtered_loc_pat_and_info_rows_list.append(row)


    # convert the filtered_loc_pat_and_info_rows_list to a dataframe
    filtered_loc_pat_and_info_df = pd.DataFrame(filtered_loc_pat_and_info_rows_list)
    # located_patterns_and_other_info_list.append(filtered_loc_pat_and_info_df) 

    if cluster_labled_windows_list is None or len(cluster_labled_windows_list) == 0:
        print("Clustered windows list is empty")
    cluster_labled_windows_df_conc = pd.concat(cluster_labled_windows_list)
    if plot_count > 0:
        plot_pattern_groups_and_finalized_sections(filtered_loc_pat_and_info_df, cluster_labled_windows_df_conc,ohcl_data_given=ohlc_data_segment)    
    plot_count -= 1              

    if patterns_to_return is None or len(patterns_to_return) == 0:
        return filtered_loc_pat_and_info_df
    else :
        # filter the filtered_loc_pat_and_info_df based on the patterns_to_return
        filtered_loc_pat_and_info_df = filtered_loc_pat_and_info_df[filtered_loc_pat_and_info_df['Chart Pattern'].isin(patterns_to_return)]
        return filtered_loc_pat_and_info_df
    
    