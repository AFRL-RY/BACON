# confidence_errors.py

# Note: ACE and ECE are 'calibration' errors, not 'confidence' errors.  'Confidence' was used by mistake.  We chose not to change the name to avoid introducing programming errors.  

import numpy as np
import pandas as pd

def get_confidence_errors(measure_df,class_count):

    m = class_count-1 
    bins = np.linspace(0,1,m)

    Acc = []
    avg_conf = []
    lg = []    

    for j in range(len(bins)-1):
        if j != len(bins) -2:
            new_df = measure_df[(measure_df["Confidence"] >= bins[j]) & (measure_df["Confidence"] < bins[j+1])].copy()
        else:
            new_df = measure_df[(measure_df["Confidence"] >= bins[j]) & (measure_df["Confidence"] <= bins[j+1])].copy()
        
        lgth = len(new_df)
        lg.append(lgth)
        if lgth == 0:
            Acc.append(np.nan)
            avg_conf.append(np.nan)
        else:
            Acc.append(sum(new_df["Accuracy"])/lgth)
            avg_conf.append(np.mean(new_df["Confidence"]))

    diff = [np.abs(Acc[k]-avg_conf[k]) for k in range(len(Acc))]
    diff_df = pd.DataFrame(diff,columns=['Values'])

    df = np.squeeze(diff_df)

    lg_copy = lg.copy()
    run = lg_copy.count(0)

    for j in range(run):
        lg_copy.remove(0)

    ECE = np.average(np.squeeze(diff_df['Values'].dropna()),weights=lg_copy)
    MCE = diff_df['Values'].dropna().max()
    MCE_idx = diff_df['Values'].idxmax(axis=0)
    MCE_count = lg[MCE_idx]
    Max_count = max(lg)
    Min_count = min(lg)

    return ECE, MCE, MCE_count, Max_count, Min_count

def get_ace_confidence(epsilon_threshold, class_count, bin_count, metrics_df, header_stem):
    ACE_L1_SUM = 0.0
    class_MCE = []
    for selected_class in range(class_count):

        ACE_L1_CLASS = 0.0
        
        metrics_frame = pd.DataFrame(columns=["Label","Accuracy","Confidence"])
        metrics_header = header_stem + str(selected_class)
        for i,item in enumerate(metrics_df.iloc()):
            if item['Label'] == selected_class:
                accur = 1
            else:
                accur = 0

            conf = item[header_stem + str(selected_class)]
            metrics_frame.loc[i] = [item['Label'],accur,conf]
        
        #metrics_frame['Confidence'][metrics_frame['Confidence'] < epsilon_threshold] = np.nan
        metrics_frame.loc[metrics_frame['Confidence'] < epsilon_threshold, 'Confidence'] = np.nan
        metrics_frame = metrics_frame.dropna()
        
        bin_labels = list(np.arange(0,bin_count))
        metrics_frame['qcut_bin'] = pd.qcut(metrics_frame['Confidence'].rank(method='first'),q=bin_count,labels = bin_labels)

        lgth_metric = [] # Count of item in the bin
        Acc_metric = []  # Accuracy of the bin
        Conf_metric = [] # mean confidence of the bin

        for i in range(bin_count):
            bin_df = metrics_frame[metrics_frame['qcut_bin']==i]
            lgth_metric.append(len(bin_df))
            Acc_metric.append(bin_df['Accuracy'].mean())
            Conf_metric.append(bin_df['Confidence'].mean())
    
        Acc_metric_arr = np.array(Acc_metric)
        Conf_metric_arr = np.array(Conf_metric)


        ACE_L1_class = np.sum(np.abs(Acc_metric_arr - Conf_metric_arr))
        ACE_L1_SUM = ACE_L1_SUM + ACE_L1_class
    
        # Here do class MCE
        diff_ace = Acc_metric_arr-Conf_metric_arr
        diff_ace = diff_ace[~np.isnan(diff_ace)]
        class_MCE.append(diff_ace.max())        


    ACE = ACE_L1_SUM/(class_count * bin_count)
        
    return ACE
