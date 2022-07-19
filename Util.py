import numpy as np
import pandas as pd

def mean_filter(df, window_size):
    # df = df.set_index('timestamp')
    sample_windows = []
    timestamp_windows = []
    for n in range(0, df.shape[0], window_size):
        sample_window = []
        timestamp_window = []
        for k in range(0, window_size):
            try:
                sample = df.iloc[n+k]
                sample_window.append(sample['acc_z'])
                timestamp_window.append(sample['timestamp'])
            except:
                break
        sample_windows.append(sample_window)
        timestamp_windows.append(timestamp_window)
    means = pd.Series([np.mean(window) for window in sample_windows])
    timestamps = [get_medien_string(window) for window in timestamp_windows]
    return means.set_axis(timestamps)


def get_medien_string(local_list):
    return local_list[int(len(local_list)/2)]