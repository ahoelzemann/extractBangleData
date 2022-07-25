import math

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
    timestamps = [get_median_string(window) for window in timestamp_windows]
    return means.set_axis(timestamps)


def get_median_string(local_list):
    return local_list[int(len(local_list)/2)]

def calc_magnitude(sample):

    return np.abs(math.sqrt(math.pow(sample[0], 2) + math.pow(sample[1], 2) + math.pow(sample[2], 2)))

def calc_magnitude_wrapper(dataframe: pd.DataFrame):
    signal = dataframe.loc[:, ['acc_x', 'acc_y', 'acc_z']].to_numpy()
    dataframe['magnitude'] = list(map(calc_magnitude, signal))

    return dataframe

def cut_df_at_timestamps(df, start, end):
    from datetime import datetime
    from bisect import bisect_left

    firstTimestamp = np.datetime64(datetime.strptime(df['timestamp'].iloc[0], '%Y-%m-%d %H:%M:%S.%f'))

    minutes_start, seconds_start = start.split(":")
    minutes_end, seconds_end = end.split(":")
    timedelta_start = pd.Timedelta(minutes=int(minutes_start), seconds=int(seconds_start)).to_timedelta64()
    timedelta_end = pd.Timedelta(minutes=int(minutes_end), seconds=int(seconds_end)).to_timedelta64()
    start_time = firstTimestamp + timedelta_start
    end_time = firstTimestamp + timedelta_end
    timestamps = pd.to_datetime(df['timestamp'])
    start_index = bisect_left(timestamps, start_time) - 1
    end_index = bisect_left(timestamps, end_time) - 1
    final_chunk = df.iloc[start_index:end_index]
    # start = datetime.strptime(start, '%M:%S')
    # end = datetime.strptime(end, '%M:%S')
    return final_chunk

def calc_fft(mag):
    from scipy.fft import fft
    mag = mag.to_numpy()
    signalFFT = fft(mag)
    signalPSD = np.abs(signalFFT)[1:101]
    dominant_freq_index = np.where(signalPSD == np.amax(signalPSD))[0]
    signal_to_noise_ratio = signalPSD[dominant_freq_index] / np.mean(np.delete(signalPSD, dominant_freq_index))
    return signalPSD, signal_to_noise_ratio[0]