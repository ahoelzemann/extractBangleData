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
                sample = df['magnitude'].iloc[n + k]
                index = df['timestamp'].iloc[n + k]
                sample_window.append(sample)
                timestamp_window.append(index)
            except:
                break
        sample_windows.append(sample_window)
        timestamp_windows.append(timestamp_window)
    means = pd.Series([np.mean(window) for window in sample_windows])
    timestamps = [get_median_string(window) for window in timestamp_windows]
    return means.set_axis(timestamps)


def get_median_string(local_list):
    return local_list[int(len(local_list) / 2)]


def calc_magnitude(sample):
    return np.abs(math.sqrt(math.pow(sample[0], 2) + math.pow(sample[1], 2) + math.pow(sample[2], 2)))


def calc_magnitude_wrapper(dataframe):
    for subject in dataframe.players:
        signal = dataframe.players[subject]['data'].loc[:, ['acc_x', 'acc_y', 'acc_z']].to_numpy()
        dataframe.players[subject]['data']['magnitude'] = list(map(calc_magnitude, signal))

    return dataframe

def calc_pca_window(window):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca = pca.fit(window)
    PCA(n_components=2)

    return pca

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
    signalPSD = np.abs(signalFFT)[1:61]
    dominant_freq_index = np.where(signalPSD == np.amax(signalPSD))[0]
    signal_to_noise_ratio = signalPSD[dominant_freq_index] / np.mean(np.delete(signalPSD, dominant_freq_index))
    return signalPSD, signal_to_noise_ratio[0]


def get_peaks(activity, df, printing=False):
    from scipy.signal import find_peaks
    import viz
    import numpy as np
    from scipy.signal import savgol_filter
    # df = max(df, key=len).reset_index().loc[:, ['timestamp', 'acc_z']]
    # df = max(df, key=len).reset_index()
    filtered_signal = mean_filter(df, 10)
    n_dribblings = None
    # filtered_signal = pd.Series(savgol_filter(df.loc[:, 'acc_z'], 100, 1))
    peaks, _ = find_peaks(filtered_signal, prominence=1.4)

    y_val_peaks = np.full(fill_value=np.nan, shape=(filtered_signal.shape[0]))
    y_val_peaks[peaks] = filtered_signal.iloc[peaks]
    y_val_peaks = pd.Series(y_val_peaks, index=filtered_signal.index)
    y_val_peaks = y_val_peaks.dropna()
    if activity == 'dribbling':
        n_dribblings = _calc_dribblings_per_second( y_val_peaks)

    # if printing:
    #     fig = viz.plot_peaks(filtered_signal, y_val_peaks, n_dribblings, subject)
    return y_val_peaks, n_dribblings, filtered_signal


def _calc_dribblings_per_second(y_val_peaks):
    import numpy as np
    from datetime import datetime

    peak_indices = y_val_peaks.index.values
    last_second = datetime.strptime(peak_indices[0], '%Y-%m-%d %H:%M:%S.%f').second
    counter = 1
    dribblings_per_second = []
    for i in range(1, len(peak_indices)):
        current_second = datetime.strptime(peak_indices[i], '%Y-%m-%d %H:%M:%S.%f').second
        if current_second == last_second:
            counter = counter + 1
        else:
            dribblings_per_second.append(counter)
            counter = 1
        last_second = current_second
        # print()

    return np.mean(dribblings_per_second)

def gather_plots(hangtime_si, hangtime_bo, participants, activity='complete_signal'):

    import viz
    experts = participants[0]
    novices = participants[1]
    plots_experts = {}
    plots_novices = {}

    # box plots std and mean
    for expert in experts:
        expert, location = expert.split("_")
        plots_experts[expert + '_' + location] = {}
        if location == "eu":
            raw_data = hangtime_si.players[expert]['data'][['timestamp', 'acc_x', 'acc_y', 'acc_z']][
                        hangtime_si.players[expert]['features'][activity]['start_index']:
                        hangtime_si.players[expert]['features'][activity]['end_index']]
            # magnitude = hangtime_si.players[expert]['data'][['timestamp', 'magnitude']][
            #             hangtime_si.players[expert]['features'][activity]['start_index']:
            #             hangtime_si.players[expert]['features'][activity]['end_index']]
            # magnitude = magnitude.set_index('timestamp')
            fft = hangtime_si.players[expert]['features'][activity]['fft']
            snr = hangtime_si.players[expert]['features'][activity]['snr']
            if activity != 'complete_signal':
                filtered_magnitude = hangtime_si.players[expert]['features'][activity]['filtered_magnitude']
            if activity == 'dribbling':
                n_dribblings = hangtime_si.players[expert]['features'][activity]['dribblings_per_seconds']
                peaks = hangtime_si.players[expert]['features'][activity]['signal_peaks']
        else:
            raw_data = hangtime_bo.players[expert]['data'][['timestamp', 'acc_x', 'acc_y', 'acc_z']][
                       hangtime_bo.players[expert]['features'][activity]['start_index']:
                       hangtime_bo.players[expert]['features'][activity]['end_index']]
            # magnitude = hangtime_bo.players[expert]['data'][['timestamp', 'magnitude']][
            #             hangtime_bo.players[expert]['features'][activity]['start_index']:
            #             hangtime_bo.players[expert]['features'][activity]['end_index']]
            # magnitude = magnitude.set_index('timestamp')

            fft = hangtime_bo.players[expert]['features'][activity]['fft']
            snr = hangtime_bo.players[expert]['features'][activity]['snr']
            if activity != 'complete_signal':
                filtered_magnitude = hangtime_bo.players[expert]['features'][activity]['filtered_magnitude']
            if activity == 'dribbling':
                n_dribblings = hangtime_bo.players[expert]['features'][activity]['dribblings_per_seconds']
                peaks = hangtime_bo.players[expert]['features'][activity]['signal_peaks']
        plots_experts[expert + '_' + location]['raw'] = viz.vizualize_one_player(raw_data, False)
        plots_experts[expert + '_' + location]['std_mean'] = viz.plot_std_mean_box_plot(filtered_magnitude)
        plots_experts[expert + '_' + location]['fft'] = viz.plot_fft(fft, snr, expert)
        if activity == 'dribbling':
            plots_experts[expert + '_' + location]['peaks'] = viz.plot_peaks(filtered_magnitude, peaks, n_dribblings)
    for novice in novices:
        novice, location = novice.split("_")
        plots_novices[novice + '_' + location] = {}
        if location == "eu":
            raw_data = hangtime_si.players[novice]['data'][['timestamp', 'acc_x', 'acc_y', 'acc_z']][
                       hangtime_si.players[novice]['features'][activity]['start_index']:
                       hangtime_si.players[novice]['features'][activity]['end_index']]
            # magnitude = hangtime_si.players[novice]['data'][['timestamp', 'magnitude']][
            #             hangtime_si.players[novice]['features'][activity]['start_index']:
            #             hangtime_si.players[novice]['features'][activity]['end_index']]
            # magnitude = magnitude.set_index('timestamp')

            fft = hangtime_si.players[novice]['features'][activity]['fft']
            snr = hangtime_si.players[novice]['features'][activity]['snr']
            if activity != 'complete_signal':
                filtered_magnitude = hangtime_si.players[novice]['features'][activity]['filtered_magnitude']
            if activity == 'dribbling':
                n_dribblings = hangtime_si.players[novice]['features'][activity]['dribblings_per_seconds']
                peaks = hangtime_si.players[novice]['features'][activity]['signal_peaks']
        else:
            raw_data = hangtime_bo.players[novice]['data'][['timestamp', 'acc_x', 'acc_y', 'acc_z']][
                       hangtime_bo.players[novice]['features'][activity]['start_index']:
                       hangtime_bo.players[novice]['features'][activity]['end_index']]
            # magnitude = hangtime_bo.players[novice]['data'][['timestamp', 'magnitude']][
            #             hangtime_bo.players[novice]['features'][activity]['start_index']:
            #             hangtime_bo.players[novice]['features'][activity]['end_index']]
            # magnitude = magnitude.set_index('timestamp')
            fft = hangtime_bo.players[novice]['features'][activity]['fft']
            snr = hangtime_bo.players[novice]['features'][activity]['snr']
            if activity != 'complete_signal':
                filtered_magnitude = hangtime_bo.players[novice]['features'][activity]['filtered_magnitude']
            if activity == 'dribbling':
                n_dribblings = hangtime_bo.players[novice]['features'][activity]['dribblings_per_seconds']
                peaks = hangtime_bo.players[novice]['features'][activity]['signal_peaks']


        plots_novices[novice + '_' + location]['raw'] = viz.vizualize_one_player(raw_data, False)
        plots_novices[novice + '_' + location]['std_mean'] = viz.plot_std_mean_box_plot(filtered_magnitude)
        plots_novices[novice + '_' + location]['fft'] = viz.plot_fft(fft, snr, novice)
        if activity == 'dribbling':
            plots_novices[novice + '_' + location]['peaks'] = viz.plot_peaks(filtered_magnitude, peaks, n_dribblings)

    return plots_novices, plots_experts

def gather_scatter_plots(hangtime_si, hangtime_bo, feature, activity):

    import viz

    plots_experts = {}
    plots_novices = {}
    mag_sums = []
    axis_sums = []
    pcas = []
    scatter_plots = {}
    for player in hangtime_si.players.keys():
        players_data = hangtime_si.players[player]
        mag_sum = players_data['features'][activity]['mag_sum']
        pca = players_data['features'][activity]['pca']
        x_sum = players_data['features'][activity]['x_sum']
        y_sum = players_data['features'][activity]['y_sum']
        z_sum = players_data['features'][activity]['z_sum']
        tmp = pd.concat([x_sum, y_sum], axis=1)
        if feature == 'pca':
            scatter_plots[player+"_eu"] = viz.create_scatter_plot(pca, feature='pca')
        if feature == 'axis_sums':
            scatter_plots[player+"_eu"] = viz.create_scatter_plot(tmp, feature='axis_sum')
        fill_value = np.full(fill_value=player + "_na", shape=(pca.shape[0], 1))
        tmp['subject'] = fill_value
        pca['subject'] = fill_value
        pcas.append(pca)
        axis_sums.append(tmp)

    for player in hangtime_bo.players.keys():
        players_data = hangtime_bo.players[player]
        mag_sum = players_data['features'][activity]['mag_sum']
        pca = players_data['features'][activity]['pca']
        x_sum = players_data['features'][activity]['x_sum']
        y_sum = players_data['features'][activity]['y_sum']
        z_sum = players_data['features'][activity]['z_sum']
        tmp = pd.concat([x_sum, y_sum], axis=1)
        if feature == 'pca':
            scatter_plots[player+"_na"] = viz.create_scatter_plot(pca, feature='pca')
        if feature == 'axis_sums':
            scatter_plots[player+"_na"] = viz.create_scatter_plot(tmp, feature='axis_sum')
        fill_value = np.full(fill_value=player + "_na", shape=(pca.shape[0], 1))
        tmp['subject'] = fill_value
        pca['subject'] = fill_value
        pcas.append(pca)
        axis_sums.append(tmp)
    pcas = pd.concat(pcas)
    axis_sums = pd.concat(axis_sums)
    if feature == 'pca':
        scatter_plots['main_plot'] =  viz.create_scatter_plot(pcas, mode="main_plot", feature='pca')
    if feature == 'axis_sums':
        scatter_plots['main_plot'] = viz.create_scatter_plot(axis_sums, mode="main_plot", feature='axis_sum')
    return scatter_plots