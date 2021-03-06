import numpy as np

import Util
import viz


def novice_vs_expert_dribbling(novices, experts):
    import viz
    import DatasetHangtime
    import Util
    starts = {
        '0846_si': '41:03',
        'ac59_si': '39:53',
        'f2ad_si': '40:30',
        '05d8_si': '40:10',
        '2dd9_bo': '24:35',
        '10f0_bo': '23:58'
    }
    ends = {
        '0846_si': '41:23',
        'ac59_si': '40:13',
        'f2ad_si': '40:50',
        '05d8_si': '40:30',
        '2dd9_bo': '24:55',
        '10f0_bo': '24:16'
    }
    for player_id in novices:
        novices[player_id] = Util.cut_df_at_timestamps(novices[player_id], starts[player_id], ends[player_id])
    for player_id in experts:
        experts[player_id] = Util.cut_df_at_timestamps(experts[player_id], starts[player_id], ends[player_id])

    nov_ffts, exp_ffts = show_periodicity_novices_experts(novices, experts)
    viz.plot_novice_vs_expert(novices=novices, experts=experts)
    viz.plot_novice_vs_expert_fft(nov_ffts, exp_ffts)


def show_periodicity_novices_experts(novices, experts):
    from scipy.fft import fft, ifft, fftfreq
    import numpy as np
    nov_ffts = []
    exp_ffts = []
    signal_to_noise_exp = []
    signal_to_noise_nov = []
    for player_id in novices:
        signal = novices[player_id].loc[:, ['acc_x', 'acc_y', 'acc_z']].to_numpy()
        mag = list(map(Util.calc_magnitude, signal))
        signalFFT = fft(mag)
        ## Get power spectral density
        signalPSD = np.abs(signalFFT)[1:101]
        dominant_freq_index = np.where(signalPSD == np.amax(signalPSD))[0]
        signal_to_noise_ratio = signalPSD[dominant_freq_index] / np.mean(np.delete(signalPSD, dominant_freq_index))
        signal_to_noise_nov.append(signal_to_noise_ratio[0])
        nov_ffts.append(signalPSD)
    for player_id in experts:
        signal = experts[player_id].loc[:, ['acc_x', 'acc_y', 'acc_z']].to_numpy()
        mag = list(map(Util.calc_magnitude, signal))
        signalFFT = fft(mag)
        ## Get power spectral density
        signalPSD = np.abs(signalFFT)[1:101]
        dominant_freq_index = np.where(signalPSD == np.amax(signalPSD))[0]
        signal_to_noise_ratio = signalPSD[dominant_freq_index] / np.mean(np.delete(signalPSD, dominant_freq_index))
        signal_to_noise_exp.append(signal_to_noise_ratio[0])
        exp_ffts.append(signalPSD)

    return nov_ffts, exp_ffts, signal_to_noise_nov, signal_to_noise_exp


def feature_analysis(dataframe, activity='complete_signal', start=0, end=-1, mode='not_specified'):
    import Util
    # calc magnitude

    for subject in dataframe.players:
        fft, signal_to_noise_ratio = Util.calc_fft(dataframe.players[subject]['data']['magnitude'][start:end])

        dataframe.players[subject]['features'][activity] = {
            'stds': dataframe.players[subject]['data'].loc[:, ['acc_x', 'acc_y', 'acc_z', 'magnitude']][start:end].std(),
            'means': dataframe.players[subject]['data'].loc[:, ['acc_x', 'acc_y', 'acc_z', 'magnitude']][
                    start:end].mean(),
            'fft': fft,
            'signal_to_noise_ratio': signal_to_noise_ratio,
            'start_index': start,
            'end_index': end,
            'mode' : mode
        }
    return dataframe


def feature_analysis_activity(dataframe, activity, start=0, end=-1, mode='longest_intervall'):

    for subject in dataframe.players:
        if mode == 'longest_intervall':
            end = dataframe.players[subject]['activity_indices'][activity]['ends'][
                dataframe.players[subject]['activity_indices'][activity]['longest_intervall']]
            start = dataframe.players[subject]['activity_indices'][activity]['starts'][
                dataframe.players[subject]['activity_indices'][activity]['longest_intervall']]
            # signal = dataframe.data[subject]['magnitude']
        feature_analysis(dataframe, activity, start, end, mode)
    return dataframe
