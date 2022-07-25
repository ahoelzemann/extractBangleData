import pandas as pd
import experiments
import viz


class HangtimeUtils:

    def __init__(self, place, rootpath):

        # self.path = rootpath + "/" + place + "/"
        self.path = rootpath + "/"
        self.data = {}
        self.skills = {}
        self.activity_indices = {}

    def load_data_of_subject(self, subject, skill):
        path = self.path + subject + ".csv"
        df = pd.read_csv(path)
        self.data[subject] = df
        self.skills[subject] = skill

    def load_activity_of_subjects(self, activity):

        # path = self.path + ".csv"
        # df = pd.read_csv(path)
        if activity == 'dribbling' or activity == 'shot' or activity == 'layup' or activity == 'rebound' or activity == 'pass':
            layer = 'basketball'
        else:
            layer = 'locomotion'
        import numpy as np
        self.activity_indices[activity] = {}
        for subject in self.data:
            starts = []
            ends = []
            activity_indices = np.where(self.data[subject][layer].str.match(activity) == True)
            starts.append(activity_indices[0][0])
            for index in range(1, activity_indices[0].shape[0]):
                current = activity_indices[0][index]
                last = activity_indices[0][index-1]
                if current - last > 1:
                    starts.append(current)
                    ends.append(last)
            ends.append(activity_indices[0][-1])

            self.activity_indices[activity][subject] = {
                'starts': starts,
                'ends': ends
            }


    def subject_analysis(self, subject, place, printing=False):

        dribbleCounter = _dribbleCounter(subject, self.load_activity_of_subject(subject, "dribbling"), place, printing)
        print(dribbleCounter)


def _dribbleCounter(subject, df, place, printing=False):
    from scipy.signal import find_peaks
    import viz
    import Util
    import numpy as np
    from scipy.signal import savgol_filter
    # df = max(df, key=len).reset_index().loc[:, ['timestamp', 'acc_z']]
    df = max(df, key=len).reset_index()
    filtered_signal = Util.mean_filter(df, 10)
    # filtered_signal = pd.Series(savgol_filter(df.loc[:, 'acc_z'], 100, 1))
    peaks, _ = find_peaks(filtered_signal, prominence=.6)

    y_val_peaks = np.full(fill_value=np.nan, shape=(filtered_signal.shape[0]))
    y_val_peaks[peaks] = filtered_signal.iloc[peaks]
    y_val_peaks = pd.Series(y_val_peaks, index=filtered_signal.index)

    n_dribblings = _calc_dribblings_per_second(filtered_signal, y_val_peaks)
    if printing:
        viz.plot_peaks(filtered_signal, y_val_peaks, n_dribblings, subject, place)
    return n_dribblings


def _calc_dribblings_per_second(y_val_peaks):
    import numpy as np
    from datetime import datetime
    y_val_peaks = y_val_peaks.dropna()
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


def sort_experts_and_novices(hangtime_si, hangtime_bo):
    experts = {}
    novices = {}
    for player_id in hangtime_si.data:
        if hangtime_si.skills[player_id] == 'novice':
            novices[player_id+'_si'] = hangtime_si.data[player_id]
        else:
            experts[player_id + '_si'] = hangtime_si.data[player_id]
    for player_id in hangtime_bo.data:
        if hangtime_bo.skills[player_id] == 'novice':
            novices[player_id + '_bo'] = hangtime_bo.data[player_id]
        else:
            experts[player_id + '_bo'] = hangtime_bo.data[player_id]

    return novices, experts

# place = 'Siegen'
# hangtime_si = HangtimeUtils('Siegen', '/Users/alexander/Documents/Resources/ready')
# place = 'Boulder'
hangtime_si = HangtimeUtils('Siegen', 'G:/Sciebo/Basketball_Activity_Data/Siegen_DE/labeled')
hangtime_bo = HangtimeUtils('Boulder', 'G:/Sciebo/Basketball_Activity_Data/Boulder_US/labeled')
hangtime_si.load_data_of_subject('0846', 'novice')
hangtime_si.load_data_of_subject('f2ad', 'expert')
hangtime_si.load_data_of_subject('ac59', 'expert')
hangtime_si.load_data_of_subject('b512', 'expert')
hangtime_si.load_data_of_subject('05d8', 'expert')
hangtime_bo.load_data_of_subject('2dd9', 'novice')
hangtime_bo.load_data_of_subject('10f0', 'novice')


# novices, experts = sort_experts_and_novices(hangtime_si, hangtime_bo)
# novices_and_experts = novices.update(experts)
# experiments.show_periodicity_novices_experts(novices, experts)
# experiments.novice_vs_expert_dribbling(novices, experts)

hangtime_si = experiments.feature_analysis(hangtime_si)
hangtime_bo = experiments.feature_analysis(hangtime_bo)
hangtime_si.load_activity_of_subjects('dribbling')
hangtime_si.load_activity_of_subjects('shot')
hangtime_si.load_activity_of_subjects('layup')
hangtime_si.load_activity_of_subjects('running')
# viz.vizualize_one_player(hangtime_si.data['b512'], '')

# viz.plot_novice_vs_expert(novices=novices, experts=experts)
labeled_subjects = ["10f0", "2dd9", "4d70", "9bd4", "ce9d", "f2ad", "ac59", "0846", "a0da", "b512", "e90f", "4991",
                    "05d8"]
# # labeled_subjects = ['05d8']
# for subject in labeled_subjects:
#     hangtime_si.subject_analysis(subject, place, printing=True)
