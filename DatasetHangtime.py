import pandas as pd


class HangtimeUtils:

    def __init__(self, place, rootpath):

        self.path = rootpath + "/" + place + "/"

    def load_activity_of_subject(self, subject, activity):

        path = self.path + subject + ".csv"
        df = pd.read_csv(path)
        df = df[df['basketball'].str.match(activity)]
        indices = df.index
        results = []

        starts = []
        ends = []
        starts.append(indices[0])

        for i in range(len(indices) - 1):
            current = indices[i]
            if indices[i + 1] - current > 1:
                starts.append(indices[i + 1])
                ends.append(indices[i])
        ends.append(indices[-1])

        for e in range(len(starts)):
            results.append(df.loc[starts[e]: ends[e]])
        return results

    def subject_analysis(self, subject, place, printing=False):

        path = self.path + subject + ".csv"
        # dribblings = pd.read_csv(path)

        dribbleCounter = _dribbleCounter(subject, self.load_activity_of_subject(subject, "dribbling"),place, printing)
        print(dribbleCounter)


def _dribbleCounter(subject, df, place, printing=False):
    from scipy.signal import find_peaks
    import viz
    import Util
    import numpy as np
    from scipy.signal import savgol_filter
    df = max(df, key=len).reset_index().loc[:, ['timestamp', 'acc_z']]
    filtered_signal = Util.mean_filter(df, 10)
    # filtered_signal = pd.Series(savgol_filter(df.loc[:, 'acc_z'], 100, 1))
    peaks, _ = find_peaks(filtered_signal, prominence=.8)

    y_val_peaks = np.full(fill_value=np.nan, shape=(filtered_signal.shape[0]))
    y_val_peaks[peaks] = filtered_signal.iloc[peaks]
    y_val_peaks = pd.Series(y_val_peaks, index=filtered_signal.index)

    n_dribblings = _calc_dribblings_per_second(filtered_signal, y_val_peaks)
    if printing:
        viz.plot_peaks(filtered_signal, y_val_peaks, n_dribblings, subject, place)
    return n_dribblings


def _calc_dribblings_per_second(filtered_signal, y_val_peaks):
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

place = 'Siegen'
hangtime_si = HangtimeUtils(place, '/Users/alexander/Documents/Resources/ready')
labeled_subjects = ["10f0", "2dd9", "4d70", "9bd4", "ce9d", "f2ad", "ac59", "0846", "a0da", "b512", "e90f", "4991", "05d8"]
# labeled_subjects = ['05d8']
for subject in labeled_subjects:
    hangtime_si.subject_analysis(subject, place, printing=True)
