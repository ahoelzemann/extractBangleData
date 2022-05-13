import numpy as np
import pandas as pd
import datetime

labeled_participants = ['10f0']


def read_labels(participant: str):
    sensor_data = pd.read_csv('/Users/alexander/Documents/Resources/decompressed/' + participant + '.csv')
    sensor_data.rename(columns={sensor_data.columns[0]: "timestamp"}, inplace=True)
    sensor_data['coarse'] = np.full(fill_value='not_labeled', shape=sensor_data.shape[0])
    sensor_data['basketball'] = np.full(fill_value='not_labeled', shape=sensor_data.shape[0])
    sensor_data['locomotion'] = np.full(fill_value='not_labeled', shape=sensor_data.shape[0])
    sensor_data['off/def'] = np.full(fill_value='not_labeled', shape=sensor_data.shape[0])
    sensor_data['in/out'] = np.full(fill_value='not_labeled', shape=sensor_data.shape[0])
    timestamps = sensor_data['timestamp'].to_numpy(dtype=str)
    timestamps = list(map(np.datetime64, timestamps))
    timestamps = np.array(timestamps)
    labels = pd.read_csv('/Users/alexander/Documents/Resources/eaf_projects/' + participant + '.txt', delimiter='\t',
                         index_col=0, header=None, lineterminator='\n')
    labels = labels.reset_index()
    columns = ['layer', 'drop1', 'start', 'start[ms]', 'end', 'end[ms]', 'length', 'length[ms]', 'label']
    labels.columns = columns
    labels = labels.drop(['drop1'], axis=1)

    starting_point = timestamps[0]
    for row in labels.iterrows():
        start_time = row[1]['start']
        end_time = row[1]['end']
        dt_start = datetime.datetime.strptime(start_time, "%H:%M:%S.%f").time()
        timedelta_start = pd.Timedelta(hours=dt_start.hour, minutes=dt_start.minute, seconds=dt_start.second,
                                       microseconds=dt_start.microsecond).to_timedelta64()
        dt_end = datetime.datetime.strptime(end_time, "%H:%M:%S.%f").time()
        timedelta_end = pd.Timedelta(hours=dt_end.hour, minutes=dt_end.minute, seconds=dt_end.second,
                                     microseconds=dt_end.microsecond).to_timedelta64()

        start = starting_point + timedelta_start
        end = starting_point + timedelta_end
        start_index = np.abs(np.subtract(timestamps, start)).argmin()
        end_index = np.abs(np.subtract(timestamps, end)).argmin()
        label_length = end_index - start_index
        new_label = np.full(fill_value=row[1]['label'], shape=(label_length, 1))

        sensor_data.loc[start_index:end_index - 1, row[1]['layer']] = new_label

    sensor_data = sensor_data.set_index(sensor_data['timestamp'])
    del sensor_data['timestamp']
    sensor_data.insert(0, 'subject', np.full(fill_value=participant, shape=sensor_data.shape[0]))
    return sensor_data


for participant in labeled_participants:
    data = read_labels(participant=participant)
    data.to_csv('/Users/alexander/Documents/Resources/ready/' + participant + '.csv', sep=',')
