import numpy as np
import pandas as pd
import datetime

# labeled_participants = [
#                         "2dd9"]
labeled_participants = ["10f0", "2dd9", "4d70", "9bd4", "ce9d", "f2ad", "ac59", "0846", "a0da", "b512", "e90f", "4991",
                        "05d8"]
place = 'Siegen'
# labeled_participants = ["2dd9", "10f0", "ce9d", "f2ad"]
# place = 'Boulder'


# labeled_participants = ["ce9d"]

def read_labels(participant: str, place: str):
    sensor_data = pd.read_csv('/Users/alexander/Documents/Resources/decompressed/' + place + "/" + participant + '.csv')
    sensor_data.rename(columns={sensor_data.columns[0]: "timestamp"}, inplace=True)
    sensor_data['coarse'] = np.full(fill_value='not_labeled', shape=sensor_data.shape[0])
    sensor_data['basketball'] = np.full(fill_value='not_labeled', shape=sensor_data.shape[0])
    sensor_data['locomotion'] = np.full(fill_value='not_labeled', shape=sensor_data.shape[0])
    sensor_data['off/def'] = np.full(fill_value='not_labeled', shape=sensor_data.shape[0])
    sensor_data['in/out'] = np.full(fill_value='not_labeled', shape=sensor_data.shape[0])
    timestamps = sensor_data['timestamp'].to_numpy(dtype=str)
    timestamps = list(map(np.datetime64, timestamps))
    timestamps = np.array(timestamps)
    labels = pd.read_csv('/Users/alexander/Documents/Resources/annotations/' + place + "/" + participant + '.txt',
                         delimiter='\t',
                         index_col=0, header=None, lineterminator='\n')

    labels = labels.reset_index()
    columns = ['layer', 'drop1', 'start', 'start[ms]', 'end', 'end[ms]', 'length', 'length[ms]', 'label']
    labels.columns = columns
    labels = labels.drop(['drop1'], axis=1)

    starting_point = timestamps[0]
    for row in labels.iterrows():
        # if row[-1][-1] != 'rebound':
        start_time = row[1]['start']
        end_time = row[1]['end']
        dt_start = datetime.datetime.strptime(start_time, "%H:%M:%S.%f")
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
    sensor_data['coarse'] = sensor_data['coarse'].replace('\s+', '', regex=True)
    sensor_data['basketball'] = sensor_data['basketball'].replace('\s+', '', regex=True)
    sensor_data['locomotion'] = sensor_data['locomotion'].replace('\s+', '', regex=True)
    # sensor_data['off/def'] = sensor_data['off/def'].replace('\s+', '', regex=True)
    sensor_data['in/out'] = sensor_data['in/out'].replace('\s+', '', regex=True)

    return sensor_data


for participant in labeled_participants:
    data = read_labels(participant=participant, place=place)
    print("Device_ID: " + participant + " Has null " + str(data.isnull().values.any()))

    data.to_csv('/Users/alexander/Documents/Resources/ready/' + place + '/' + participant + '.csv', sep=',')
