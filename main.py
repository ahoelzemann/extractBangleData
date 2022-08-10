import math
from glob import glob
import os

import numpy

import decompressor
import numpy as np
import pandas as pd
import wave


# import viz


def saveAsWave(imu_data, deviceId, place, true_freq):
    nchannels = 1
    sample_width = 2
    # true_freq = 50
    audio_x = (imu_data[:, 1] * (2 ** 15 - 1)).astype("<h")
    audio_y = (imu_data[:, 2] * (2 ** 15 - 1)).astype("<h")
    audio_z = (imu_data[:, 3] * (2 ** 15 - 1)).astype("<h")
    with wave.open("/Users/alexander/Documents/Resources/wave/" + place + "/" + deviceId + "_x.wav", "w") as f:
        f.setnchannels(nchannels)
        f.setsampwidth(sample_width)
        f.setframerate(true_freq)
        f.writeframes(audio_x.tobytes())
    with wave.open("/Users/alexander/Documents/Resources/wave/" + place + "/" + deviceId + "_y.wav", "w") as f:
        f.setnchannels(nchannels)
        f.setsampwidth(sample_width)
        f.setframerate(true_freq)
        f.writeframes(audio_y.tobytes())
    with wave.open("/Users/alexander/Documents/Resources/wave/" + place + "/" + deviceId + "_z.wav", "w") as f:
        f.setnchannels(nchannels)
        f.setsampwidth(sample_width)
        f.setframerate(true_freq)
        f.writeframes(audio_z.tobytes())


def checkfordrift(df):
    timestamps = df.index
    counter = 1
    true_freqs = []
    last = None
    for entry in timestamps:
        second = entry.second
        if last is not None:
            if second != last:
                true_freqs.append(counter)
                # if counter < 49:
                #     # print("Drift discovered at: " + str(entry) + " - " + str(counter))
                #     idx = idx + 1
                counter = 1
            else:
                counter = counter + 1
        last = second

    return true_freqs


def getLastNonNaN(series, index, missingvalues=1):
    if not pd.isna(series[index - 1]):
        return series[index - 1], missingvalues
    else:
        return getLastNonNaN(series, index - 1)


def getNextNonNaN(series, index, missingvalues=1):
    if index + 1 == series.shape[0]:
        return series[index - missingvalues], missingvalues
    try:
        if not pd.isna(series[index + 1]):
            return series[index + 1], missingvalues
        else:
            return getNextNonNaN(series, index + 1, missingvalues=missingvalues + 1)
    except:
        print('')


def replaceNaNValues(series, output_dtype='float'):
    if output_dtype == 'float':
        if series is not np.array:
            series = np.array(series)
        if pd.isna(series[0]):
            series[0] = series[1]
        if pd.isna(series[series.shape[0] - 1]):
            lastNonNan, numberOfMissingValues = getLastNonNaN(series, series.shape[0] - 1)
            if numberOfMissingValues != 1:
                for k in range(1, numberOfMissingValues):
                    series[series.shape[0] - 1 - k] = lastNonNan
            series[series.shape[0] - 1] = series[series.shape[0] - 2]
        for x in range(0, series.shape[0]):
            if pd.isna(series[x]):
                lastNonNan, _ = getLastNonNaN(series, x)
                nextNonNan, _ = getNextNonNaN(series, x)
                missingValue = np.average([lastNonNan, nextNonNan])
                series[x] = missingValue

    elif output_dtype == 'int' or output_dtype == 'string':
        if series is not np.array:
            series = np.array(series)
        if pd.isna(series[0]):
            series[0] = series[1]
        if pd.isna(series[series.shape[0] - 1]):
            lastNonNan, numberOfMissingValues = getLastNonNaN(series, series.shape[0] - 1)
            if numberOfMissingValues != 1:
                for k in range(1, numberOfMissingValues):
                    series[series.shape[0] - 1 - k] = lastNonNan
            series[series.shape[0] - 1] = series[series.shape[0] - 2]
        for x in range(0, series.shape[0]):
            if pd.isna(series[x]):
                lastNonNan, missingValuesLast = getLastNonNaN(series, x)
                nextNonNan, missingValuesNext = getNextNonNaN(series, x)
                if missingValuesLast < missingValuesNext:
                    series[x] = lastNonNan
                else:
                    series[x] = nextNonNan
    else:
        print("Please choose a valid output dtype. You can choose between float, int and string.")
        exit(0)

    return series.T


def getNaNsAndInterpolationLength(data):
    nanValuesStart = []
    nanValuesEnd = []
    activityDataStarts = [0]
    missingValues = np.where(data.isnull())[0]
    nanValuesStart.append(missingValues[0])
    for index in range(len(missingValues)):
        try:
            if (missingValues[index + 1] - missingValues[index]) > 1:
                nanValuesStart.append(missingValues[index + 1])
                nanValuesEnd.append(missingValues[index])
        except:
            pass
            # if (missingValues[index + 1] - missingValues[index]) > 1:
            #     nanValuesStart.append(missingValues[index + 1])
            #     nanValuesEnd.append(missingValues[index])

    nanValuesEnd.append(missingValues[-1])
    interpolationLengths = np.subtract(nanValuesEnd, nanValuesStart)

    for n in range(len(interpolationLengths)):
        activityDataStarts.append(nanValuesStart[n] - 1)
        # activityDataEnds.append(nanValuesEnd[n] + 1)

    activityDataStarts.pop()
    return nanValuesStart, nanValuesEnd, interpolationLengths, activityDataStarts


def find_nearest_ts(ts, list_timestamps):
    from bisect import bisect_left
    # Given a presorted list of timestamps:  s = sorted(index)
    k = bisect_left(list_timestamps, ts)
    result = list_timestamps[max(0, k - 1): k + 2], key = lambda t: abs(ts - t)
    return min(result)


def copy_nans_to_loc(data):
    nanValuesStart, nanValuesEnd, interpolationLengths, activityDataChunks = getNaNsAndInterpolationLength(
        data)
    complete_file = []
    for c in range(len(activityDataChunks)):
        tmp_result = []
        if c == 0:
            asdf = 0
        else:
            asdf = nanValuesEnd[c - 1] + 1
        chunk = data[asdf:nanValuesEnd[c] + 1]
        nan_start = chunk.shape[0] - (nanValuesEnd[c] - nanValuesStart[c]) - 1

        chunk = chunk.reset_index().to_numpy()
        n_nans = chunk[nan_start:].shape[0]
        indices = chunk[:, 0]
        chunk = np.delete(chunk, 0, axis=1)
        while n_nans != 0:
            chunk, n_nans = add_remaining_nans(chunk, n_nans, indices.shape[0])
        complete_file.append(chunk)
    complete_file.append(data[nanValuesEnd[-1] + 1:])
    complete_file = np.concatenate(complete_file, axis=0)
    return complete_file


def add_remaining_nans(data, remaining_nans, original_length):
    from decimal import Decimal, ROUND_UP

    every_nth_dec = original_length / remaining_nans
    every_nth = math.ceil(every_nth_dec)
    every_nth_dec = every_nth_dec - int(every_nth_dec)
    every_nth_dec = float(Decimal(every_nth_dec).quantize(Decimal(".1"), rounding=ROUND_UP))
    counter = 0
    idx = round(original_length / remaining_nans)
    nans = np.array([np.NaN, np.NaN, np.NaN])
    decimal_counter = 0.0
    while remaining_nans != 0:
        try:
            data = np.insert(data, obj=idx, values=nans, axis=0)
            # data = data[:-1]
            decimal_counter = decimal_counter + every_nth_dec
            if decimal_counter > 1:
                idx = idx + every_nth - 1
                decimal_counter = decimal_counter - 1.0
            else:
                idx = idx + every_nth

            # idx = idx + round((original_length - idx) / remaining_nans)
            remaining_nans = remaining_nans - 1
            counter = counter + 1
        except Exception as e:
            break

    return data, remaining_nans


def add_remaining_nans_copy(data, remaining_nans, original_length):
    every_nth = math.floor(original_length / remaining_nans)
    start = 0
    end = every_nth - 1
    nans = np.array([np.NaN, np.NaN, np.NaN])
    result = []
    while remaining_nans != -1:
        if remaining_nans != 0 and end < data.shape[0]:
            tmp = list(data[start:end])
            tmp.append(nans)
            start = end
            end = end + every_nth - 1
        else:
            tmp = list(data[start:])

        result.append(np.array(tmp))
        remaining_nans = remaining_nans - 1

    result = np.concatenate(result, axis=0)
    rest = original_length - result.shape[0]

    return result, rest


def interpolate(data1: np.array, timestamps):
    data1 = data1[:, 1:].astype('float64')
    la = len(data1[:, 1])
    result_tmp = []
    # result.append(timestamps)
    for column in range(data1.shape[1]):
        result_tmp.append(np.interp(np.linspace(0, la - 1, num=len(timestamps)), np.arange(la), data1[:, column]))
    result_tmp = pd.DataFrame(np.array(result_tmp).T, index=timestamps, columns=["acc_x", "acc_y", "acc_z"])
    return result_tmp


def save_decompressed_files(dataframes: list, device_id: str, place: str, true_freq: list):
    from bisect import bisect_left
    indices = []
    data = []
    global_starting_time = pd.Timestamp('2022-02-26 14:13:21.000')
    if place == "Siegen":
        options = {"10f0": 16906,  # sync, RAML, labeled 16855
                   "2dd9": 16920,  # sync,       labeled 16868 16931
                   "4d70": 16949,  # sync, RAML, labeled
                   "9bd4": 17018,  # sync,       labeled 16924
                   "ce9d": 16950,  # sync, RAML, labeled 16761 16719
                   "f2ad": 16992,
                   "ac59": 16987,  # sync, RAML 16751
                   "0846": 16949,  # sync, RAML, labeled  16882,
                   "a0da": 17006,  # sync, RAML, labeled 17006
                   "b512": 16917,  # sync,16893 RAML, labeled  16917
                   "e90f": 17111,  # sync, RAML, labeled 16982 16960
                   "4991": 17001,  # sync, RAML, labeled 16802
                   "05d8": 16950,  # sync, RAML, labeled 16975
                   "c6f3": 16208  # Gianni
                   }
    else:
        options = {"10f0": 29600, # 29575
                   "2dd9": 3805,  #
                   "4d70": 32710, #  32079
                   "9bd4": 33728,
                   "ce9d": 4945,  #
                   "f2ad": 7563,
                   "ac59": 29579,
                   "0846": 7235,
                   "a0da": 37293,
                   "b512": 35593,
                   "c6f3": 35877  # 36519
                   }
    start = options[device_id]
    for dataframe in dataframes:
        indices = indices + dataframe.index.tolist()
        data = data + dataframe.values.tolist()
    df = pd.DataFrame(data, index=indices, columns=["acc_x", "acc_y", "acc_z"])
    df = df.resample("20ms").mean()
    df = df.interpolate()
    starting_index = bisect_left(indices, global_starting_time) - 1
    # test = df.index.get_loc(starting_timestamp)
    df = df[start:]
    # df = df[starting_index:]
    print("starting_index: " + str(starting_index))
    print("Device_ID: " + device_id + " Has null " + str(df.isnull().values.any()))
    df.to_csv("/Users/alexander/Documents/Resources/decompressed/" + place + "/" + device_id + ".csv")
    # checkfordrift(df)
    print("DeviceID: " + device_id + " saved.")
    return df

def make_equidistant(subject_files):
    import datetime
    true_freqs = []
    for fc in range(len(subject_files)):
        current_df = subject_files[fc]
        # check if file the is last file
        if fc != len(subject_files) - 1:
            next_df = subject_files[fc + 1]

            current_df = current_df.append(next_df.iloc[0], ignore_index=False)
            # subject_files[fc + 1] = next_df.iloc[1:, :]
            t = pd.date_range(start=current_df.index[0],
                              end=current_df.index[-1],
                              periods=current_df.shape[0])
            current_df = current_df.set_index(t)[:-1]

        else:
            true_freq = round(np.mean(true_freqs))
            freq_new_ms = int(((1 / true_freq) * 1000))
            new_range = pd.date_range(current_df.index[0], current_df.index[-1], freq=str(round(freq_new_ms)) + "ms")
            n_nans = new_range.shape[0] - current_df.shape[0]
            if n_nans > 0:
                while n_nans != 0:
                    current_df, n_nans = add_remaining_nans(current_df, n_nans, new_range.shape[0])
            elif n_nans < 0:
                new_range = []
                new_range.append(current_df.index[0])
                for idx in range(1, current_df.index.shape[0]):
                    new_range.append(current_df.index[idx-1]+datetime.timedelta(milliseconds=freq_new_ms))
                current_df = current_df.set_index(pd.DatetimeIndex(new_range))
            else:
                current_df = current_df.set_index(pd.DatetimeIndex(new_range))
        subject_files[fc] = current_df
        true_freqs.append(round(np.mean(checkfordrift(current_df))))

    return subject_files, true_freqs


def readBinFile(path):
    bufferedReader = open(path, "rb")
    return bufferedReader.read()


selected_subject = 'ac59'
all_ = True
place = "Boulder"
dataset_folder = ""
if place == "Boulder":
    dataset_folder = "/Users/alexander/Downloads/Boulder Study/smartwatch_data/"
else:
    dataset_folder = "/Users/alexander/Documents/Resources/IMU_BBSI/"
activityFilesOrdered = []
stepsAndMinutesOrdered = []

if all_:
    folders = [x[1] for x in os.walk(dataset_folder)][0]
    for folder in folders:
        activityFiles = glob(dataset_folder + folder + "/" + "*.bin")
        try:
            activityFiles.remove(dataset_folder + folder + "/" + "d20statusmsgs.bin")
        except:
            pass
        activityFiles = sorted(activityFiles)
        activityFilesOrdered.append(activityFiles)
        day = ""
        filesOfOneDay = []
        for files in activityFilesOrdered:
            result = list(map(readBinFile, files))
            i = 0
            subjectData = []
            for file in files:
                bangleID, filename = file.split("/")[-2], file.split("/")[-1]
                if len(result[i]) > 0:
                    try:
                        subjectData.append(decompressor.decompress(result[i]))
                    except:
                        try:
                            dataframe = decompressor.decompress(result[i][17:])
                        except:
                            pass
                i += 1

        subjectData, true_freqs =  make_equidistant(subjectData)
        subjectData = save_decompressed_files(subjectData, folder, place, true_freqs)
        if place == "Siegen":
            decompressed_folder = '/Users/alexander/Documents/Resources/decompressed/Siegen/'
        else:
            decompressed_folder = '/Users/alexander/Documents/Resources/decompressed/Boulder/'
        path = decompressed_folder + folder + '.csv'
        print("saving as wave")
        true_freqs = checkfordrift(subjectData)
        saveAsWave(pd.read_csv(path).to_numpy(), folder, place, np.mean(true_freqs))
        print('done')


else:
    activityFiles = glob(dataset_folder + selected_subject + "/" + "*.bin")
    try:
        activityFiles.remove(dataset_folder + selected_subject + "/" + "d20statusmsgs.bin")
    except:
        pass
    activityFiles = sorted(activityFiles)
    activityFilesOrdered.append(activityFiles)
    day = ""
    filesOfOneDay = []
    for files in activityFilesOrdered:
        result = list(map(readBinFile, files))
        i = 0
        subjectData = []
        for file in files:
            bangleID, filename = file.split("/")[-2], file.split("/")[-1]
            if len(result[i]) > 0:
                try:
                    subjectData.append(decompressor.decompress(result[i]))
                except:
                    try:
                        dataframe = decompressor.decompress(result[i][17:])
                    except:
                        pass
            i += 1

    subjectData, true_freqs = make_equidistant(subjectData)
    saved_df = save_decompressed_files(subjectData, selected_subject, place, true_freqs)
    if place == "Siegen":
        decompressed_folder = '/Users/alexander/Documents/Resources/decompressed/Siegen/'
    else:
        decompressed_folder = '/Users/alexander/Documents/Resources/decompressed/Boulder/'
    # files = os.listdir(decompressed_folder)
    # for file in files:
    # if file != ".DS_Store" or file != "vids":
    path = decompressed_folder + selected_subject + '.csv'
    print("saving as wave")
    true_freqs = checkfordrift(saved_df)
    saveAsWave(pd.read_csv(path).to_numpy(), selected_subject, place, np.mean(true_freqs))
    print('done')