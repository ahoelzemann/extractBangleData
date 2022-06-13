import math
from glob import glob
import os
import decompressor
import numpy as np
import pandas as pd
import wave


def saveAsWave(imu_data, deviceId):
    nchannels = 1
    sample_with = 2
    framerate = 50
    audio_x = (imu_data[:, 1] * (2 ** 15 - 1)).astype("<h")
    audio_y = (imu_data[:, 2] * (2 ** 15 - 1)).astype("<h")
    audio_z = (imu_data[:, 3] * (2 ** 15 - 1)).astype("<h")
    with wave.open("/Users/alexander/Documents/Resources/wave/" + deviceId + "_x.wav", "w") as f:
        f.setnchannels(nchannels)
        f.setsampwidth(sample_with)
        f.setframerate(framerate)
        f.writeframes(audio_x.tobytes())
    with wave.open("/Users/alexander/Documents/Resources/wave/" + deviceId + "_y.wav", "w") as f:
        f.setnchannels(nchannels)
        f.setsampwidth(sample_with)
        f.setframerate(framerate)
        f.writeframes(audio_y.tobytes())
    with wave.open("/Users/alexander/Documents/Resources/wave/" + deviceId + "_z.wav", "w") as f:
        f.setnchannels(nchannels)
        f.setsampwidth(sample_with)
        f.setframerate(framerate)
        f.writeframes(audio_z.tobytes())


def checkfordrift(df):
    timestamps = df.index
    counter = 0
    last = None
    for entry in timestamps:
        second = entry.second
        if last is not None:
            if second != last:
                # if counter != 50:
                #     print(str(entry) + ": " + str(counter)+"Hz")
                # if counter == 50:
                #     print("50Hz")
                # if counter > 50:
                #     print("Drift discovered at: " + str(entry))
                counter = 0
            else:
                counter = counter + 1
        last = second


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

    return series.round(5).T


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


def make_equidistant1(data: pd.DataFrame, new_freq):
    freq_new_ms = round(((1 / new_freq) * 1000))
    # data = data.set_index("timestamp")
    columns = data.columns
    result = []
    # activityIDs = data['activityID']
    # data = data.drop(['activityID'], axis=1)
    # activityIDs = activityIDs.resample(str(freq_new_ms) + "ms", axis=0).nearest()
    # new_indices = activityIDs.index
    # result.append(activityIDs.to_numpy())

    for column in data:
        c = data[column]
        resampledData = c.resample(str(freq_new_ms) + "ms", axis=0).mean()
        if resampledData.isnull().values.any():
            resampledData = replaceNaNValues(resampledData)
        result.append(resampledData)
    result = np.array(result).T
    new_timestamps = pd.DatetimeIndex(np.linspace(c.index[0].value, c.index[-1].value, len(result), dtype=np.int64))
    result = pd.DataFrame(result, columns=columns, index=new_timestamps, dtype=np.float64)

    return result


def make_equidistant(data: pd.DataFrame, new_freq):
    freq_new_ms = round(((1 / new_freq) * 1000))
    # data = data.set_index("timestamp")
    columns = data.columns
    result = []

    c = data.copy()
    resampledData = c.resample(str(freq_new_ms) + "ms", axis=0).mean()
    new_indices = resampledData.index
    if resampledData.isnull().values.any():
        nans = [np.NaN, np.NaN, np.NaN, np.NaN]
        nanValuesStart, nanValuesEnd, interpolationLengths, activityDataChunks = getNaNsAndInterpolationLength(
            resampledData)

        for c in range(len(activityDataChunks)):
            chunk = data[activityDataChunks[c]:nanValuesStart[c] + 1]
            chunk = chunk.reset_index().to_numpy()
            final_timestamps = pd.date_range(start=chunk[0, 0], end=chunk[-1, 0],
                                             periods=resampledData[:len(chunk) + interpolationLengths[c]].shape[
                                                 0]).tolist()
            chunk = interpolate(chunk, final_timestamps)
            result.append(chunk)
    result = pd.concat(result)
    result = make_equidistant1(result, 50)
    return result


def find_nearest_ts(ts, list_timestamps):
    from bisect import bisect_left
    # Given a presorted list of timestamps:  s = sorted(index)
    k = bisect_left(list_timestamps, ts)
    result = list_timestamps[max(0, k - 1): k + 2], key = lambda t: abs(ts - t)
    return min(result)


def interpolate(data1: np.array, timestamps):
    data1 = data1[:, 1:].astype('float64')
    la = len(data1[:, 1])
    result_tmp = []
    # result.append(timestamps)
    for column in range(data1.shape[1]):
        result_tmp.append(np.interp(np.linspace(0, la - 1, num=len(timestamps)), np.arange(la), data1[:, column]))
    result_tmp = pd.DataFrame(np.array(result_tmp).T, index=timestamps, columns=["acc_x", "axx_y", "acc_z"])
    return result_tmp


def save_decompressed_files(dataframes: list, device_id: str, place:str):
    from bisect import bisect_left
    indices = []
    data = []
    global_starting_time = pd.Timestamp('2022-02-26 14:13:18.868502784')
    if place == "Siegen":
        options = {"10f0": 16733,  # sync, RAML, labeled
               "2dd9": 16749,  # sync labeled
               "4d70": 16764,  # sync, RAML
               "9bd4": 16979,  # sync, labeled
               "ce9d": 16761,  # sync, RAML
               "f2ad": 16100,
               "ac59": 16783,  # sync, RAML
               "0846": 16752,  # sync, RAML
               "a0da": 16825,  # sync, RAML
               "b512": 16798,  # sync, RAML
               "e90f": 16960,  # sync, RAML, labeled
               "4991": 16863,  # sync, RAML
               "05d8": 16790,  # sync, RAML, labeled
               "c6f3": 16208   # Gianni
               }
    else:
        options = {"10f0": 1,
                   "2dd9": 1,
                   "4d70": 1,
                   "9bd4": 1,
                   "ce9d": 1,
                   "f2ad": 1,
                   "ac59": 1,
                   "0846": 1,
                   "a0da": 1,
                   "b512": 1,
                   "e90f": 1,
                   "4991": 1,
                   "05d8": 1,
                   "c6f3": 1
                   }
    start = options[device_id]
    for dataframe in dataframes:
        indices = indices + dataframe.index.tolist()
        data = data + dataframe.values.tolist()
    df = pd.DataFrame(data, index=indices)
    df = make_equidistant(df, 50)
    # if df.shape[0] > 326800:
    #     start = df.shape[0] - 326800
    # df_copy = df.copy().reset_index()['index']
    # df_copy = df_copy.reset_index().loc[df_copy.Date == '2020-12-05', 'index']

    # hours = np.where(df.index.hour.to_numpy() == 14)
    # # hours =
    # minutes = df.index.minute.to_numpy()[:hours[0][-1]]
    # minutes = np.where(minutes == 10)
    # microseconds = df.index.microsecond.to_numpy()[minutes]
    # starting_index = np.absolute(np.subtract(microseconds, starting_timestamp_millisecond)).argmin()
    starting_index = bisect_left(indices, global_starting_time) - 1
    # test = df.index.get_loc(starting_timestamp)
    df = df[start:]
    print("starting_index: " + str(starting_index))
    df.to_csv("/Users/alexander/Documents/Resources/decompressed/" + place + "/" + device_id + ".csv")
    checkfordrift(df)
    print("DeviceID: " + device_id + " saved.")


def readBinFile(path):
    bufferedReader = open(path, "rb")
    return bufferedReader.read()


selected_subject = '2dd9'
all_ = False
# dataset_folder = "/Users/alexander/Downloads/Boulder Study/smartwatch_data/"
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

        save_decompressed_files(subjectData, folder, "Siegen")

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

    save_decompressed_files(subjectData, selected_subject, "Siegen")

data = []
decompressed_folder = '/Users/alexander/Documents/Resources/decompressed/Siegen/'
files = os.listdir(decompressed_folder)
for file in files:
    if file != ".DS_Store":
        path = decompressed_folder + file
        print("saving as wave")
        saveAsWave(pd.read_csv(path).to_numpy(), file.split(".")[0])
