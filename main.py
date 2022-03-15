from glob import glob
import os
import decompressor


def save_decompressed_files(dataframes: list, device_id: str):
    import pandas as pd
    df = pd.concat(dataframes)[16100:]
    df.to_csv("/Users/alexander/Documents/Ressources/decompressed/" + device_id + ".csv")
    print("DeviceID: " + device_id + " saved.")


def readBinFile(path):
    bufferedReader = open(path, "rb")
    return bufferedReader.read()


dataset_folder = "/Users/alexander/Documents/Ressources/IMU_BBSI/"
folders = [x[1] for x in os.walk(dataset_folder)][0]

activityFilesOrdered = []
stepsAndMinutesOrdered = []
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

    save_decompressed_files(subjectData, folder)
