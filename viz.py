import os

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def plot_imu_data(imu, title, time_as_indices=True):
    """
    Method written by kvl for reading out the files that is written by the imu itself
    :param imu:
    :param title:
    :param time_as_indices:
    """
    acc = imu[:, 1:4]
    timestamps = imu[:, 0]
    if time_as_indices:
        timestamps = timestamps.astype('datetime64[ms]')


    fig, axes = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(14, 8))
    if time_as_indices:
        pd.DataFrame(acc, index=timestamps).plot(ax=axes[0])
    else:
        axes[0].plot(acc, lw=2, ls='-')
        # sb.lineplot(y=acc.T, axes=axes[0])


    axes[0].set_prop_cycle('color', ['red', 'green', 'blue'])
    axes[0].set_ylabel('Acceleration (mg)', fontsize=16)
    axes[0].set_ylim((-10, 10))
    axes[0].legend(['x-axis', 'y-axis', 'z-axis'], loc='upper left')
    plt.xticks(rotation=45)
    plt.xlabel('Timestamp', fontsize=16)
    plt.suptitle(title, fontsize=19)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    #Todo add HR and Body-Temp to subplots

    plt.show()


all = False
data = []
subject = 'ce9d'
rootdir = '/Users/alexander/Documents/Resources/decompressed/Boulder/'
time_as_indices = True
if all:
    files = os.listdir(rootdir)
    files.remove(".DS_Store")
    for filenumber in range(0, len(files)):
        if files[filenumber] != ".DS_Store":
            file = rootdir + files[filenumber]
            # file2 = rootdir + files[filenumber+1]
            plot_imu_data(pd.read_csv(file).to_numpy(), file, time_as_indices=time_as_indices)
            data.append(pd.read_csv(file))
else:
    rootdir = rootdir + subject + ".csv"
    plot_imu_data(pd.read_csv(rootdir).to_numpy(), rootdir, time_as_indices=time_as_indices)