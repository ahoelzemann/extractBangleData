def int64_to_str(a, signed):
    import math

    negative = signed and a[7] >= 128
    H = 0x100000000
    D = 1000000000
    h = a[4] + a[5] * 0x100 + a[6] * 0x10000 + a[7] * 0x1000000
    l = a[0] + a[1] * 0x100 + a[2] * 0x10000 + a[3] * 0x1000000
    if negative:
        h = H - 1 - h
        l = H - l

    hd = math.floor(h * H / D + l / D)
    ld = (((h % D) * (H % D)) % D + l) % D
    ldStr = str(ld)
    ldLength = len(ldStr)
    sign = ''
    if negative:
        sign = '-'
    if hd != 0:
        result = sign + str(hd) + ('0' * (9 - ldLength))
    else:
        result = sign + ldStr

    return result


def decompress(value, csvStr=''):
    from datetime import datetime
    import numpy as np
    import pandas as pd

    # value = np.uint8(data)
    hd = value[0: 32]
    accxA = []
    accyA = []
    acczA = []

    ts = np.frombuffer(value[0:8], dtype=np.int64)[0]
    millis = int64_to_str(hd, True)
    GS = 8
    HZ = 12.5
    if hd[8] == 16:
        GS = 8
    elif hd[8] == 8:
        GS = 4
    elif hd[8] == 0:
        GS = 2
    if hd[9] == 0:
        HZ = 12.5
    elif hd[9] == 1:
        HZ = 25
    elif hd[9] == 2:
        HZ = 50
    elif hd[9] == 3:
        HZ = 100
    if HZ == 100:
        HZ = 90  # HACK!!
    delta = False
    deltaval = -1
    packt = 0
    sample = np.zeros(6, dtype='int64')
    # infoStr = "header: " + str(hd) + "\n" do not preceed with #!––
    lbls = []
    itr = 0
    for ii in range(32, len(value) - 3, 3):  # iterate over data
        if (ii - 32) % 7200 == 0:
            pass
            # infoStr += "\n==== new page ====\n"  # mark start of new page
        if not delta:
            if (int(value[ii]) == 255) and (int(value[ii + 1]) == 255) and (packt == 0):  # delta starts
                if int(value[ii + 2]) == 255:
                    pass
                    # infoStr += "\n*" + str((ii + 2)) + "\n"  # error -> this should only happen at the end of a page
                else:
                    # infoStr += "\nd" + value[ii + 2] + ":"
                    delta = True
                    deltaval = int(value[ii + 2])
            else:
                if packt == 0:
                    sample[0] = int(value[ii])
                    sample[1] = int(value[ii + 1])
                    sample[2] = int(value[ii + 2])
                    packt = 1
                else:
                    sample[3] = int(value[ii])
                    sample[4] = int(value[ii + 1])
                    sample[5] = int(value[ii + 2])
                    packt = 0
                    mts = datetime.fromtimestamp(ts / 1000 + itr * (1000 / HZ) / 1000)
                    lbls.append(mts)
                    tmp = np.int16(sample[0] | (sample[1] << 8))
                    accxA.append(round((tmp / 4096), 5))
                    tmp = np.int16(sample[2] | (sample[3] << 8))
                    accyA.append(round((tmp / 4096), 5))
                    tmp = np.int16(sample[4] | (sample[5] << 8))
                    acczA.append(round((tmp / 4096), 5))
                    itr += 1


        else:
            sample[0] = int(value[ii])
            sample[2] = int(value[ii + 1])
            sample[4] = int(value[ii + 2])  # fill LSBs after delta
            mts = datetime.fromtimestamp(ts / 1000 + itr * (1000 / HZ) / 1000)
            lbls.append(mts)
            tmp = np.int16(sample[0] | (sample[1] << 8))
            accxA.append(round((tmp / 4096), 5))
            tmp = np.int16(sample[2] | (sample[3] << 8))
            accyA.append(round((tmp / 4096), 5))
            tmp = np.int16(sample[4] | (sample[5] << 8))
            acczA.append(round((tmp / 4096), 5))
            itr += 1
            deltaval -= 1
            if (deltaval < 0):
                delta = False

    activity_data = np.array([accxA, accyA, acczA], dtype=np.float64).T
    dataframe = pd.DataFrame(data=activity_data, columns=["x_axis", "y_axis", "z_axis"], index=lbls)
    dataframe = dataframe[~dataframe.index.duplicated(keep='first')]

    return dataframe
