import pandas as pd


def load_activity_of_subject(subject, activity, place):
    path = ""
    if place == "Boulder":
        path = "/Users/alexander/Documents/Resources/ready/Boulder/"
    else:
        path = "/Users/alexander/Documents/Resources/ready/Siegen/"
    path = path + subject + ".csv"
    df = pd.read_csv(path)
    df = df[df['basketball'].str.match(activity)]
    indices = df.index
    results = []

    starts = []
    ends = []
    starts.append(indices[0])

    for i in range(len(indices)-1):
        current = indices[i]
        if indices[i+1] - current > 1:
            starts.append(indices[i+1])
            ends.append(indices[i])
    ends.append(indices[-1])

    for e in range(len(starts)):
        results.append(df.loc[starts[e]: ends[e]])
    return results

class ActivityLoader:
    pass
