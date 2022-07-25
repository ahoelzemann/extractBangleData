import os

import numpy as np
import pandas as pd
import plotly.colors
import seaborn as sb
import matplotlib.pyplot as plt


def plot_basketball_classes(subject, activity, place, pages):
    from plotly.subplots import make_subplots
    import plotly.express as px
    import DatasetHangtime
    import plotly.graph_objects as go

    hangtime = DatasetHangtime()
    data = DatasetHangtime.load_activity_of_subject(subject, activity, place)
    # Initialize figure with subplots
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4")
    )

    # Add traces
    colors = px.colors.qualitative.Light24
    step_size = 6
    if pages != 'all':
        pages = pages * step_size
    mode = 'lines'
    for segment in range(0, len(data), step_size):
        if pages != 'all' and segment == pages:
            break
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Segment " + str(segment), "Segment " + str(segment + 1), "Segment " + str(segment + 2),
                            "Segment " + str(segment + 3), "Segment " + str(segment + 4), "Segment " + str(segment + 5))
        )

        sensor_data_0 = data[segment][['timestamp', 'acc_x', 'acc_y', 'acc_z']]
        sensor_data_1 = data[segment + 1][['timestamp', 'acc_x', 'acc_y', 'acc_z']]
        sensor_data_2 = data[segment + 2][['timestamp', 'acc_x', 'acc_y', 'acc_z']]
        sensor_data_3 = data[segment + 3][['timestamp', 'acc_x', 'acc_y', 'acc_z']]
        sensor_data_4 = data[segment + 4][['timestamp', 'acc_x', 'acc_y', 'acc_z']]
        sensor_data_5 = data[segment + 5][['timestamp', 'acc_x', 'acc_y', 'acc_z']]

        fig.add_trace(
            go.Scatter(x=sensor_data_0['timestamp'], y=sensor_data_0[sensor_data_0.columns[1]], showlegend=True,
                       mode=mode,
                       name='x-axis', legendgroup='x-axis',
                       line=dict(width=2, color=colors[0])), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=sensor_data_0['timestamp'], y=sensor_data_0[sensor_data_0.columns[2]], showlegend=True,
                       mode=mode,
                       name='y-axis', legendgroup='y-axis',
                       line=dict(width=2, color=colors[8])),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=sensor_data_0['timestamp'], y=sensor_data_0[sensor_data_0.columns[3]], showlegend=True,
                       mode=mode,
                       name='z-axis', legendgroup='z-axis',
                       line=dict(width=2, color=colors[13])),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=sensor_data_1['timestamp'], y=sensor_data_1[sensor_data_1.columns[1]], showlegend=False,
                       mode=mode,
                       name='x-axis', legendgroup='x-axis',
                       line=dict(width=2, color=colors[0])),
            row=1, col=2)
        fig.add_trace(
            go.Scatter(x=sensor_data_1['timestamp'], y=sensor_data_1[sensor_data_1.columns[2]], showlegend=False,
                       mode=mode,
                       name='y-axis', legendgroup='y-axis',
                       line=dict(width=2, color=colors[8])),
            row=1, col=2)
        fig.add_trace(
            go.Scatter(x=sensor_data_1['timestamp'], y=sensor_data_1[sensor_data_1.columns[3]], showlegend=False,
                       mode=mode,
                       name='z-axis', legendgroup='z-axis',
                       line=dict(width=2, color=colors[13])),
            row=1, col=2)
        fig.add_trace(
            go.Scatter(x=sensor_data_2['timestamp'], y=sensor_data_2[sensor_data_2.columns[1]], showlegend=False,
                       mode=mode,
                       name='x-axis', legendgroup='x-axis',
                       line=dict(width=2, color=colors[0])),
            row=2, col=1)
        fig.add_trace(
            go.Scatter(x=sensor_data_2['timestamp'], y=sensor_data_2[sensor_data_2.columns[2]], showlegend=False,
                       mode=mode,
                       name='y-axis', legendgroup='y-axis',
                       line=dict(width=2, color=colors[8])),
            row=2, col=1)
        fig.add_trace(
            go.Scatter(x=sensor_data_2['timestamp'], y=sensor_data_2[sensor_data_2.columns[3]], showlegend=False,
                       mode=mode,
                       name='z-axis', legendgroup='z-axis',
                       line=dict(width=2, color=colors[13])),
            row=2, col=1)
        fig.add_trace(
            go.Scatter(x=sensor_data_3['timestamp'], y=sensor_data_3[sensor_data_3.columns[1]], showlegend=False,
                       mode=mode,
                       name='x-axis', legendgroup='x-axis',
                       line=dict(width=2, color=colors[0])),
            row=2, col=2)
        fig.add_trace(
            go.Scatter(x=sensor_data_3['timestamp'], y=sensor_data_3[sensor_data_3.columns[2]], showlegend=False,
                       mode=mode,
                       name='y-axis', legendgroup='y-axis',
                       line=dict(width=2, color=colors[8])),
            row=2, col=2)
        fig.add_trace(
            go.Scatter(x=sensor_data_3['timestamp'], y=sensor_data_3[sensor_data_3.columns[3]], showlegend=False,
                       mode=mode,
                       name='z-axis', legendgroup='z-axis',
                       line=dict(width=2, color=colors[13])),
            row=2, col=2)
        fig.add_trace(
            go.Scatter(x=sensor_data_4['timestamp'], y=sensor_data_4[sensor_data_4.columns[1]], showlegend=False,
                       mode=mode,
                       name='x-axis', legendgroup='x-axis',
                       line=dict(width=2, color=colors[0])),
            row=3, col=1)
        fig.add_trace(
            go.Scatter(x=sensor_data_4['timestamp'], y=sensor_data_4[sensor_data_4.columns[2]], showlegend=False,
                       mode=mode,
                       name='y-axis', legendgroup='y-axis',
                       line=dict(width=2, color=colors[8])),
            row=3, col=1)
        fig.add_trace(
            go.Scatter(x=sensor_data_4['timestamp'], y=sensor_data_4[sensor_data_4.columns[3]], showlegend=False,
                       mode=mode,
                       name='z-axis', legendgroup='z-axis',
                       line=dict(width=2, color=colors[13])),
            row=3, col=1)
        fig.add_trace(
            go.Scatter(x=sensor_data_5['timestamp'], y=sensor_data_5[sensor_data_5.columns[1]], showlegend=False,
                       mode=mode,
                       name='x-axis', legendgroup='x-axis',
                       line=dict(width=2, color=colors[0])),
            row=3, col=2)
        fig.add_trace(
            go.Scatter(x=sensor_data_5['timestamp'], y=sensor_data_5[sensor_data_5.columns[2]], showlegend=False,
                       mode=mode,
                       name='y-axis', legendgroup='y-axis',
                       line=dict(width=2, color=colors[8])),
            row=3, col=2)
        fig.add_trace(
            go.Scatter(x=sensor_data_5['timestamp'], y=sensor_data_5[sensor_data_5.columns[3]], showlegend=False,
                       mode=mode,
                       name='z-axis', legendgroup='z-axis',
                       line=dict(width=2, color=colors[13])),
            row=3, col=2)

        # Update xaxis properties
        fig.update_xaxes(title_text="Time", row=1, col=1, showgrid=False)
        fig.update_xaxes(title_text="Time", row=1, col=2, showgrid=False)
        fig.update_xaxes(title_text="Time", row=2, col=1, showgrid=False)
        fig.update_xaxes(title_text="Time", row=2, col=2, showgrid=False)
        fig.update_xaxes(title_text="Time", row=3, col=1, showgrid=False)
        fig.update_xaxes(title_text="Time", row=3, col=2, showgrid=False)

        # Update yaxis properties
        fig.update_yaxes(title_text="Acceleration in g", row=1, col=1, showgrid=False)
        fig.update_yaxes(title_text="Acceleration in g", row=1, col=2, showgrid=False)
        fig.update_yaxes(title_text="Acceleration in g", row=2, col=1, showgrid=False)
        fig.update_yaxes(title_text="Acceleration in g", row=2, col=2, showgrid=False)
        fig.update_yaxes(title_text="Acceleration in g", row=3, col=1, showgrid=False)
        fig.update_yaxes(title_text="Acceleration in g", row=3, col=2, showgrid=False)

        # Update title and height
        fig.update_layout(title_text="Subject: <b>" + subject + "<br>" + "</b>Activity: <b>" + activity + "</b>")

        fig.show()


def plot_novice_vs_expert(novices, experts):
    from plotly.subplots import make_subplots
    import plotly.express as px
    import DatasetHangtime
    import plotly.graph_objects as go

    colors = px.colors.qualitative.Light24

    novices_keys = list(novices.keys())
    novices_data = list(novices.values())
    experts_keys = list(experts.keys())
    experts_data = list(experts.values())
    # hangtime = DatasetHangtime()
    # data = DatasetHangtime.load_activity_of_subject(subject, activity, place)
    # Initialize figure with subplots
    fig = make_subplots(
        rows=3, cols=2, subplot_titles=(
            'Novice: ' + novices_keys[0], "Expert: " + experts_keys[0], 'Novice: ' + novices_keys[1],
            "Expert: " + experts_keys[1], 'Novice: ' + novices_keys[2], "Expert: " + experts_keys[2])
    )
    # n = 200000
    for x in range(0, 3):


        for y in range(0, 2):
            showlegend = False
            if x == 0 and y == 0:
                showlegend = True
            if y == 0:
                xs = novices_data[x]['timestamp']
                ys_accx = novices_data[x]['acc_x']
                ys_accy = novices_data[x]['acc_y']
                ys_accz = novices_data[x]['acc_z']
            else:
                xs = experts_data[x]['timestamp']
                ys_accx = experts_data[x]['acc_x']
                ys_accy = experts_data[x]['acc_y']
                ys_accz = experts_data[x]['acc_z']
            # Add traces
            fig.add_trace(
                go.Scatter(x=xs, y=ys_accx, showlegend=showlegend,
                           mode='lines',
                           name='x-axis', legendgroup='x-axis',
                           line=dict(width=2, color=colors[0])), row=x + 1, col=y + 1)
            fig.add_trace(
                go.Scatter(x=xs, y=ys_accy, showlegend=showlegend,
                           mode='lines',
                           name='y-axis', legendgroup='y-axis',
                           line=dict(width=2, color=colors[8])), row=x + 1, col=y + 1)
            fig.add_trace(
                go.Scatter(x=xs, y=ys_accz, showlegend=showlegend,
                           mode='lines',
                           name='z-axis', legendgroup='z-axis',
                           line=dict(width=2, color=colors[13])), row=x + 1, col=y + 1)
            # Update xaxis properties
            fig.update_xaxes(title_text="Time", row=x + 1, col=y + 1, showgrid=False)

            # Update yaxis properties
            fig.update_yaxes(title_text="Acceleration in g", row=x + 1, col=y + 1, showgrid=False)

    # Update title and height
    fig.update_layout(title_text="Novices vs. Experts</b>")

    fig.show()


def plot_novice_vs_expert_fft(novices, experts):
    from plotly.subplots import make_subplots
    import plotly.express as px
    import DatasetHangtime
    import plotly.graph_objects as go

    colors = px.colors.qualitative.Light24

    # novices_keys = list(novices.keys())
    # novices_data = list(novices.values())
    # experts_keys = list(experts.keys())
    # experts_data = list(experts.values())

    # fig = make_subplots(
    #     rows=3, cols=2, subplot_titles=(
    #         'Novice: ' + novices_keys[0], "Expert: " + experts_keys[0], 'Novice: ' + novices_keys[1],
    #         "Expert: " + experts_keys[1], 'Novice: ' + novices_keys[2], "Expert: " + experts_keys[2])
    # )
    fig = make_subplots(
        rows=3, cols=2, subplot_titles=(
            "Novice: 1", "Expert: 1", "Novice: 2",
            "Expert: 2", "Novice: 3", "Expert: 3")
    )
    # n = 200000
    for x in range(0, 3):
        for y in range(0, 2):
            if y == 0:
                fft_result = novices[x]
                # xs = novices_data[x]['timestamp']
                # ys_accx = novices_data[x]['acc_x']
                # ys_accy = novices_data[x]['acc_y']
                # ys_accz = novices_data[x]['acc_z']
            else:
                fft_result = experts[x]
                # xs = experts_data[x]['timestamp']
                # ys_accx = experts_data[x]['acc_x']
                # ys_accy = experts_data[x]['acc_y']
                # ys_accz = experts_data[x]['acc_z']
            # Add traces
            fig.add_trace(
                go.Scatter(y=fft_result, showlegend=False,
                           mode='lines',
                           name='fft', legendgroup='fft',
                           line=dict(width=2, color=colors[13])), row=x + 1, col=y + 1)
            # Update xaxis properties
            fig.update_xaxes(title_text="Frequency", row=x + 1, col=y + 1, showgrid=False)

            # Update yaxis properties
            fig.update_yaxes(title_text="Magnitude", row=x + 1, col=y + 1, showgrid=False)

    # Update title and height
    fig.update_layout(title_text="Novices vs. Experts</b>")

    fig.show()

def vizualize_one_player(player_data, player_id):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=player_data['timestamp'], y=player_data['acc_x'],
                             # mode='lines',
                             name='x-axis'))
    fig.add_trace(go.Scatter(x=player_data['timestamp'], y=player_data['acc_y'],
                             mode='lines',
                             name='y-axis'))
    fig.add_trace(go.Scatter(x=player_data['timestamp'], y=player_data['acc_z'],
                             mode='lines',
                             name='z-axis'))

    fig.update_traces(marker_line_width=4)
    # fig = px.line(x=ts, y=ys.loc[:, ['acc_z']].values.tolist())
    fig.show()


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

    # Todo add HR and Body-Temp to subplots

    plt.show()


def plot_peaks(df, peaks, dribblings_per_second, subject, place):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df,
                             mode='lines',
                             name='lines'))
    fig.add_trace(go.Scatter(x=peaks.index, y=peaks,
                             mode='markers',
                             name='markers', connectgaps=False))
    fig.update_layout(
        title_text="City: <b>" + place + "</b>, " + "Subject: <b>" + subject + "<br>" + "</b>Activity: <b>dribbling</b><br>Dribblings/sec: <b>" + str(
            dribblings_per_second) + "</b>")
    fig.show()

# plot_basketball_classes("10f0", "shot", "Siegen", 'all')

# all = False
# data = []
# subject = '2dd9' #
# rootdir = '/Users/alexander/Documents/Resources/decompressed/Boulder/'
# time_as_indices = False
# if all:
#     files = os.listdir(rootdir)
#     files.remove(".DS_Store")
#     for filenumber in range(0, len(files)):
#         if files[filenumber] != ".DS_Store":
#             file = rootdir + files[filenumber]
#             # file2 = rootdir + files[filenumber+1]
#             plot_imu_data(pd.read_csv(file).to_numpy(), file, time_as_indices=time_as_indices)
#             data.append(pd.read_csv(file))
# else:
#     rootdir = rootdir + subject + ".csv"
#     plot_imu_data(pd.read_csv(rootdir).to_numpy(), rootdir, time_as_indices=time_as_indices)
