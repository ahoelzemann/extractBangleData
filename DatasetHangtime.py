import pandas as pd
import experiments
import viz
import Util


class HangtimeUtils:

    def __init__(self, place, rootpath):

        self.path = rootpath + "/" + place + "/"
        # self.path = rootpath + "/"
        self.players = {}
        # self.data = {}
        # self.skills = {}
        # self.activity_indices = {}
        # self.features = {}

    def load_data_of_subject(self, subject, skill):
        path = self.path + "no_void/"+ subject + ".csv"
        df = pd.read_csv(path)
        self.players[subject] = {}
        self.players[subject]['data'] = df
        self.players[subject]['skill'] = skill
        self.players[subject]['features'] = {}
        self.players[subject]['activity_indices'] = {}

    def load_activity_of_subjects(self, activity):

        # path = self.path + ".csv"
        # df = pd.read_csv(path)
        if activity == 'dribbling' or activity == 'shot' or activity == 'layup' or activity == 'rebound' or activity == 'pass':
            layer = 'basketball'
        else:
            layer = 'locomotion'
        import numpy as np

        for subject in self.players:
            self.players[subject]['activity_indices'][activity] = {}
            starts = []
            ends = []
            activity_indices = np.where(self.players[subject]['data'][layer].str.match(activity) == True)
            starts.append(activity_indices[0][0])
            for index in range(1, activity_indices[0].shape[0]):
                current = activity_indices[0][index]
                last = activity_indices[0][index - 1]
                if current - last > 1:
                    starts.append(current)
                    ends.append(last)
            ends.append(activity_indices[0][-1])
            longest_intervall = np.subtract(ends,
                                            starts).argmax()
            self.players[subject]['activity_indices'][activity] = {
                'starts': starts,
                'ends': ends,
                'longest_intervall': longest_intervall
            }

    def subject_analysis(self, subject, place, printing=False):

        dribbleCounter = _dribbleCounter(subject, self.load_activity_of_subject(subject, "dribbling"), place, printing)
        print(dribbleCounter)


def sort_experts_and_novices(hangtime_si, hangtime_bo):
    experts = {}
    novices = {}
    for player_id in hangtime_si.data:
        if hangtime_si.skills[player_id] == 'novice':
            novices[player_id + '_si'].data = hangtime_si.data[player_id]
            novices[player_id + '_si'].features = hangtime_si.data[player_id]
            novices[player_id + '_si'].skills = hangtime_si.data[player_id]
        else:
            experts[player_id + '_si'].skill = hangtime_si.data[player_id]
    for player_id in hangtime_bo.data:
        if hangtime_bo.skills[player_id].data == 'novice':
            novices[player_id + '_bo'].data = hangtime_bo.data[player_id]
        else:
            experts[player_id + '_bo'].data = hangtime_bo.data[player_id]

    return novices, experts


# place = 'Siegen'
# hangtime_si = HangtimeUtils('Siegen', '/Users/alexander/Documents/Resources/ready')
# place = 'Boulder'
hangtime_si = HangtimeUtils('Siegen', 'G:/hangtime/ready')
hangtime_bo = HangtimeUtils('Boulder', 'G:/hangtime/ready')
# hangtime_si = HangtimeUtils('Siegen', '/Users/alexander/Documents/Resources/ready')
# place = 'Boulder'
# hangtime_bo = HangtimeUtils('Boulder', '/Users/alexander/Documents/Resources/ready')
# hangtime_si.load_data_of_subject('0846', 'novice')
hangtime_si.load_data_of_subject('f2ad', 'expert')
# hangtime_si.load_data_of_subject('e90f', 'expert')
# hangtime_si.load_data_of_subject('ac59', 'expert')
# hangtime_si.load_data_of_subject('b512', 'expert')
# hangtime_si.load_data_of_subject('2dd9', 'expert')
# hangtime_si.load_data_of_subject('4d70', 'expert')
hangtime_si.load_data_of_subject('10f0', 'expert')
# hangtime_si.load_data_of_subject('ce9d', 'expert')
hangtime_si.load_data_of_subject('05d8', 'expert')
hangtime_bo.load_data_of_subject('2dd9', 'novice')
# hangtime_bo.load_data_of_subject('ac59', 'expert')
hangtime_bo.load_data_of_subject('ce9d', 'novice')
# hangtime_bo.load_data_of_subject('b512', 'expert')
hangtime_bo.load_data_of_subject('c6f3', 'novice')

# novices_and_experts = novices.update(experts)
# experiments.show_periodicity_novices_experts(novices, experts)
# experiments.novice_vs_expert_dribbling(novices, experts)
# viz.vizualize_one_player(hangtime_si.players['10f0']['data'], show=True, xticks='linear')
hangtime_si = Util.calc_magnitude_wrapper(hangtime_si)
hangtime_bo = Util.calc_magnitude_wrapper(hangtime_bo)
hangtime_si = experiments.feature_analysis(hangtime_si)
hangtime_bo = experiments.feature_analysis(hangtime_bo)
hangtime_si.load_activity_of_subjects('dribbling')
hangtime_si.load_activity_of_subjects('shot')
hangtime_si.load_activity_of_subjects('layup')
hangtime_si.load_activity_of_subjects('pass')
hangtime_si.load_activity_of_subjects('rebound')
hangtime_si.load_activity_of_subjects('running')
hangtime_si.load_activity_of_subjects('walking')
hangtime_si.load_activity_of_subjects('sitting')
hangtime_si.load_activity_of_subjects('standing')
hangtime_si.load_activity_of_subjects('jumping')
hangtime_bo.load_activity_of_subjects('dribbling')
hangtime_bo.load_activity_of_subjects('shot')
hangtime_bo.load_activity_of_subjects('layup')
# hangtime_bo = experiments.feature_analysis_activity(hangtime_bo, 'dribbling', mode='longest_intervall')
# hangtime_si = experiments.feature_analysis_activity(hangtime_si, 'dribbling', mode='longest_intervall')
hangtime_si = experiments.feature_analysis_activity(hangtime_si, 'shot', mode='complete_signal')
hangtime_bo = experiments.feature_analysis_activity(hangtime_bo, 'shot', mode='complete_signal')
hangtime_si = experiments.feature_analysis_activity(hangtime_si, 'layup', mode='complete_signal')
hangtime_bo = experiments.feature_analysis_activity(hangtime_bo, 'layup', mode='complete_signal')

# viz.plot_class_scatter_plots(hangtime_si, hangtime_bo, 'axis_sums', activity=['shot', 'layup'])

indices = [[62489, 63489], [73400, 74400], [83765, 84765], [89175, 90175], [139490, 140490],
           [196480, 196630], [1526, 1676], [19450, 19600], [19350, 19500] , [19400, 19450]]
activities = ['sitting', 'standing', 'walking', 'running', 'dribbling',
              'shot', 'layup', 'pass', 'rebound', 'jumping']
viz.plot_class_data(hangtime_si.players['10f0'], indices, activities)

# viz.vizualize_one_player(hangtime_si.players['4d70']['data'], show=True)
# novices, experts = sort_experts_and_novices(hangtime_si, hangtime_bo)
# viz.plot_novice_vs_expert(novices=novices, experts=experts)
# labeled_subjects = ["10f0", "2dd9", "4d70", "9bd4", "ce9d", "f2ad", "ac59", "0846", "a0da", "b512", "e90f", "4991",
#                     "05d8"]
# viz.plot_novice_vs_expert(hangtime_si, hangtime_bo, activity='layup')
# for p in hangtime_si.players.keys():
#     print("Hangtime_SI Dribblings per seconds for " + p + ": " + str(hangtime_si.players[p]['features']['dribbling']['dribblings_per_seconds']))
#     print("Hangtime_SI snr for " + p + ": " + str(
#         hangtime_si.players[p]['features']['dribbling']['snr']))
# for p in hangtime_bo.players.keys():
#     print("Hangtime_BO Dribblings per seconds for " + p + ": " + str(hangtime_bo.players[p]['features']['dribbling']['dribblings_per_seconds']))
#     print("Hangtime_BO snr for " + p + ": " + str(
#         hangtime_bo.players[p]['features']['dribbling']['snr']))
# viz.plot_full_feature_analysis(hangtime_si, hangtime_bo, (['10f0_eu', '05d8_eu', 'f2ad_eu'], ['2dd9_na', 'c6f3_na', 'ce9d_na']), activity='dribbling')

# # labeled_subjects = ['05d8']
# for subject in labeled_subjects:
#     hangtime_si.subject_analysis(subject, place, printing=True)
