import pandas as pd
from definitions import ROOT_DIR

class Data(object):

    def __init__(self):
        self.start2tg = {
            1.5: 468.15,
            2.5: 493.15,
            3.5: 512.15,
            5.0: 530.15,
            10.0: 572.15,
            15.0: 604.15,
            20.0: 631.15,
            25.0: 658.15,
            30.0: 684.15,
        }

        self.end2tg = {
            1.5: 838.15,
            2.5: 863.15,
            3.5: 891.15,
            5.0: 929.15,
            10.0: 975.15,
            15.0: 1036.15,
            20.0: 1061.15,
            25.0: 1082.43,
            30.0: 1116.15
        }

        self.code2method = {
            '100': 'MLP',
            '010': 'RF',
            '001': 'DT'
        }

        data = pd.read_csv('{:}/result/evaluating_range/ranges2.csv'.format(ROOT_DIR))
        self.data = data.iloc[:, 1:]
        self.sdata = None

    def transform(self):
        pass

    def cols_name(self):
        return self.data.columns

    def map_data(self, method=['MLP', 'MLP', 'MLP'], metric='Global_mean_RRMSE'):

        col_1 = 'S_{:}'.format(method[0])
        col_2 = 'M_{:}'.format(method[1])
        col_3 = 'E_{:}'.format(method[2])
        condition = (self.data[col_1] == 1) & (self.data[col_2] == 1) & (self.data[col_3] == 1)
        sdata = self.data[condition]
        x_i = sdata['S']
        x_f = sdata['E']
        y = sdata[metric]
        self.sdata = sdata
        return x_i.tolist(), x_f.tolist(), y.tolist()

    def data_2d(self, value_fixed, metric='RRMSE', variable_fixed='x'):
        if self.sdata is None:
            return [], []
        if variable_fixed == 'x':
            slice_data = self.sdata[self.sdata['S'] == value_fixed]
            x = slice_data['E']
        else:
            slice_data = self.sdata[self.sdata['E'] == value_fixed]
            x = slice_data['S']
        metric_y = 'Global_mean_{:}'.format(metric)
        y = slice_data[metric_y]

        metric_dy = 'Global_sd_{:}'.format(metric)
        dy = slice_data[metric_dy]
        indices = slice_data.index

        return x.tolist(), y.tolist(), dy.tolist(), indices

    def data_errors_range(self, index, metric):
        single_data = self.sdata.loc[index]
        x = [0, 1, 2]
        x_ticks = ['initial', 'center', 'final']
        metric_S = 'Local_S_mean_{:}'.format(metric)
        metric_M = 'Local_M_mean_{:}'.format(metric)
        metric_E = 'Local_E_mean_{:}'.format(metric)
        y = [single_data[metric_S], single_data[metric_M], single_data[metric_E]]
        return x, x_ticks, y
