import glob
import pandas as pd
import numpy as np
import bayesiantests as bt
from scipy.stats import wilcoxon


def relative_deviation(obs, pred):
    return np.sum(np.abs(obs-pred)/obs)/len(obs) * 100


def R2(target, pred):
    return np.corrcoef(target, pred)[0, 1] ** 2


def RRMSE(target, pred):
    num = np.sum((target - pred) ** 2)
    den = np.sum((np.mean(target) - target) ** 2)
    return np.sqrt(num/den)


def RMSE(target, pred):
    N = len(target)
    return np.sqrt(np.sum((target-pred)**2)/N)

def get_tun_result(metric='RMSE'):
    folds = glob.glob("../../result/logs/performance_best_models_tg*.csv")

    lines = []
    idx = pd.read_csv(folds[0], index_col=0).columns.tolist()

    for f in folds:
        data = pd.read_csv(f, index_col=0)
        lines.append(data.loc[metric].values)
    matrix = np.array(lines)

    if metric == 'R2':
        matrix = 1-matrix
    return idx, matrix

def get_def_result(metric='RMSE'):
    folds = glob.glob("../../result/logs/performance_standard_models_tg*.csv")

    lines = []
    idx = pd.read_csv(folds[0], index_col=0).columns.tolist()

    for f in folds:
        data = pd.read_csv(f, index_col=0)
        lines.append(data.loc[metric].values)
    matrix = np.array(lines)

    if metric == 'R2':
        matrix = 1-matrix

    return idx, matrix

def evaluate(y, y_pred, metric):
    if metric == 'RMSE':
        return RMSE(y, y_pred)
    elif metric == 'RRMSE':
        return RMSE(y, y_pred)
    elif metric == 'R2':
        return R2(y, y_pred)
    elif metric == 'relative_deviation':
        return relative_deviation(y, y_pred)
    else:
        raise ValueError("Valid metric are: RMSE, RRMSE, R2 and RD.")

def get_low_middle_high(low, high, metric='RMSE'):
    folds = glob.glob('../../result/logs/predictions_best_models_tg_*.csv')

    idx_low, idx_middle, idx_high = [], [], []
    matrix_low, matrix_middle, matrix_high = [], [], []
    for fold in folds:
        data = pd.read_csv(fold, index_col=0)
        line_h = []
        line_m = []
        line_l = []
        idx = []
        for i in range(0, data.shape[1], 2):
            idx.append(data.columns[i])
            y = data.iloc[:, i]

            idx_h = np.where(y.values >= high)[0]
            idx_m = np.where(np.logical_and(y.values < high, y.values > low))[0]
            idx_l = np.where(y.values <= low)[0]

            y_h = data.iloc[idx_h, i]
            y_pred_h = data.iloc[idx_h, i+1]
            value_h = evaluate(y_h, y_pred_h, metric)
            line_h.append(value_h)

            y_m = data.iloc[idx_m, i]
            y_pred_m = data.iloc[idx_m, i+1]
            value_m = evaluate(y_m, y_pred_m, metric)
            line_m.append(value_m)

            y_l = data.iloc[idx_l, i]
            y_pred_l = data.iloc[idx_l, i+1]
            value_l = evaluate(y_l, y_pred_l, metric)
            line_l.append(value_l)

        matrix_high.append(line_h)
        matrix_middle.append(line_m)
        matrix_low.append(line_l)

        idx_low, idx_middle, idx_high = idx.copy(), idx.copy(), idx.copy()


    if metric == 'R2':
        matrix_low = 1-matrix_low
        matrix_middle = 1-matrix_middle
        matrix_high = 1-matrix_high

    ret = [(idx_low, np.array(matrix_low)),
           (idx_middle, np.array(matrix_middle)),
           (idx_high, np.array(matrix_high))]
    return ret


def matrix_vs_matrix(idx, matrix):
    result = []
    for i, alg1 in enumerate(idx):
        line1 = matrix[:, i]
        aux = []
        for j, alg2 in enumerate(idx):
            line2 = matrix[:, j]
            if i != j:
                w_dif, p_dif = wilcoxon(x=line1, y=line2)
                w_gre, p_gre = wilcoxon(x=line1, y=line2, alternative='greater')
                w_les, p_les = wilcoxon(x=line1, y=line2, alternative='less')
            else:
                w_dif, p_dif = w_gre, p_gre = w_less, p_les = None, None

            if p_dif is not None:
                aux.append((alg1, alg2,{
                    'diff': (w_dif, p_dif, p_dif<0.05),
                    'greater': (w_gre, p_gre, p_gre<0.05),
                    'less': (w_les, p_les, p_les<0.05)}))
            else:
                aux.append((alg1, alg2,{
                    'diff': (None, None, None),
                    'greater': (None, None, None),
                    'less': (None, None, None)}))
        result.append(aux)

    return result

def test_tun_vs_tun(metric):
    idx, matrix = get_tun_result(metric)
    return matrix_vs_matrix(idx, matrix)

def test_tun_vs_tun_lmh(metric):
    idx, matrix = get_tun_result(metric)
    result = []
    for idx, matrix in get_low_middle_high(450, 1150):
        result.append(matrix_vs_matrix(idx, matrix))
    return result

def test_tun_vs_def(metric):
    idx1, matrix1 = get_tun_result(metric)
    idx2, matrix2 = get_def_result(metric)

    result = []
    for i, alg1 in enumerate(idx1):
        line1 = matrix1[:, i]
        line2 = matrix2[:, i]
        aux = []

        w_dif, p_dif = wilcoxon(x=line1, y=line2)
        w_gre, p_gre = wilcoxon(x=line1, y=line2, alternative='greater')
        w_les, p_les = wilcoxon(x=line1, y=line2, alternative='less')
        aux.append((alg1,'Default',{
            'diff': (w_dif, p_dif, p_dif<0.05),
            'greater': (w_gre, p_gre, p_gre<0.05),
            'less': (w_les, p_les, p_les<0.05)}))
        result.append(aux)

    return result

def ger_table(data):
    result = []
    names_lin = []
    names_col = None
    for res in data:
        lin, col = None, None
        names_col = []
        line = []
        for lin, col, dic in res:
            names_col.append(col)
            if dic['diff'][2]:
                if dic['greater'][2]:
                    line.append(-1)
                else:
                    line.append(1)
            else:
                line.append(0)
        result.append(line)
        names_lin.append(lin)

    return pd.DataFrame(result, index=names_lin, columns=names_col)

for metric in ['RMSE', 'RRMSE', 'R2', 'relative_deviation']:
    print(">>>>>>>> Using", metric)
    print()

    result_tun_def = test_tun_vs_def(metric)
    print("tuned vs default")
    print(ger_table(result_tun_def))
    print()

    result_tun_tun = test_tun_vs_tun(metric)
    print("tuned vs tuned")
    print(ger_table(result_tun_tun))
    print()

    result_tun_tun_lmh = test_tun_vs_tun_lmh(metric)
    print("tuned vs tuned - low")
    print(ger_table(result_tun_tun_lmh[0]))
    print()
    print("tuned vs tuned - middle")
    print(ger_table(result_tun_tun_lmh[1]))
    print()
    print("tuned vs tuned - high")
    print(ger_table(result_tun_tun_lmh[2]))
    print()

    ###############################################################################

# stat_test = []
# rope=0.01 # We consider two classifers equivalent when the difference of
#           # accuracy is less that 1%
# rho=1/10  # We are performing 10 folds, cross-validation
#
# for i in range(4, len(idx)-1):
#     print("rf vs ", idx[i])
#     names = [idx[i], "rf"]
#     diff = np.array([(matrix[:, -1]) - matrix[:, i]])
#     print(diff)
#     stat_test.append(bt.hierarchical(
#         diff=diff,
#         rope=rope,
#         rho=rho,
#         lowerAlpha=0.5,
#         upperAlpha=5,
#         lowerBeta=0.05,
#         upperBeta=.15,
#         verbose=True, names=names))
#
# for i in range(0, len(idx)-1):
#     pl, pe, pr = stat_test[i]
#     names = [idx[i],"rf"]
#     print('P({c1} > {c2}) = {pl}, P(rope) = {pe}, P({c2} > {c1}) = {pr}'.
#           format(c1=names[0], c2=names[1], pl=pl, pe=pe, pr=pr))
