import pandas as pd
from os.path import join


input_prefix = '../../result'

standard = pd.read_csv(join(input_prefix, 'performance_standard_models.csv'))
cols = list(standard.columns)
cols[0] = 'metric'
standard.columns = cols
standard.set_index('metric', inplace=True, drop=True)
best = pd.read_csv(join(input_prefix, 'performance_best_models.csv'))
n_col = best.shape[1]
cols = list(best.columns)
cols[0] = 'metric'
best.columns = cols
best.set_index('metric', inplace=True, drop=True)


regressors = {
    'dt': 'DT',
    'knn': 'k-NN',
    'mlp': 'MLP',
    'svr': 'SVR',
    'rf': 'RF',
    # 'dt': 'DT',
}

metrics = {
    'relative_deviation': 'RD',
    'R2': '$R^2$',
    'RMSE': 'RMSE',
    'RRMSE': 'RRMSE'
}

with open(join(input_prefix, 'results_table.tex'), 'w') as f:
    f.write('\\begin{table}[!htbp]\n')
    f.write('\t\\setlength{\\tabcolsep}{3pt}\n')
    f.write('\t\\begin{{tabular}}{{{0}}}\n'.format((2*(n_col + 2))*'c'))
    f.write('\t\t\\toprule\n')
    header = '\t\t\\multirow{2}{*}{Metric} & & '
    ftechs = []
    for freg in regressors.values():
        ftechs.append('\\multicolumn{{2}}{{c}}{{{0}}}'.format(freg))
    header = '{0}{1}\\\\\n'.format(header, ' & & '.join(ftechs))
    f.write(header)
    hlines = ['\t\t']
    ci = 3
    for reg in regressors.keys():
        hlines.append('\\cline{{{0}-{1}}}'.format(ci, ci+1))
        ci += 3
    hlines = '{}\n'.format(' '.join(hlines))
    f.write(hlines)
    sub_header = ['\t\t ']
    for reg in regressors.keys():
        sub_header.append('Default & Tuning'.format(ci, ci+1))
    sub_header = ' & & '.join(sub_header)
    f.write(sub_header + '\\\\\n')
    f.write('\t\t\\midrule\n')
    for metric in metrics.keys():
        line = ['\t\t{}'.format(metrics[metric])]
        for reg in regressors.keys():
            line.append(
                '{:.4f} & {:.4f}'.format(
                    standard.loc[metric, reg], best.loc[metric, reg]
                )
            )
        line = '{}\\\\\n'.format(' & & '.join(line))
        f.write(line)

    f.write('\t\t\\bottomrule\n')
    f.write('\t\\end{tabular}\n')
    f.write('\\end{table}\n')
