import pandas as pd
from os.path import join
from constants import OUTPUT_PATH as input_prefix
from constants import TARGETS_LIST as targets
from constants import REGRESSORS_FORMATTED as regressors
from constants import METRICS_FORMATTED as metrics


def generate_table():
    for target in targets:
        mean_standard = pd.read_csv(
            join(
                input_prefix,
                'mean_performance_standard_{0}_all.csv'.format(target)
            )
        )
        std_standard = pd.read_csv(
            join(
                input_prefix,
                'std_performance_standard_{0}_all.csv'.format(target)
            )
        )
        mean_standard.set_index('metric', inplace=True, drop=True)
        std_standard.set_index('metric', inplace=True, drop=True)

        mean_best = pd.read_csv(
            join(
                input_prefix,
                'mean_performance_best_{0}_all.csv'.format(target)
            )
        )
        std_best = pd.read_csv(
            join(
                input_prefix,
                'std_performance_best_{0}_all.csv'.format(target)
            )
        )
        mean_best.set_index('metric', inplace=True, drop=True)
        std_best.set_index('metric', inplace=True, drop=True)

        n_col = mean_best.shape[1]
        with open(
                join(input_prefix, 'results_table_{}.tex'.format(target)), 'w'
             ) as f:
            f.write('\\begin{table}[!htbp]\n')
            f.write('\t\\setlength{\\tabcolsep}{3pt}\n')
            f.write('\t\\resizebox{\\textwidth}{!}{\n')
            f.write('\t\\begin{{tabular}}{{l{0}}}\n'.format(
                (2*(n_col + 3) - 1)*'r')
            )
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
                        '${0:.2f} \\pm {1:.2f}$ & ${2:.2f} \\pm {3:.2f}$'.
                        format(
                            mean_standard.loc[metric, reg],
                            std_standard.loc[metric, reg],
                            mean_best.loc[metric, reg],
                            std_best.loc[metric, reg]
                        )
                    )
                line = '{}\\\\\n'.format(' & & '.join(line))
                f.write(line)

            f.write('\t\t\\bottomrule\n')
            f.write('\t\\end{tabular}}\n')
            f.write('\\end{table}\n')


if __name__ == '__main__':
    generate_table()
