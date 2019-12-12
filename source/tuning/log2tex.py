import pandas as pd
from os.path import join
from constants import OUTPUT_PATH as input_prefix
from constants import TARGETS_LIST as targets
from constants import REGRESSORS_FORMATTED as regressors
from constants import METRICS_FORMATTED as metrics
from math import log10, floor


def round_to_significant_figures(x):
    return -int(floor(log10(abs(x - int(x)))))


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
                (2*(n_col + 3))*'r')
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
                    standard_digits = round_to_significant_figures(
                        std_standard.loc[metric, reg]
                    )
                    best_digits = round_to_significant_figures(
                        std_best.loc[metric, reg]
                    )

                    line.append(
                        '${0:.{prec_std}f} \\pm {1:.{prec_std}f}$ & ${2:.{prec_best}f} \\pm {3:.{prec_best}f}$'.
                        format(
                            round(mean_standard.loc[metric, reg], standard_digits),
                            round(std_standard.loc[metric, reg], standard_digits),
                            round(mean_best.loc[metric, reg], best_digits),
                            round(std_best.loc[metric, reg], best_digits),
                            prec_std=standard_digits,
                            prec_best=best_digits
                        )
                    )
                line = '{}\\\\\n'.format(' & & '.join(line))
                f.write(line)

            f.write('\t\t\\bottomrule\n')
            f.write('\t\\end{tabular}}\n')
            f.write('\\end{table}\n')


if __name__ == '__main__':
    generate_table()
