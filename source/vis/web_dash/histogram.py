import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from definitions import ROOT_DIR

if __name__ == '__main__':
    data = pd.read_csv('{:}/data/clean/oxides_Tg_train.csv'.format(ROOT_DIR))

    x = data['Tg'].tolist()
    print(len(x))

    # print(x)
    # print(type(x))


    hist, bin_edges = np.histogram(x, 40)
    with open('data/histogram.dat'.format(ROOT_DIR), 'w') as fd:
        for (h, b) in zip(hist, bin_edges[:-1]):
            fd.write('{:} {:}\n'.format(b, h))

    plt.bar(bin_edges[:-1], hist, 20)
    plt.show()