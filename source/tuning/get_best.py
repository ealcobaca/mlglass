import pickle
import glob
import os
import numpy as np


def myFunc(data):
    return np.mean(data['errors'])

def get_best(path):
    print(path)
    dirs = [x[0] for x in os.walk(path)]
    del dirs[0]
    fold = -1
    for di in dirs:
        fold += 1
        files = glob.glob("{0}/*.rcfg".format(di))
        # print(files)
        data = [pickle.load(open(f, 'rb')) for f in files]
        data.sort(key=myFunc)
        print("fold{2} mean={0:5.4f} -- std={1:5.4f}".format(
            np.mean(data[0]['errors']),
            np.std(data[0]['errors']), fold))
    print()

def main():
    # d = '.'
    # [os.path.join(d, o) for o in os.listdir(d)
    #  if os.path.isdir(os.path.join(d,o))]
    paths = ["../../result/dt/tg-old",
             "../../result/dt/tg",
             "../../result/rf/tg-old",
             "../../result/rf/tg",
             "../../result/knn/tg",
             "../../result/catboost/tg",
             "../../result/mlp/tg",
             "../../result/svr/tg"]
    [get_best(path) for path in paths]


if __name__ == "__main__":
    main()

