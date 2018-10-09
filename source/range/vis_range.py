import pickle
from os import listdir
from os.path import isfile, join

def aggr(mypath="../../result/result_low/"):
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    path_files = [join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f))]

    for f, path in zip():
        exp_range = f.split('_')[1]
        exp_alg = f.split('_')[2].split('.')[0]
        np.arange(start=1, stop=value+1,step=1)
    # result = pickle.load( open(path_files, "rb" ))


np.mean(result, axis=0)
np.std(result, axis=0)
