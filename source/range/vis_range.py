import pickle

result = pickle.load( open( "../../result/result_all/result_all_DT.csv", "rb" ))
np.mean(result, axis=0)
np.std(result, axis=0)
