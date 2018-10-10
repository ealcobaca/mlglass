import pickle
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

def aggr(read_path, save_path):
    files = [f for f in listdir(read_path) if isfile(join(read_path, f))]
    path_files = [join(read_path,f) for f in listdir(read_path) if isfile(join(read_path, f))]

    data=[]
    for f, path in zip(files, path_files):
        exp_range = f.split('_')[1]
        exp_alg = f.split('_')[2].split('.')[0]
        result = pickle.load( open(path, "rb" ))
        [(r.append(exp_range),r.append(exp_alg), data.append(r)) for r in result]

    df = pd.DataFrame(data=data, columns=["mean_absolute_error",
                                          "mean_absolute_error",
                                          "r2_score",
                                          "RRMSE",
                                          "RMSE",
                                          "range",
                                          "alg"])

    if save_path != None:
        return df.to_csv(save_path)

    return df


def aggr_all(read_path, save_path, data_path, str_class):
    files = [f for f in listdir(read_path) if isfile(join(read_path, f))]
    path_files = [join(read_path,f) for f in listdir(read_path) if isfile(join(read_path, f))]

    tot_cv = 10*5
    dt = pd.read_csv(data_path)
    X = dt.drop([str_class], axis=1).values
    y = dt[str_class].values

    perceltil_inf = [5*i for i in range(1,9)]
    perceltil_sup = [(100-perceltil_inf[i]) for i in range(len(perceltil_inf))]
    range_high_TG_per = [np.percentile(y, perceltil_sup[i]) for i in range(len(perceltil_sup))]
    range_low_TG_per = [np.percentile(y, perceltil_inf[i]) for i in range(len(perceltil_inf))]
    range_low_TG = np.arange(start=400, stop=650+1, step=25)
    range_high_TG = np.arange(start=900, stop=1150+1, step=25)

    data=[]
    for f, path in zip(files, path_files):
        exp_range = f.split('_')[1]
        exp_alg = f.split('_')[2].split('.')[0]
        if exp_alg in ("high", "low"):
            alg = f.split('_')[3].split('.')[0]
            result = pickle.load(open(path, "rb" ))
            if(exp_alg == "high"):
                if("percentil" in read_path.split("/")[3].split("_")):
                    ran = range_high_TG_per * tot_cv
                else:
                    ran = range_high_TG.tolist() * tot_cv
            else:
                if("percentil" in read_path.split("/")[3].split("_")):
                    ran = range_low_TG_per * tot_cv
                else:
                    ran = range_low_TG.tolist() * tot_cv
            result = [j for i in result for j in i]
            [(r.append(ran_i),r.append(alg), data.append(r)) for r, ran_i in zip(result, ran)]
        else:
            result = pickle.load(open(path, "rb" ))
            [(r.append(exp_range),r.append(exp_alg), data.append(r)) for r in result]

    df = pd.DataFrame(data=data, columns=["mean_absolute_error",
                                          "mean_absolute_error",
                                          "r2_score",
                                          "RRMSE",
                                          "RMSE",
                                          "range",
                                          "alg"])
    if save_path != None:
        return df.to_csv(save_path)

    return df

def plot_bar(alg, save_path, measure, percentil=False):
    title = None
    if percentil == True:
        title = alg+": All X Range (percentile)"
        save_path = save_path+alg+"_"+measure+"_allXrange_percentile.html"
        df_all = pd.read_csv('../../result/aggr/result_all_percentil.csv')
        df_high = pd.read_csv('../../result/aggr/result_high_percentil.csv')
        df_low = pd.read_csv('../../result/aggr/result_low_percentil.csv')
    else:
        title = alg+": All X Range"
        save_path = save_path+alg+"_"+measure+"_allXrange.html"
        df_all = pd.read_csv('../../result/aggr/result_all.csv')
        df_high = pd.read_csv('../../result/aggr/result_high.csv')
        df_low = pd.read_csv('../../result/aggr/result_low.csv')
        df_all = df_all[df_all['range'] != '400']

    df = df_all[df_all["alg"] == alg]
    trace0 = go.Box(
	y=df[measure],
	x=df['range'],
	name="all",
	marker=dict(color='#1f78b4'))

    df_l = df_low[df_low["alg"] == alg]
    df_h = df_high[df_high["alg"] == alg]
    df = pd.concat([df_l, df_h])
    trace1 = go.Box(
	y=df[measure],
	x=df['range'],
	name="range",
	marker=dict(color='#33a02c'))

    data = [trace0, trace1]
    layout = go.Layout(
	yaxis=dict(title=measure, zeroline=False),
	xaxis=dict(title='Tg', zeroline=False),
	title=title,
	boxmode='group')
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=save_path)

def plot_bar_algs(save_path, measure, percentil=False):
    title = None
    if percentil == True:
        title = "All algorithms: Range (percentile)"
        save_path = save_path+"_"+measure+"_all_algorithm_range_percentile.html"
        df_high = pd.read_csv('../../result/aggr/result_high_percentil.csv')
        df_low = pd.read_csv('../../result/aggr/result_low_percentil.csv')
    else:
        title = "All algorithms: Range"
        save_path = save_path+"_"+measure+"_all_algorithm_range.html"
        df_high = pd.read_csv('../../result/aggr/result_high.csv')
        df_low = pd.read_csv('../../result/aggr/result_low.csv')

    colors = ["#33a02c", "#fb9a99", "#a6cee3", "#7570b3"]
    algs = ["DT", "MLP", "SVM", "RF"]
    data = []
    for alg, col in zip(algs, colors):
        df_l = df_low[df_low["alg"] == alg]
        df_h = df_high[df_high["alg"] == alg]
        df = pd.concat([df_l, df_h])

        trace = go.Box(
            y=df[measure],
            x=df['range'],
            name=alg,
            marker=dict(color=col))
        data.append(trace)

    layout = go.Layout(
	yaxis=dict(title=measure, zeroline=False),
	xaxis=dict(title='Tg', zeroline=False),
	title=title,
	boxmode='group')
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=save_path)

def plot_bar_all(save_path, measure, percentil=False):
    title = None
    if percentil == True:
        title = "All algorithms: all data (percentile)"
        save_path = save_path+measure+"_all_algorithm_percentile.html"
        df_all = pd.read_csv('../../result/aggr/result_all_percentil.csv')
    else:
        title = "All algorithms: all data"
        save_path = save_path+measure+"_all_algorithm.html"
        df_all = pd.read_csv('../../result/aggr/result_all.csv')

    colors = ["#33a02c", "#fb9a99", "#a6cee3", "#7570b3"]
    algs = ["DT", "MLP", "SVM", "RF"]
    data = []
    for alg, col in zip(algs, colors):
        df = df_all[df_all["alg"] == alg]
        df = df[df["range"] == "all"]
        trace = go.Box(
            y=df[measure],
            x=df['alg'],
            name=alg,
            marker=dict(color=col))
        data.append(trace)

    layout = go.Layout(
	yaxis=dict(title=measure, zeroline=False),
	xaxis=dict(title='Tg', zeroline=False),
	title=title,
	boxmode='group')
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=save_path)


def plot_bar_algs(save_path, measure, percentil=False):
    title = None
    if percentil == True:
        title = "All algorithms: Range (percentile)"
        save_path = save_path+measure+"_all_algorithm_range_percentile.html"
        df_high = pd.read_csv('../../result/aggr/result_high_percentil.csv')
        df_low = pd.read_csv('../../result/aggr/result_low_percentil.csv')
    else:
        title = "All algorithms: Range"
        save_path = save_path+measure+"_all_algorithm_range.html"
        df_high = pd.read_csv('../../result/aggr/result_high.csv')
        df_low = pd.read_csv('../../result/aggr/result_low.csv')

    colors = ["#33a02c", "#fb9a99", "#a6cee3", "#7570b3"]
    algs = ["DT", "MLP", "SVM", "RF"]
    data = []
    for alg, col in zip(algs, colors):
        df_l = df_low[df_low["alg"] == alg]
        df_h = df_high[df_high["alg"] == alg]
        df = pd.concat([df_l, df_h])

        trace = go.Box(
            y=df[measure],
            x=df['range'],
            name=alg,
            marker=dict(color=col))
        data.append(trace)

    layout = go.Layout(
	yaxis=dict(title=measure, zeroline=False),
	xaxis=dict(title='Tg', zeroline=False),
	title=title,
	boxmode='group')
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=save_path)


def run():

    result_path = "../../result/aggr/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    aggr("../../result/result_low/", "../../result/aggr/result_low.csv")
    aggr("../../result/result_high/", "../../result/aggr/result_high.csv")
    aggr("../../result/result_low_percentil/", "../../result/aggr/result_low_percentil.csv")
    aggr("../../result/result_high_percentil/", "../../result/aggr/result_high_percentil.csv")
    aggr_all("../../result/result_all/", "../../result/aggr/result_all.csv", "../../data/clean/oxides_Tg_train.csv", "Tg")
    aggr_all("../../result/result_all_percentil/", "../../result/aggr/result_all_percentil.csv", "../../data/clean/oxides_Tg_train.csv", "Tg")

    save_path = "../../result/plots/boxplot-range/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    algs = ["DT", "MLP", "SVM", "RF"]
    # measure=["mean_absolute_error", "mean_absolute_error", "r2_score", "RRMSE","RMSE"]
    measure=["RMSE"]

    for alg in algs:
        for me in measure:
            plot_bar(alg, save_path, me, True)
            plot_bar(alg, save_path, me, False)
    plot_bar_algs(save_path, measure[0], percentil=False)
    plot_bar_algs(save_path, measure[0], percentil=True)

    plot_bar_all(save_path, measure[0], percentil=False)

run()
