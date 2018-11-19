import pandas as pd
import numpy as np
import pickle
import os
from range.regressors import compute_performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot


path_result = "../result/result_oracle/default-model/"

def lineplot_local_evaluation(
    paths=["../result/result_oracle/default-model/mode_test_.list", "../result/baselines/log/predictions_raw_RF.list",
           "../result/result_oracle/default-model/modeoverlap_test_.list", "../result/baselines/log/predictions_raw_MLP.list"]):

    for path, name in zip(paths, ["mode","RF", "mode_overlap", "MLP"]):
        print(path)
        if str.find(path, 'baselines') >= 0:
            _, y_true, y_pred = pickle.load(open(path, 'rb'))
        else:
            y_true, y_pred = pickle.load(open(path, 'rb'))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Create a trace
        trace1 = go.Scatter(x=y_true, y=y_pred,mode = 'markers', showlegend=False,
                                marker=dict(
                                    size=8,
                                    color = np.abs(y_true-y_pred), #set color equal to a variable
                                    colorscale='Viridis',
                                    showscale=True))
        trace2 = go.Scatter(x=[300, 1500], y=[300,1500],
                            line = dict(color = ('rgb(0, 0, 0)'),
                                        width = 4, dash = 'dot'),
                            showlegend=False)
        trace3 = go.Scatter(x=[300, 1500], y=[400,1600],
                            line = dict(color = ('rgb(22, 96, 167)'),
                                        width = 4, dash = 'dot'),
                            showlegend=False)
        trace4 = go.Scatter(x=[300, 1500], y=[200,1400],
                            line = dict(color = ('rgb(22, 96, 167)'),
                                        width = 4, dash = 'dot'),
                            showlegend=False)

        data = [trace1, trace2, trace3, trace4]
        layout = go.Layout(title="Difference between predicted and truth by range",
                           yaxis=dict(title="TG predicted"), xaxis=dict(title="TG True"))
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename='lineplot-{0}.html'.format(name), auto_open=True)


def boxplot_local_evaluation(metric="RMSE",
    paths=["../result/result_oracle/default-model/mode_test_.list", "../result/baselines/log/predictions_raw_RF.list"]):

    dic_measure = {"MAE":0, "MSE":1, "R2_S":2, "RRMSE":3, "RMSE":4, "MARE":5, "R2":6}

    data = []
    data2 = []
    data3 = []
    for path in paths:
        print(path)
        if str.find(path, 'baselines') >= 0:
            _, y_true, y_pred = pickle.load(open(path, 'rb'))
        else:
            y_true, y_pred = pickle.load(open(path, 'rb'))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        max_value = np.max(y_true)
        min_value = np.min(y_true)
        steps = [[min_value+(50*i),min_value+(50*(i+1))] for i in range(0,int((max_value-min_value)/50)-4)]
        steps[len(steps)-1][1] = max_value+1

        x = []
        y = []
        x2 = []
        y2 = []
        sd = []
        mare = []
        mare_sd = []

        for step in steps:
            aux_true = y_true[np.all([step[0]<=y_true, y_true<step[1]], axis=0)]
            aux_pred = y_pred[np.all([step[0]<=y_true, y_true<step[1]], axis=0)]
            diff = aux_pred - aux_true
            y = y+diff.tolist()
            x = x+(["{0} - {1}".format(step[0],step[1])]*diff.shape[0])
            x2 = x2+["{0} - {1}".format(step[0],step[1])]
            aux_mare = (np.abs(aux_true - aux_pred) / aux_pred)*100
            mare = mare + [np.mean(aux_mare)]
            mare_sd = mare_sd + [np.std(aux_mare)]
            y2 = y2+[np.mean(np.abs(diff))]
            sd = sd+[np.std(np.abs(diff))]

        data.append([x,y])
        data2.append([x2, y2, sd])
        data3.append([x2, mare, mare_sd])

    trace0 = go.Box(x=data[0][0], y=data[0][1], name='mode', marker=dict(color='#3D9970'), boxmean=True)
    trace1 = go.Box(x=data[1][0], y=data[1][1], name='baseline-RF', marker=dict(color='#FF851B'), boxmean=True)
    data = [trace0, trace1]
    layout = go.Layout(title="Difference between predicted and truth by range",
        yaxis=dict(title='Difference (predict-truth)', zeroline=False),
        boxmode='group')
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='boxplot.html', auto_open=True)


    trace0 = go.Bar(x=data2[0][0], y=data2[0][1], name='mode',
                    error_y=dict( type='data', array=data2[0][2], visible=True))
    trace1 = go.Bar(x=data2[1][0], y=data2[1][1], name='baseline-RF',
                    error_y=dict( type='data', array=data2[1][2], visible=True))
    data = [trace0, trace1]
    layout = go.Layout(title="Absolute difference between predicted and truth by range",
        barmode='group', yaxis=dict(title="Abs Diff"))
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename = 'barplot-mare.html', auto_open=True)

    trace0 = go.Bar(x=data3[0][0], y=data3[0][1], name='mode',
                    error_y=dict( type='data', array=data3[0][2], visible=True))
    trace1 = go.Bar(x=data3[1][0], y=data3[1][1], name='baseline-RF',
                    error_y=dict( type='data', array=data3[1][2], visible=True))
    data = [trace0, trace1]
    layout = go.Layout(title="MARE measure by range",
        barmode='group', yaxis=dict(title="MARE"))
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename = 'barplot-diff.html', auto_open=True)


def local_evaluation(metric="RMSE"):
    paths = [
        "../result/baselines/log/predictions_raw_RF.list", #baseline
        "../result/baselines/log/predictions_raw_DT.list", #baseline
        "../result/baselines/log/predictions_raw_MLP.list", #baseline
        "../result/baselines/log/predictions_pca20_RF.list", #baseline
        "../result/baselines/log/predictions_pca20_DT.list", #baseline
        "../result/baselines/log/predictions_pca20_MLP.list", #baseline
        "../result/result_oracle/default-model/mean_test_.list", #trmb
        "../result/result_oracle/default-model/mode_test_.list", #trmb
        "../result/result_oracle/default-model/mean_test_all_leaf.list", #oracle
        "../result/result_oracle/default-model/mode_test_all_leaf.list", #oracle
    ]

    dic_measure = {"MAE":0, "MSE":1, "R2_S":2, "RRMSE":3, "RMSE":4, "MARE":5, "R2":6}

    data = []
    tags = ["baseline-RF", "baseline-DT", "baseline-MLP",
            "baseline-pca-RF", "baseline-pca-DT", "baseline-pca-MLP",
            "mean", "mode",
            "mean-oracle", "mode-oracle"]
    steps = []
    for path,tag in zip(paths, tags):
        print(path)
        if str.find(path, 'baselines') >= 0:
            _, y_true, y_pred = pickle.load(open(path, 'rb'))
        else:
            y_true, y_pred = pickle.load(open(path, 'rb'))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        max_value = np.max(y_true)
        min_value = np.min(y_true)
        steps = [[min_value+(50*i),min_value+(50*(i+1))] for i in range(0,int((max_value-min_value)/50)-4)]
        steps[len(steps)-1][1] = max_value+1
        value = [tag]
        for step in steps:
            aux_true = y_true[np.all([step[0]<=y_true, y_true<step[1]], axis=0)]
            aux_pred = y_pred[np.all([step[0]<=y_true, y_true<step[1]], axis=0)]
            perf = compute_performance(aux_true, aux_pred)
            value.append(perf[dic_measure[metric]])
        data.append(value)
    y_true, _ = pickle.load(open(path, 'rb'))
    value = ["sample-amount"]
    for step in steps:
        aux_true = y_true[np.all([step[0]<=y_true, y_true<step[1]], axis=0)]
        value.append(aux_true.shape[0])
    data.append(value)

    directory = "../result/performance/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    steps = ["{0} - {1}".format(i,j) for i,j in steps]
    columns = ["Tag"] + steps
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("../result/performance/local_test_perf_{0}.csv".format(metric), index=False)


def rem_baseline_evaluation():

    paths = ["../result/result_oracle/default-model/mean_test_rem_.list",
             "../result/result_oracle/default-model/mode_test_rem_.list",
             '../result/baselines/log/predictions_ext_raw_RF.list',
             '../result/baselines/log/predictions_ext_raw_MLP.list',
             ]

    data1 = {}
    data2 = {}
    for path, name in zip(paths,["mode", "mean","baseline-RF", "baseline-MLP"]):
        sample = ["s"+str(i) for i in range(1,13)]
        if str.find(path, 'baselines') >= 0:
            _, y_true, y_pred = pickle.load(open(path, 'rb'))
        else:
            y_true, y_pred = pickle.load(open(path, 'rb'))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        perc = (np.abs(y_true - y_pred) / y_true) * 100
        data1[name] = y_pred
        data2[name] = perc

    df = pd.DataFrame(data1)
    df.to_csv("../result/performance/{0}_test_rem_perf.csv".format("TG"), index=False)
    df = pd.DataFrame(data2)
    df.to_csv("../result/performance/{0}_test_rem_perf.csv".format("MARE"), index=False)


def rem_evaluation():

    paths = [
        ["../result/result_oracle/default-model/mean_test_rem_.list",
        "../result/result_oracle/default-model/mean_test_rem_root_.list"],
        ["../result/result_oracle/default-model/mode_test_rem_.list",
        "../result/result_oracle/default-model/mode_test_rem_root_.list"]
    ]
    for path, save in zip(paths,["mean","mode"]):
        sample = ["s"+str(i) for i in range(1,13)]
        y_true, y_pred = pickle.load(open(path[0], 'rb'))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        perc = np.abs(y_true - y_pred) / y_true

        y_true_cls, y_pred_cls = pickle.load(open(path[1], 'rb'))
        classif_hit = np.array(y_true_cls) == np.array(y_pred_cls)
        print(y_true_cls)
        print(y_pred_cls)

        data = {"Sample":sample, "y_true":y_true, "y_pred":y_pred, "Precentage":perc, "Classifier hit":classif_hit}
        df = pd.DataFrame(data)
        df.to_csv("../result/performance/{0}_test_rem_perf.csv".format(save), index=False)


def internal_classifier_evaluation():
    paths = [
        "../result/result_oracle/default-model/mean_test_root_.list",
        "../result/result_oracle/default-model/mode_test_root_.list"
    ]

    data = []
    tags = ["mean", "mode"]
    for path, tag in zip(paths, tags):
        print(path)

        y_true, y_pred = pickle.load(open(path, 'rb'))
        perf = [accuracy_score(y_true, y_pred),
                f1_score(y_true, y_pred, average="macro")]
        print("conf matrix")
        print('Confusion matrix, without normalization')
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print("Normalized confusion matrix")
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)

        data.append([tag]+perf)
    print()
    directory = "../result/performance/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    columns = ["Tag", "ACC", "F1"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("../result/performance/root_test_perf.csv", index=False)


def internal_regressors_evaluation():
    paths = [
        "../result/result_oracle/default-model/mean_test_leaf-start_.list",
        "../result/result_oracle/default-model/mean_test_leaf-middle_.list",
        "../result/result_oracle/default-model/mean_test_leaf-end_.list",
        "../result/result_oracle/default-model/mode_test_leaf-start_.list",
        "../result/result_oracle/default-model/mode_test_leaf-middle_.list",
        "../result/result_oracle/default-model/mode_test_leaf-end_.list",
    ]

    dic_measure = {"MAE":0, "MSE":1, "R2_S":2, "RRMSE":3, "RMSE":4, "MARE":5, "R2":6}
    data = []
    tags = ["mean-start", "mean-middle", "mean-end",
            "mode-start", "mode-middle", "mode-end"]
    for path, tag in zip(paths, tags):
        print(path)

        y_true, y_pred = pickle.load(open(path, 'rb'))
        perf = compute_performance(y_true, y_pred)
        line =[]
        line.append(tag)

        for measure in  dic_measure.keys():
            print("------ "+measure+" ------")
            print(perf[dic_measure[measure]])
            line.append(perf[dic_measure[measure]])
        print("-------------------------\n")
        data.append(line)

    directory = "../result/performance/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    columns = ["Tag"] + list(dic_measure.keys())
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("../result/performance/leaf_test_perf.csv", index=False)


def global_evaluation():

    paths = [
        "../result/baselines/log/predictions_raw_RF.list", #baseline
        "../result/baselines/log/predictions_raw_DT.list", #baseline
        "../result/baselines/log/predictions_raw_MLP.list", #baseline
        "../result/baselines/log/predictions_pca20_RF.list", #baseline
        "../result/baselines/log/predictions_pca20_DT.list", #baseline
        "../result/baselines/log/predictions_pca20_MLP.list", #baseline
        "../result/result_oracle/default-model/mean_test_.list", #trmb
        "../result/result_oracle/default-model/mode_test_.list", #trmb
        "../result/result_oracle/default-model/mean_test_all_leaf.list", #oracle
        "../result/result_oracle/default-model/mode_test_all_leaf.list", #oracle
    ]

    dic_measure = {"MAE":0, "MSE":1, "R2_S":2, "RRMSE":3, "RMSE":4, "MARE":5, "R2":6}

    data = []
    tags = ["baseline-RF", "baseline-DT", "baseline-MLP",
            "baseline-pca-RF", "baseline-pca-DT", "baseline-pca-MLP",
            "mean", "mode",
            "mean-oracle", "mode-oracle"]
    for path,tag in zip(paths, tags):
        print(path)
        if str.find(path, 'baselines') >= 0:
            _, y_true, y_pred = pickle.load(open(path, 'rb'))
        else:
            y_true, y_pred = pickle.load(open(path, 'rb'))
        perf = compute_performance(y_true, y_pred)

        line = []
        line.append(tag)
        for measure in  dic_measure.keys():
            print("------ "+measure+" ------")
            print(perf[dic_measure[measure]])
            line.append(perf[dic_measure[measure]])
        print("-------------------------\n")
        data.append(line)

    directory = "../result/performance/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    columns = ["Tag"] + list(dic_measure.keys())
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("../result/performance/global_test_perf.csv", index=False)


global_evaluation()
internal_regressors_evaluation()
internal_classifier_evaluation()
rem_evaluation()
local_evaluation(metric="RMSE")
local_evaluation(metric="MARE")
boxplot_local_evaluation()
lineplot_local_evaluation()
rem_baseline_evaluation()
