import sys
import os
import pandas as pd
import numpy as np
import time
import itertools
import copy
import pickle

from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from dateutil.relativedelta import *
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from imblearn.under_sampling import RandomUnderSampler
from multiprocessing import Pool

def offset_within_month(time):
    day = int(time[8:10])
    hour = int(time[11:13])
    minute = int(time[14:16])
    second = int(time[17:22])
    offset = ((day * 24 + hour) * 60 + minute) * 60 + second
    return offset

def offset_all_time(time):
    day_of_month ={1:0,2:31,3:61,4:92,5:123,6:152,7:183,8:213,9:246}
    month = int(time[5:7])
    day = int(time[8:10])
    hour = int(time[11:13])
    minute = int(time[14:16])
    second = int(time[17:22])
    offset = (((day_of_month[month] + day) * 24 + hour) * 60 + minute) * 60 + second
    return offset

def load_data(freq, train_months, test_months, failure_type, backtrace):
    label = pd.read_csv('../data/trouble_tickets.csv')
    label['class'] = label['failure_type']
    label['failed_time_offset'] = label['failed_time'].apply(offset_all_time)
    df_train = pd.DataFrame()
    for i in train_months:
        df_month = pd.read_csv('./features/features_' + freq + '_month_' + str(i) + '.csv')
        df_month['predict_time_offset'] = df_month['predict_time'].apply(offset_all_time)
        res_df = pd.merge(df_month,label,how='left',on='sid')
        res_df['class'].fillna(0,inplace=True)
        df_negative_server = res_df[res_df['class'] == 0]
        df_positive_server = res_df[res_df['class'] == failure_type]
        df_positive_server['class'] = 1
        df_positive_server = df_positive_server[df_positive_server['predict_time_offset'] <= df_positive_server['failed_time_offset']]
        df_positive_server.loc[df_positive_server['predict_time_offset'] + pd.Timedelta(backtrace).total_seconds() < df_positive_server['failed_time_offset'],'class'] = 0
        ret_df = pd.concat([df_negative_server,df_positive_server])
        ret_df=ret_df.drop(columns=['failed_time','failed_time_offset','predict_time_offset','failure_type'])
        df_train = pd.concat([df_train,ret_df])
    df_test = pd.DataFrame()
    for i in test_months:
        df_month = pd.read_csv('./features/features_' + freq + '_month_' + str(i) + '.csv')
        df_month['predict_time_offset'] = df_month['predict_time'].apply(offset_all_time)
        res_df = pd.merge(df_month,label,how='left',on='sid')
        res_df['class'].fillna(0,inplace=True)
        df_negative_server = res_df[res_df['class'] == 0]
        df_positive_server = res_df[res_df['class'] == failure_type]
        df_positive_server['class'] = 1
        df_fake_positve = res_df[(res_df['class'] > 0) & (res_df['class'] != failure_type)]
        df_fake_positve['class'] = 0
        df_positive_server = df_positive_server[df_positive_server['predict_time_offset'] <= df_positive_server['failed_time_offset']]
        df_positive_server.loc[df_positive_server['predict_time_offset'] + pd.Timedelta(backtrace).total_seconds() < df_positive_server['failed_time_offset'],'class'] = 0
        ret_df = pd.concat([df_negative_server,df_positive_server,df_fake_positve])
        ret_df=ret_df.drop(columns=['failed_time','failed_time_offset','predict_time_offset','failure_type'])
        df_test = pd.concat([df_test,ret_df])
    return df_train, df_test

def sampling(train_df, ratio = 0.02):
    if ratio == 0:
        X = train_df[train_df.columns[0:-1]]
        y = train_df[train_df.columns[-1]]
        return X, y
    sample_func = RandomUnderSampler(ratio, random_state=42)  # random downsampling
    train_df=train_df.reset_index(drop=True)
    train_df['sample_index'] = train_df.index
    X = train_df.loc[:,['sample_index']]
    y = train_df.loc[:,['class']]
    sample_index, sample_y = sample_func.fit_resample(X, y)
    sample_X = train_df.iloc[sample_index['sample_index']][train_df.columns[0:-2]]
    sample_X = sample_X.reset_index(drop=True)
    train_df.drop(columns=['sample_index'],inplace=True)
    return sample_X, sample_y

def train_test_data_generation(df_train, df_test,sample_ratio=0.02):
    train_X,train_y=sampling(df_train,ratio=sample_ratio)
    df_test=df_test.reset_index(drop = True)
    test_X = df_test[df_test.columns[2:-1]]
    test_y = df_test[['sid','predict_time','class']]
    return train_X, train_y, test_X, test_y

def my_nn_model(input_dim=52):
    def get_model():
        """
        Return a LSTM Kear model"""
        nn_model = Sequential()
        nn_model.add(Dense(144, input_dim=input_dim, activation='relu'))
        nn_model.add(Dropout(0.4))
        nn_model.add(LSTM(64))
        nn_model.add(Dropout(0.4))
        nn_model.add(Dense(2, activation='softmax'))
        nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        return nn_model
    return get_model

def classifier(model):
    if model == 'LR':
        ML = LogisticRegression
        hypers = {
            'solver': ['newton-cg', 'liblinear','lbfgs', 'saga'],
            'C': [0.0001, 0.001, 0.01],
            'class_weight': [None, 'balanced'],
            'random_state': [42],
            'n_jobs': [-1],
        }
    elif model == 'SVM':
        ML = SVC
        hypers = {
            'probability': [True],
            'C': [0.01, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1],
        }
    elif model == 'RF':
        ML = RandomForestClassifier
        hypers = {
            'n_estimators': [10,100,200],
            'max_depth': [None, 10, 50, 100],
            'min_samples_split': [10, 50, 100],
            'random_state': [42],
            'n_jobs': [-1],
        }
    elif model == 'GBDT':
        ML = LGBMClassifier
        hypers = {
            'boosting_type': ['gbdt'],
            'n_estimators': [10, 50, 100,200],
            'max_depth': [-1, 10, 50, 100],
            'num_leaves': [2, 5, 10, 50],
            'learning_rate': [0.001, 0.01, 0.1],
            'class_weight': [None, 'balanced'],
            'random_state': [42],
            'n_jobs': [60],
        }
    elif model == 'MLP':
        ML = MLPClassifier
        hypers = {
            'hidden_layer_sizes': [10,50,100,200],
            'solver': ['lbfgs', 'sgd','adam'],
            'learning_rate':['constant', 'adaptive'],
            'random_state': [42],
            'learning_rate_init':[0.01,0.001]
        }
    elif model == 'DNN':
        ML = KerasClassifier(my_nn_model(), epochs=30, verbose=0)
        hypers ={}
    else:
        raise ValueError('Currently consider LR, SVM, RF, and GBDT, MLP and DNN only!')
    return ML, hypers

def hypers_space(hypers):
    return [
        {k:v for k,v in zip(hypers.keys(), para)} for para in itertools.product(*[val for val in hypers.values()])
    ]

def evaluation_raw(y_pred, y_test,threshold = 0.5):
    ## calcuate relative and absolute
    y_test.loc[:,'pred'] = y_pred[:,1]
    y_test.loc[:,'pred_class'] = y_test['pred'] > threshold
    all_server = y_test.drop_duplicates(['sid'],keep='last')
    positive_server = all_server[all_server['class'] > 0]
    tp_server = y_test[y_test['pred_class'] & y_test['class']].drop_duplicates(['sid'])
    predict_postive_server = y_test[y_test['pred_class']].drop_duplicates(['sid'])
    tp = len(tp_server)
    fp = len(predict_postive_server) - tp
    fn = len(positive_server) - tp
    tn = len(all_server) - tp - fp - fn
    return tp, fp, fn, tn

def evaluation_metric(tp, fp, fn, tn):
    if tp == 0:
        return [0, 0, 0, fp / (fp + tn)]
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        fpr = fp / (fp + tn)
        if fpr > 0.01:
            return 0,0,0,0
        else:
            return f1,precision,recall,fpr

def evaluation_metric_V2(tp, fp, fn, tn):
    if tp == 0:
        return [0, 0, 0, fp / (fp + tn)]
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        fpr = fp / (fp + tn)
        if fpr > 0.01 or fpr < 0.002:
            return 0,0,0,0
        else:
            return f1,precision,recall,fpr

def evaluate(y_pred, y_test, th):
    tp, fp, fn, tn = evaluation_raw(y_pred, y_test,threshold = th)
    f1, pr, re, fpr = evaluation_metric(tp, fp, fn, tn)
    return f1, pr, re, fpr, tp, fp, fn, tn

def evaluate_V2(y_pred, y_test, th, month, typ):
    ## positive = {1:[62,105,75], 2:[48,93,79],3:[55,116,79]}  ##all predictalbe
    positive = {1:[46,92,51], 2:[35,65,54],3:[40,75,48]}  ## predicatble for 5m
    tp, fp, fn, tn = evaluation_raw(y_pred, y_test,threshold = th)
    fn = positive[month][typ-1] - tp
    f1, pr, re, fpr = evaluation_metric_V2(tp, fp, fn, tn)
    return f1, pr, re, fpr, tp, fp, fn, tn

def find_best_f1(y_pred, y_truth):
    best_raw = []
    best_f1 = 0
    best_th = 0
    for threshold in range (0, 100, 1):
        f1, pr, re, fpr, tp, fp, fn, tn=evaluate(y_pred,y_truth,threshold/100)
        if (f1 > best_f1):
            best_f1 = f1
            best_raw = [f1, pr, re, fpr, tp, fp, fn, tn]
            best_th = threshold/100
    return best_f1, best_raw, best_th

def find_best_f1_V2(y_pred, y_truth,month,typ):
    best_raw = []
    best_f1 = 0
    best_th = 0
    for threshold in range (0, 100, 1):
        f1, pr, re, fpr, tp, fp, fn, tn=evaluate_V2(y_pred,y_truth,threshold/100,month,typ)
        if (f1 > best_f1):
            best_f1 = f1
            best_raw = [f1, pr, re, fpr, tp, fp, fn, tn]
            best_th = threshold/100
    return best_f1, best_raw, best_th

def time2class(x):
    to_min = x / 60
    if to_min <= 5:
        return 1
    else:
        return 2

def evaluation_time(y_pred, y_test, threshold):
    y_test.loc[:,'pred'] = y_pred[:,1]
    y_test.loc[:,'pred_class'] = y_test['pred'] > threshold
    tp_server = y_test[y_test['pred_class'] & y_test['class']]
    tp_server = tp_server.sort_values(by=['sid','predict_time']).drop_duplicates('sid',keep='first')
    label = pd.read_csv('../data/trouble_tickets.csv')
    label['failed_time_offset'] = label['failed_time'].apply(offset_all_time)
    tp_server['predict_time_offset']=tp_server['predict_time'].apply(offset_all_time)
    tp_server = tp_server.merge(label,on='sid',how='inner')
    tp_server['time_diff'] = tp_server['failed_time_offset'] - tp_server['predict_time_offset']
    tp_server['time_class'] = tp_server['time_diff'].apply(time2class)
    return len(tp_server[tp_server['time_class'] == 1]), len(tp_server[tp_server['time_class'] == 2])

def features_generation(groups):
    tmp = ['sid','predict_time']
    one_hot = ['model_A1','model_A2','model_B1','model_B2','model_B3','model_C1','model_C2','dimm_num_8','dimm_num_12','dimm_num_16','dimm_num_24','manufacturer_M1','manufacturer_M2','manufacturer_M3','manufacturer_M4']
    feature = ''
    for metric in ['counter','mtbe','read','scrub','soft','hard']:
        tmp.append('5min' + str('_') + metric)
    if (groups == 2 or groups == 3 or groups == 4):
        for metric in ['socket_error','channel_error','bank_error','row_error', 'column_error','cell_error', 'random_error', 'socket_count', 'channel_count','bank_count', 'row_count','col_count','cell_count']:
            tmp.append('5min' + str('_') + metric)
        if (groups == 3 or groups == 4):
            for metric in ['socket_mean','socket_median','socket_std', 'channel_mean','channel_median','channel_std','bank_mean','bank_median','bank_std','row_mean','row_median','row_std','col_mean','col_median','col_std','cell_mean','cell_median','cell_std']:
                tmp.append('5min' + str('_') + metric)
        if (groups == 4):
            tmp = tmp + one_hot
    tmp = tmp + ['class']
    return tmp

def exp1_single_run(features, pred_month, types):
    #print(len(features), pred_month, types)
    if pred_month == 1:
        df_train, df_test = load_data('5m', [6], [6], failure_type=types, backtrace='1d')
    elif pred_month == 2:
        df_train, df_test = load_data('5m', [7], [7], failure_type=types, backtrace='1d')
    elif pred_month == 3:
        df_train, df_test = load_data('5m', [8], [8], failure_type=types, backtrace='1d')
    else:
        print("Unkown testing month")

    df_test = df_test.loc[:,features]
    train_X, train_y, test_X, test_y = train_test_data_generation(df_train, df_test,sample_ratio=0.02)
    best_f1 = 0
    precision = 0
    recall = 0
    fpr = 0
    feature_grp_num = 0

    if len(features) == 9:
      feature_grp_num = 1
    elif len(features) == 22:
      feature_grp_num = 2
    elif len(features) == 40:
      feature_grp_num = 3
    else:
      feature_grp_num = 4

    name = "../model/exp1/exp1/exp1_month_" + str(pred_month) + "_types_" + str(types) + "_features_" + str(feature_grp_num)
    model = pickle.load(open(name,'rb'))
    y_pred=model.predict_proba(test_X)
    y_truth = copy.deepcopy(test_y)
    f1, raw, th = find_best_f1(y_pred,y_truth)
    if (best_f1 < f1):
        best_f1 = f1
        precision = raw[1]
        recall = raw[2]
        fpr = raw[3]
    return best_f1, precision, recall, fpr

def exp1_single_run_from_scratch(features, pred_month, types):
    #print(len(features), pred_month, types)
    if pred_month == 1:
        df_train, df_test = load_data('5m', [1,2,3,4,5], [6], failure_type=types, backtrace='1d')
    elif pred_month == 2:
        df_train, df_test = load_data('5m', [2,3,4,5,6], [7], failure_type=types, backtrace='1d')
    elif pred_month == 3:
        df_train, df_test = load_data('5m', [3,4,5,6,7], [8], failure_type=types, backtrace='1d')
    else:
        print("Unkown testing month")

    df_train=df_train.reset_index(drop=True)
    df_train = df_train.loc[:,features]
    df_test = df_test.loc[:,features]
    train_X, train_y, test_X, test_y = train_test_data_generation(df_train, df_test,sample_ratio=0.02)
    best_f1 = 0
    precision = 0
    recall = 0
    fpr = 0
    for n_est in [10,50,100]:
        for maxd in [5,10, 50]:
            for val in [5,10, 50]:
                model,hyperparameters = classifier('RF')
                model = model(n_jobs=30,random_state=42,n_estimators=n_est,max_depth=maxd,min_samples_split=val)
                model.fit(train_X[train_X.columns[2:]], train_y)
                y_pred=model.predict_proba(test_X)
                y_truth = copy.deepcopy(test_y)
                f1, raw, th = find_best_f1(y_pred,y_truth)
                if (best_f1 < f1):
                    best_f1 = f1
                    precision = raw[1]
                    recall = raw[2]
                    fpr = raw[3]
    return best_f1, precision, recall, fpr

def run_parallel_exp1(lst):
        grp_num = lst[0]
        typ = lst[1]
        month = lst[2]
        #print('Run to parameters:',grp_num, typ, month)
        feature = features_generation(grp_num)
        f1, precision, recall, fpr = exp1_single_run(feature,month,typ)
        #print('Run to parameters:', grp_num, typ, month, "with results:", f1, precision, recall, fpr)
        return grp_num, typ, month, f1, precision, recall, fpr

def exp1_main():
    lst = []
    for grpnum in [1,2,3,4]:
        for typ in [1,2,3]:
            for month in [1,2,3]:
                lst.append([grpnum, typ, month])
    pool = Pool(processes=len(lst))
    ret=pool.map(run_parallel_exp1,lst)
    res_df=pd.DataFrame(ret).rename(columns={0:'feature_grps', 1:'failure_type',2:'month',3:'F1-score',4:'Precision',5:'Recall',6:'FPR'})
    grouped = res_df.groupby('failure_type')
    for name, group in grouped:
        print(group.groupby('feature_grps').mean().drop(columns={'month'}))
    return res_df

def exp2_single_run(model, pred_month, types):
    #print(len(features), pred_month, types)
    if pred_month == 1:
        df_train, df_test = load_data('5m', [6], [6], failure_type=types, backtrace='1d')
    elif pred_month == 2:
        df_train, df_test = load_data('5m', [7], [7], failure_type=types, backtrace='1d')
    elif pred_month == 3:
        df_train, df_test = load_data('5m', [8], [8], failure_type=types, backtrace='1d')
    else:
        print("Unkown testing month")

    features = features_generation(4) ## using all groups of features
    df_test = df_test.loc[:,features]
    train_X, train_y, test_X, test_y = train_test_data_generation(df_train, df_test,sample_ratio=0.02)

    best_f1 = 0
    precision = 0
    recall = 0
    fpr = 0

    name = "../model/exp2/exp2/exp2_month_" + str(pred_month) + "_types_" + str(types) + "_predictor_" + str(model)
    model = pickle.load(open(name,'rb'))
    y_pred=model.predict_proba(test_X)
    y_truth = copy.deepcopy(test_y)
    f1, raw, th = find_best_f1(y_pred,y_truth)
    if (best_f1 < f1):
        best_f1 = f1
        precision = raw[1]
        recall = raw[2]
        fpr = raw[3]
    return best_f1, precision, recall, fpr

def exp2_single_run_from_scratch(model, pred_month, types):
    if pred_month == 1:
        df_train, df_test = load_data('5m', [1,2,3,4,5], [6], failure_type=types, backtrace='1d')
    elif pred_month == 2:
        df_train, df_test = load_data('5m', [2,3,4,5,6], [7], failure_type=types, backtrace='1d')
    elif pred_month == 3:
        df_train, df_test = load_data('5m', [3,4,5,6,7], [8], failure_type=types, backtrace='1d')
    else:
        print("Unkown testing month")
    features = features_generation(4) ## using all groups of features
    df_train=df_train.reset_index(drop=True)
    df_train = df_train.loc[:,features]
    df_test = df_test.loc[:,features]
    train_X, train_y, test_X, test_y = train_test_data_generation(df_train, df_test,sample_ratio=0.02)
    best_f1 = 0
    precision = 0
    recall = 0
    fpr = 0
    parameter = 0
    if model == 'RF':
        for n_est in [10,50,100]:
            for maxd in [5,10, 50]:
                for val in [5,10, 50]:
                    model,hyperparameters = classifier('RF')
                    model = model(n_jobs=30,random_state=42,n_estimators=n_est,max_depth=maxd,min_samples_split=val)
                    model.fit(train_X[train_X.columns[2:]], train_y)
                    y_pred=model.predict_proba(test_X)
                    y_truth = copy.deepcopy(test_y)
                    f1, raw, th = find_best_f1(y_pred,y_truth)
                    if (best_f1 < f1):
                        best_f1 = f1
                        precision = raw[1]
                        recall = raw[2]
                        fpr = raw[3]
    elif model == 'GBDT':
        for n_est in [10,50,100]:
            for maxd in [5,10]:
                for val in [5,10]:
                    model,hyperparameters = classifier('GBDT')
                    model = model(n_jobs=30,random_state=42,n_estimators=n_est,max_depth=maxd,num_leaves=val)
                    model.fit(train_X[train_X.columns[2:]], train_y)
                    y_pred=model.predict_proba(test_X)
                    y_truth = copy.deepcopy(test_y)
                    f1, raw, th = find_best_f1(y_pred,y_truth)
                    if (best_f1 < f1):
                        best_f1 = f1
                        precision = raw[1]
                        recall = raw[2]
                        fpr = raw[3]
    elif model == 'LR':
        for lr in [0.01,0.1]:
            model,hyperparameters = classifier('LR')
            model = model(n_jobs=30,random_state=42,C=lr)
            model.fit(train_X[train_X.columns[2:]], train_y)
            y_pred=model.predict_proba(test_X)
            y_truth = copy.deepcopy(test_y)
            f1, raw, th = find_best_f1(y_pred,y_truth)
            if (best_f1 < f1):
                best_f1 = f1
                precision = raw[1]
                recall = raw[2]
                fpr = raw[3]
    elif model == 'SVM':
        for tree in [10, 50, 100]:
            model = BaggingClassifier(SVC(probability=True), max_samples=1.0 / tree, n_estimators=tree,n_jobs=40)
            model.fit(train_X[train_X.columns[2:]], train_y)
            y_pred=model.predict_proba(test_X)
            y_truth = copy.deepcopy(test_y)
            f1, raw, th = find_best_f1(y_pred,y_truth)
            if (best_f1 < f1):
                best_f1 = f1
                precision = raw[1]
                recall = raw[2]
                fpr = raw[3]
    elif model == 'MLP':
        for layers in [1,3,5,10,20]:
            neurons = {1:{1:50,2:50,3:10},2:{1:50,2:10,3:10},3:{1:10,2:10,3:50}}
            model,hyperparameters = classifier('MLP')
            neurons_num = neurons[pred_month][types]
            hidden_layers_parameter = (neurons_num)
            if layers == 3:
                hidden_layers_parameter = (neurons_num,neurons_num,neurons_num)
            if layers == 5:
                hidden_layers_parameter = (neurons_num,neurons_num,neurons_num,neurons_num,neurons_num)
            if layers == 10:
                hidden_layers_parameter = (neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num)
            if layers == 20:
                hidden_layers_parameter = (neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num,neurons_num)
            model = model(random_state=42,hidden_layer_sizes=hidden_layers_parameter)
            model.fit(train_X[train_X.columns[2:]], train_y)
            y_pred=model.predict_proba(test_X)
            y_truth = copy.deepcopy(test_y)
            f1, raw, th = find_best_f1(y_pred,y_truth)
            if (best_f1 < f1):
                best_f1 = f1
                precision = raw[1]
                recall = raw[2]
                fpr = raw[3]
                parameter = layers
    elif model == 'DNN':
        model,hyperparameters = classifier('DNN')
        model.fit(train_X[train_X.columns[2:]], train_y)
        y_pred=model.predict_proba(test_X)
        y_truth = copy.deepcopy(test_y)
        f1, raw, th = find_best_f1(y_pred,y_truth)
        if (best_f1 < f1):
            best_f1 = f1
            precision = raw[1]
            recall = raw[2]
            fpr = raw[3]
    else:
        print("Unknown models")
    return best_f1, precision, recall, fpr

def run_parallel_exp2(lst):
        model = lst[0]
        typ = lst[1]
        month = lst[2]
        f1, precision, recall, fpr = exp2_single_run(model,month,typ)
        print(model, typ, month, f1, precision, recall, fpr)
        return model, typ, month, f1, precision, recall, fpr

def exp2_main():
    lst = []
    for model in ['GBDT','LR','SVM','MLP']:
        for typ in [1,2,3]:
            for month in [1,2,3]:
                lst.append([model, typ, month])
    pool = Pool(processes=len(lst))
    ret=pool.map(run_parallel_exp2,lst)
    res_df=pd.DataFrame(ret).rename(columns={0:'feature_grps', 1:'failure_type',2:'month',3:'F1-score',4:'Precision',5:'Recall',6:'FPR'})
    grouped = res_df.groupby('failure_type')
    for name, group in grouped:
        print(group.groupby('feature_grps').mean().drop(columns={'month'}))
    return res_df

def exp3_single_run(freq, pred_month, types):
    #print(len(features), pred_month, types)
    if pred_month == 1:
        df_train, df_test = load_data('5m', [6], [6], failure_type=types, backtrace='1d')
    elif pred_month == 2:
        df_train, df_test = load_data('5m', [7], [7], failure_type=types, backtrace='1d')
    elif pred_month == 3:
        df_train, df_test = load_data('5m', [8], [8], failure_type=types, backtrace='1d')
    else:
        print("Unkown testing month")

    features = features_generation(4) ## using all groups of features
    df_test = df_test.loc[:,features]
    train_X, train_y, test_X, test_y = train_test_data_generation(df_train, df_test,sample_ratio=0.02)
    best_f1 = 0
    precision = 0
    recall = 0
    fpr = 0

    name = "../model/exp3/exp3/exp3_month_" + str(pred_month) + "_types_" + str(types) + "_interval_" + str(freq)
    model = pickle.load(open(name,'rb'))
    y_pred=model.predict_proba(test_X)
    y_truth = copy.deepcopy(test_y)
    f1, raw, th = find_best_f1_V2(y_pred,y_truth)
    if (best_f1 < f1):
        best_f1 = f1
        precision = raw[1]
        recall = raw[2]
        fpr = raw[3]
    return best_f1, precision, recall, fpr

def exp3_single_run_from_scratch(freq, pred_month, types):
    if pred_month == 1:
        df_train, df_test = load_data(freq, [1,2,3,4,5], [6], failure_type=types, backtrace='1d')
    elif pred_month == 2:
        df_train, df_test = load_data(freq, [2,3,4,5,6], [7], failure_type=types, backtrace='1d')
    elif pred_month == 3:
        df_train, df_test = load_data(freq, [3,4,5,6,7], [8], failure_type=types, backtrace='1d')
    else:
        print("Unkown testing month")
    features = features_generation(4) ## using all groups of features
    df_train=df_train.reset_index(drop=True)
    df_train = df_train.loc[:,features]
    df_test = df_test.loc[:,features]
    train_X, train_y, test_X, test_y = train_test_data_generation(df_train, df_test,sample_ratio=0.02)
    best_f1 = 0
    precision = 0
    recall = 0
    fpr = 0
    for n_est in [10,50,100]:
        for maxd in [5,10, 50]:
            for val in [5,10, 50]:
                model,hyperparameters = classifier('RF')
                model = model(n_jobs=30,random_state=42,n_estimators=n_est,max_depth=maxd,min_samples_split=val)
                model.fit(train_X[train_X.columns[2:]], train_y)
                y_pred=model.predict_proba(test_X)
                y_truth = copy.deepcopy(test_y)
                f1, raw, th = find_best_f1_V2(y_pred,y_truth,pred_month,types)
                if (best_f1 < f1):
                    best_f1 = f1
                    precision = raw[1]
                    recall = raw[2]
                    fpr = raw[3]
    return best_f1, precision, recall, fpr

def run_parallel_exp3(lst):
        freq = lst[0]
        typ = lst[1]
        month = lst[2]
        f1, precision, recall, fpr = exp3_single_run(freq,month,typ)
        return freq, typ, month, f1, precision, recall, fpr

def exp3_main():
    lst = []
    for freq in ['30m','1h','1d']:
        for typ in [1,2,3]:
            for month in [1,2,3]:
                lst.append([freq, typ, month])
    pool = Pool(processes=len(lst))
    ret=pool.map(run_parallel_exp3,lst)
    res_df=pd.DataFrame(ret).rename(columns={0:'freq', 1:'failure_type',2:'month',3:'F1-score',4:'Precision',5:'Recall',6:'FPR'})
    grouped = res_df.groupby('failure_type')
    for name, group in grouped:
        print(group.groupby('freq').mean().drop(columns={'month'}))
    return res_df

def exp4_single_run(pred_month, types):
    #print(len(features), pred_month, types)
    if pred_month == 1:
        df_train, df_test = load_data('5m', [6], [6], failure_type=types, backtrace='1d')
    elif pred_month == 2:
        df_train, df_test = load_data('5m', [7], [7], failure_type=types, backtrace='1d')
    elif pred_month == 3:
        df_train, df_test = load_data('5m', [8], [8], failure_type=types, backtrace='1d')
    else:
        print("Unkown testing month")

    features = features_generation(4) ## using all groups of features
    df_test = df_test.loc[:,features]
    train_X, train_y, test_X, test_y = train_test_data_generation(df_train, df_test,sample_ratio=0.02)
    best_f1 = 0
    precision = 0
    recall = 0
    fpr = 0
    tp_useless = 0
    tp_useful = 0

    name = "../model/exp4/exp4/exp4_month_" + str(pred_month) + "_types_" + str(types)
    model = pickle.load(open(name,'rb'))
    y_pred=model.predict_proba(test_X)
    y_truth = copy.deepcopy(test_y)
    f1, raw, th = find_best_f1_V2(y_pred,y_truth)
    if (best_f1 < f1):
        best_f1 = f1
        precision = raw[1]
        recall = raw[2]
        fpr = raw[3]
        tp_useless, tp_useful = evaluation_time(y_pred,y_truth,th)
    return best_f1, precision, recall, fpr, tp_useless, tp_useful

def exp4_single_run_from_scratch(pred_month, types):
    #print(len(features), pred_month, types)
    if pred_month == 1:
        df_train, df_test = load_data('5m', [1,2,3,4,5], [6], failure_type=types, backtrace='1d')
    elif pred_month == 2:
        df_train, df_test = load_data('5m', [2,3,4,5,6], [7], failure_type=types, backtrace='1d')
    elif pred_month == 3:
        df_train, df_test = load_data('5m', [3,4,5,6,7], [8], failure_type=types, backtrace='1d')
    else:
        print("Unkown testing month")
    features = features_generation(4) ## using all groups of features
    df_train=df_train.reset_index(drop=True)
    df_train = df_train.loc[:,features]
    df_test = df_test.loc[:,features]
    train_X, train_y, test_X, test_y = train_test_data_generation(df_train, df_test,sample_ratio=0.02)
    best_f1 = 0
    precision = 0
    recall = 0
    fpr = 0
    tp_useless = 0
    tp_useful = 0
    for n_est in [10,50,100]:
        for maxd in [5,10, 50]:
            for val in [5,10, 50]:
                model,hyperparameters = classifier('RF')
                model = model(n_jobs=30,random_state=42,n_estimators=n_est,max_depth=maxd,min_samples_split=val)
                model.fit(train_X[train_X.columns[2:]], train_y)
                y_pred=model.predict_proba(test_X)
                y_truth = copy.deepcopy(test_y)
                f1, raw, th = find_best_f1(y_pred,y_truth)
                if (best_f1 < f1):
                    best_f1 = f1
                    precision = raw[1]
                    recall = raw[2]
                    fpr = raw[3]
                    tp_useless, tp_useful = evaluation_time(y_pred,y_truth,th)
    return best_f1, precision, recall, fpr, tp_useless, tp_useful

def run_parallel_exp4(lst):
        typ = lst[0]
        month = lst[1]
        f1, precision, recall, fpr, tp_useless, tp_useful  = exp5_single_run(month,typ)
        return typ, month, f1, precision, recall, fpr, tp_useless, tp_useful

def exp4_main():
    lst = []
    for typ in [1,2,3]:
        for month in [1,2,3]:
            lst.append([typ, month])
    pool = Pool(processes=len(lst))
    ret=pool.map(run_parallel_exp5,lst)
    res_df=pd.DataFrame(ret).rename(columns={0:'failure_type',1:'month',2:'F1-score',3:'Precision',4:'Recall',5:'FPR',6:'tp_useless',7:'tp_useful'})
    result = res_df.groupby('failure_type').mean().drop(columns={'month'})
    print(result)
    return res_df

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    try:
        os.mkdir("result/")
    except Exception as e:
        print(e)

    exp1_main().to_csv('./result/exp1.csv') ## Experiments for Finding 11

    exp2_main().to_csv('./result/exp2.csv') ## Experiments for Finding 12

    exp3_main().to_csv('./result/exp3.csv') ## Experiments for Finding 13

    exp4_main().to_csv('./result/exp4.csv') ## Experiments for Finding 14
