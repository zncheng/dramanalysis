from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import numpy as np
import time
import copy
import os
import sys

def offset_all_time(time):
    day_of_month = {1:0,2:31,3:61,4:92,5:123,6:152,7:183,8:213,9:246}
    month = int(time[5:7])
    day = int(time[8:10])
    hour = int(time[11:13])
    minute = int(time[14:16])
    second = int(time[17:22])
    offset = (((day_of_month[month] + day) * 24 + hour) * 60 + minute) * 60 + second
    return int(offset)

def offset2time(offset):
    offset_of_month = [86400,2764800,5356800,8035200,10713600,13219200,15897600,18489600,21340800,23932800]
    time = '0001-0'
    month = 0
    for i in range(len(offset_of_month)):
        if offset < offset_of_month[i]:
            month = i
            break
    time = time + str(month)+'-'
    offset_within_month = offset - offset_of_month[month-1]
    days = int(offset_within_month / (3600 * 24)) + 1
    if(days<10):
        time = time + str(0) + str(days)
    else:
        time = time + str(days)
    time = time + ' '
    hours = int((offset_within_month % (24 * 3600)) / 3600)
    if(hours<10):
        time = time + str(0) + str(hours)
    else:
        time = time + str(hours)
    time = time + ':'
    minutes = int((offset_within_month % 3600) / 60)
    if(minutes<10):
        time = time + str(0) + str(minutes)
    else:
        time = time + str(minutes)
    seconds = (offset_within_month % 3600) % 60
    time = time + ':'
    if(seconds<10):
        time = time + str(0) + str(seconds)
    else:
        time = time + str(seconds)
    return time

def load_data():
    df = pd.read_csv('../data/mcelog.csv')
    df = df[df['error_type'] < 3]
    df['offset'] = df['error_time'].apply(offset_all_time)
    return df

def large_server_counter_features(sid,args):
    start = args[0]
    end = args[1]
    wind_size = args[2]
    split = range(start, end, wind_size)
    columns_name=['sid','marker']
    res_list = []
    for i in split:
        right_bound = i + wind_size
        df_check = large_df[(large_df['offset'] >= i) & (large_df['offset'] < right_bound)]
        if len(df_check) > 0:
            res_list.append(dict(zip(columns_name, [sid,right_bound])))
    res_df=pd.DataFrame(res_list)
    res_df.columns=columns_name
    final_df=apply_parallel_group(res_df.groupby(['marker']),large_server_counter_features_parallel,64, args)
    final_columns_name=['sid','predict_time']
    for metric in ['counter','mtbe','read','scrub','soft','hard']:
        feature = '5min' + str('_') + metric
        final_columns_name.append(feature)
    final_df=final_df.fillna(0)
    final_df.columns = final_columns_name
    return final_df

def large_server_counter_features_parallel(name, df, args):
    sid = df.iloc[0]['sid']
    right_bound = df.iloc[0]['marker']
    left_bound_5m = right_bound - pd.Timedelta(minutes=5).total_seconds()
    left_bound_1mon = right_bound - pd.Timedelta(days=30).total_seconds()
    df_all = large_df[(large_df['offset'] >= left_bound_1mon) & (large_df['offset'] < right_bound)]
    df_all_first = df_all.drop_duplicates(['memoryid','rankid','bankid','row','col'],keep='first').rename(columns={'offset':'first_time'})
    df_all_last = df_all.drop_duplicates(['memoryid','rankid','bankid','row','col'],keep='last').rename(columns={'offset':'last_time'})
    df_cell = df_all_first.merge(df_all_last,on=['memoryid','rankid','bankid','row','col'],how='inner')
    df_cell['TBE'] = df_cell['last_time'] - df_cell['first_time'] / 3600 / 24  # unit in days
    df_cell['is_hard'] = df_cell['TBE'].apply(lambda x: 1 if np.round(x) >= 1 else 0) # if 0 soft error, otherwise hard error
    df_all = df_all.merge(df_cell.loc[:,['memoryid','rankid','bankid','row','col','is_hard']],on=['memoryid','rankid','bankid','row','col'], how='inner')
    df_5min = df_all[(df_all['offset'] >= left_bound_5m) & (df_all['offset'] < right_bound)]
    value = [sid,offset2time(right_bound),
                     len(df_5min),
                     df_5min['offset'].sort_values().diff().mean(),
                     len(df_5min[df_5min['error_type'] == 1]),
                     len(df_5min[df_5min['error_type'] == 2]),
                     len(df_5min[df_5min['is_hard'] == 0]),
                     len(df_5min[df_5min['is_hard'] == 1])
    ]
    columns_name = ['sid','predict_time']
    for metric in ['counter','mtbe','read','scrub','soft','hard']:
        feature = '5min' + str('_') + metric
        columns_name.append(feature)

    res_df = pd.DataFrame([dict(zip(columns_name, value))])
    if (len(res_df) > 0):
        res_df.columns=columns_name
    return res_df

#### generate features
def counter_features(name, df, args):
    start =args[0]
    end = args[1]
    wind_size = args[2]
    split = range(start, end, wind_size)
    columns_name = ['sid','predict_time']
    for metric in ['counter','mtbe','read','scrub','soft','hard']:
        feature = '5min' + str('_') + metric
        columns_name.append(feature)
    res_list = []
    for i in split:
        right_bound = i + wind_size
        left_bound_1mon = right_bound - int(pd.Timedelta(days=30).total_seconds())
        df_check=df[(df['offset'] >= i) & (df['offset'] < right_bound)]
        if len(df_check) > 0:
            df_all = df[(df['offset'] >= left_bound_1mon) & (df['offset'] < right_bound)]
            df_all_first = df_all.drop_duplicates(['memoryid','rankid','bankid','row','col'],keep='first').rename(columns={'offset':'first_time'})
            df_all_last = df_all.drop_duplicates(['memoryid','rankid','bankid','row','col'],keep='last').rename(columns={'offset':'last_time'})
            df_cell = df_all_first.merge(df_all_last,on=['memoryid','rankid','bankid','row','col'],how='inner')
            df_cell['TBE'] = (df_cell['last_time'] - df_cell['first_time']) / 3600 / 24  # unit in days
            df_cell['is_hard'] = df_cell['TBE'].apply(lambda x: 1 if np.round(x) >= 1 else 0) # if 0 soft error, otherwise hard error
            df_all = df_all.merge(df_cell.loc[:,['memoryid','rankid','bankid','row','col','is_hard']],on=['memoryid','rankid','bankid','row','col'], how='inner')
            df_5min = df_all[(df_all['offset'] >= i) & (df_all['offset'] < right_bound)]
            value = [name,offset2time(right_bound),
                     len(df_5min),
                     df_5min['offset'].sort_values().diff().mean(),
                     len(df_5min[df_5min['error_type'] == 1]),
                     len(df_5min[df_5min['error_type'] == 2]),
                     len(df_5min[df_5min['is_hard'] == 0]),
                     len(df_5min[df_5min['is_hard'] == 1])
            ]
            res_list.append(dict(zip(columns_name, value)))
    res_df=pd.DataFrame(res_list).fillna(0)
    if (len(res_df) > 0):
        res_df.columns=columns_name

    return res_df

def apply_parallel_group(dfGrouped, func, n_jobs, args):
    retList=Parallel(n_jobs=n_jobs,verbose=1,backend="multiprocessing")(delayed(func)(name, group, args) for name, group in dfGrouped)
    return pd.concat(retList)

def counter_features_main(df, month, last_month, start, end, freq):
    raw_df = pd.DataFrame()
    if month == 1:
        raw_df = df[(df['offset'] >= start) & (df['offset'] < end)]
    else:
        current_df = df[(df['offset'] >= start) & (df['offset'] < end)]
        last_df = df[(df['offset'] >= last_month) & (df['offset'] < start)]
        last_df = last_df[last_df['sid'].isin(current_df.sid.unique())]
        raw_df = pd.concat([current_df,last_df])
    final_df = pd.DataFrame()
    final_df = apply_parallel_group(raw_df.groupby(['sid']),counter_features,64,[start,end,freq])
    return final_df

large_df = pd.DataFrame()
def counter_features_main(df, month, last_month, start, end, freq):
    raw_df = pd.DataFrame()
    if month == 1:
        raw_df = df[(df['offset'] >= start) & (df['offset'] < end)]
    else:
        current_df = df[(df['offset'] >= start) & (df['offset'] < end)]
        last_df = df[(df['offset'] >= last_month) & (df['offset'] < start)]
        last_df = last_df[last_df['sid'].isin(current_df.sid.unique())]
        raw_df = pd.concat([current_df,last_df])
    raw_df = raw_df.sort_values(by=['memoryid','rankid','bankid','row','col','offset'])
    num_per_server = raw_df.groupby('sid')['memoryid'].count().reset_index(name='counter')
    small_server = num_per_server[num_per_server['counter'] < 100000]
    large_server = num_per_server[num_per_server['counter'] >= 100000]
    final_df = pd.DataFrame()
    global large_df
    for sid in large_server.sid.unique()[0:1]:
        large_df = raw_df[raw_df['sid'] == sid]
        cnt_df = large_server_counter_features(sid,[start,end,freq])
        file_name = './backup/' + str(month) + '/P1_server_' + sid
        cnt_df.to_pickle(file_name)
        final_df = pd.concat([final_df,cnt_df])
    raw_df = raw_df[raw_df['sid'].isin(small_server.sid.unique())]
    small_df=apply_parallel_group(raw_df.groupby(['sid']),counter_features,64,[start,end,freq])
    final_df = pd.concat([final_df,small_df])
    return final_df
counter_df=counter_features_main(df,1,offset_all_time('0001-01-01 00:00:00'),offset_all_time('0001-01-01 00:00:00'),offset_all_time('0001-02-01 00:00:00'),int(pd.Timedelta(minutes=5).total_seconds()))

## function for component failure classifications
def component_split_fixed_time(df, right_bound):
    error_res = []
    # find socket failure
    grouped=df.groupby(['socketid','channelid']).channelid.count().reset_index(name='num')
    df_channel=grouped.groupby(['socketid']).channelid.count().reset_index(name='channel_count')
    df_error=grouped.groupby(['socketid']).num.sum().reset_index(name='error_count')

    df_channel_error = pd.merge(df_channel,df_error,how='inner',on=['socketid'])
    df_socket_result= df_channel_error[(df_channel_error['error_count'] > 1000) & (df_channel_error['channel_count'] > 1)]

    # save result and remove socket failure
    df=pd.merge(df,df_socket_result,how='left',on=['socketid'])
    df_res=df[(df['error_count'] > 1000) & (df['channel_count'] > 1)]
    for tm in [pd.Timedelta(minutes=5).total_seconds()]:
        df_tm = df_res[df_res['offset'] >= right_bound - tm]
        error_res.append(str(len(df_tm)))

    df=df[~((df['error_count'] > 1000) & (df['channel_count'] > 1))].drop(columns=['error_count','channel_count'])

    grouped=df.groupby(['socketid','channelid','dimmid','rankid','bankid']).channelid.count().reset_index(name='num')
    df_bank=grouped.groupby(['socketid','channelid','dimmid','rankid']).bankid.count().reset_index(name='bank_count')
    df_error=grouped.groupby(['socketid','channelid','dimmid','rankid']).num.sum().reset_index(name='error_count')

    df_bank_error = pd.merge(df_bank,df_error,how='inner',on=['socketid','channelid','dimmid','rankid'])
    df_error_total=df_bank_error.groupby(['socketid','channelid']).error_count.sum().reset_index(name='total_error_count')
    df_bank_total=df_bank_error.groupby(['socketid','channelid']).bank_count.sum().reset_index(name='total_bank_count')
    df_total_result=pd.merge(df_error_total,df_bank_total,how='inner',on=['socketid','channelid'])
    df_channel_result= df_total_result[(df_total_result['total_error_count'] > 1000) & (df_total_result['total_bank_count'] > 1)]

    #remove channel failure
    df=pd.merge(df,df_channel_result,how='left',on=['socketid','channelid'])
    df_res=df[(df['total_error_count'] > 1000) & (df['total_bank_count'] > 1)]
    for tm in [pd.Timedelta(minutes=5).total_seconds()]:
        df_tm = df_res[df_res['offset'] >= right_bound - tm]
        error_res.append(str(len(df_tm)))
    df=df[~((df['total_error_count'] > 1000) & (df['total_bank_count'] > 1))].drop(columns=['total_error_count','total_bank_count'])


    grouped=df.groupby(['socketid','channelid','dimmid','rankid','bankid','row']).channelid.count().reset_index(name='num')
    df_row=grouped.groupby(['socketid','channelid','dimmid','rankid','bankid']).row.count().reset_index(name='row_count')
    df_error=grouped.groupby(['socketid','channelid','dimmid','rankid','bankid']).num.sum().reset_index(name='error_count')
    df_row_error = pd.merge(df_row,df_error,how='inner',on=['socketid','channelid','dimmid','rankid','bankid'])
    df_bank_result= df_row_error[(df_row_error['error_count'] > 1000) & (df_row_error['row_count'] > 1)]

    #remove bank failure and find row failure
    df=pd.merge(df,df_bank_result,how='left',on=['socketid','channelid','dimmid','rankid','bankid'])
    df_res=df[(df['error_count'] > 1000) & (df['row_count'] > 1)]
    for tm in [pd.Timedelta(minutes=5).total_seconds()]:
        df_tm = df_res[df_res['offset'] >= right_bound - tm]
        error_res.append(str(len(df_tm)))
    df=df[~((df['error_count'] > 1000) & (df['row_count'] > 1))].drop(columns=['error_count','row_count'])

    grouped=df.groupby(['socketid','channelid','dimmid','rankid','bankid','row','col']).channelid.count().reset_index(name='num')
    df_column=grouped.groupby(['socketid','channelid','dimmid','rankid','bankid','row']).col.count().reset_index(name='col_count')
    df_error=grouped.groupby(['socketid','channelid','dimmid','rankid','bankid','row']).num.sum().reset_index(name='error_count')

    df_column_error = pd.merge(df_column,df_error,how='inner',on=['socketid','channelid','dimmid','rankid','bankid','row'])
    df_row_result= df_column_error[df_column_error['col_count'] > 1]

    #remove row failure and find column failure
    df=pd.merge(df,df_row_result,how='left',on=['socketid','channelid','dimmid','rankid','bankid','row'])
    df_res=df[df['col_count'] > 1]
    for tm in [pd.Timedelta(minutes=5).total_seconds()]:
        df_tm = df_res[df_res['offset'] >= right_bound - tm]
        error_res.append(str(len(df_tm)))
    df=df[~(df['col_count'] > 1)].drop(columns=['error_count','col_count'])

    grouped=df.groupby(['socketid','channelid','dimmid','rankid','bankid','row','col']).channelid.count().reset_index(name='num')
    df_row=grouped.groupby(['socketid','channelid','dimmid','rankid','bankid','col']).row.count().reset_index(name='row_count')
    df_error=grouped.groupby(['socketid','channelid','dimmid','rankid','bankid','col']).num.sum().reset_index(name='error_count')

    df_row_error = pd.merge(df_row,df_error,how='inner',on=['socketid','channelid','dimmid','rankid','bankid','col'])
    df_col_result= df_row_error[df_row_error['row_count'] > 1]

    df=pd.merge(df,df_col_result,how='left',on=['socketid','channelid','dimmid','rankid','bankid','col'])
    df_res=df[df['row_count'] > 1]
    for tm in [pd.Timedelta(minutes=5).total_seconds()]:
        df_tm = df_res[df_res['offset'] >= right_bound - tm]
        error_res.append(str(len(df_tm)))
    df=df[~(df['row_count'] > 1)].drop(columns=['error_count','row_count'])

    df=df.sort_values(by=['socketid','channelid','dimmid','rankid','bankid','row','col','offset'])
    df['time_diff']=df.groupby(['socketid','channelid','dimmid','rankid','bankid','row','col'])['offset'].diff().fillna(70)
    df_cell=df.groupby(['socketid','channelid','dimmid','rankid','bankid','row','col'])['time_diff'].min().reset_index(name='min_diff')
    df=pd.merge(df,df_cell,how='left',on=['socketid','channelid','dimmid','rankid','bankid','row','col'])

    ## cell failure
    df_res=df[df['min_diff'] <= 60]
    for tm in [pd.Timedelta(minutes=5).total_seconds()]:
        df_tm = df_res[df_res['offset'] >= right_bound - tm]
        error_res.append(str(len(df_tm)))

    ## spurious failure
    df_res=df[df['min_diff'] > 60]
    for tm in [pd.Timedelta(minutes=5).total_seconds()]:
        df_tm = df_res[df_res['offset'] >= right_bound - tm]
        error_res.append(str(len(df_tm)))

    return error_res

def component_features(name, df, args):
    start = args[0]
    end = args[1]
    wind_size = args[2]
    split = range(start, end, wind_size)
    columns_name=['sid','predict_time']
    for metric in ['socket_error','channel_error','bank_error','row_error', 'column_error','cell_error','random_error']:
        feature = '5min' + str('_') + metric
        columns_name.append(feature)
    res_list = []
    for i in split:
        right_bound = i + wind_size
        left_bound_1mon = right_bound - pd.Timedelta(days=30).total_seconds()
        df_check=df[(df['offset'] >= i) & (df['offset'] < right_bound)]
        if len(df_check) > 0:
            df_all = df[(df['offset'] >= left_bound_1mon) & (df['offset'] < right_bound)]
            component_res = component_split_fixed_time(df_all, right_bound)
            value = [name,offset2time(right_bound)] + component_res
            res_list.append(dict(zip(columns_name, value)))

    res_df=pd.DataFrame(res_list).fillna(0)
    if (len(res_df) > 0):
        res_df.columns=columns_name

    return res_df

def large_server_parallel_component_features(name, df, args):
    sid = df.iloc[0]['sid']
    right_bound = df.iloc[0]['marker']
    left_bound_1mon = right_bound - pd.Timedelta(days=30).total_seconds()
    df_all = large_df[(large_df['offset'] >= left_bound_1mon) & (large_df['offset'] < right_bound)]
    component_res = component_split_fixed_time(df_all, right_bound)
    columns_name=['sid','predict_time']
    for metric in ['socket_error','channel_error','bank_error','row_error', 'column_error','cell_error','random_error']:
        feature = '5min' + str('_') + metric
        columns_name.append(feature)
    value = [sid, offset2time(right_bound)] + component_res
    res_df = pd.DataFrame([dict(zip(columns_name, value))])
    if (len(res_df) > 0):
        res_df.columns=columns_name
    return res_df

def large_server_component_features_main(sid,args):
    start = args[0]
    end = args[1]
    wind_size = args[2]
    split = range(start, end, wind_size)
    columns_name=['sid','marker']
    res_list = []
    for i in split:
        right_bound = i + wind_size
        df_check = large_df[(large_df['offset'] >= i) & (large_df['offset'] < right_bound)]
        if len(df_check) > 0:
            res_list.append(dict(zip(columns_name, [sid,right_bound])))
    res_df=pd.DataFrame(res_list)
    res_df.columns=columns_name
    final_df=apply_parallel_group(res_df.groupby(['marker']),large_server_parallel_component_features,64, args)
    final_columns_name=['sid','predict_time']
    for metric in ['socket_error','channel_error','bank_error','row_error', 'column_error','cell_error','random_error']:
        feature = '5min' + str('_') + metric
        final_columns_name.append(feature)
    final_df=final_df.fillna(0)
    final_df.columns = final_columns_name
    return final_df

large_df = pd.DataFrame()
def component_features_main(df, month, last_month, start, end, freq):
    raw_df = pd.DataFrame()
    if month == 1:
        raw_df = df[(df['offset'] >= start) & (df['offset'] < end)]
    else:
        current_df = df[(df['offset'] >= start) & (df['offset'] < end)]
        last_df = df[(df['offset'] >= last_month) & (df['offset'] < start)]
        last_df = last_df[last_df['sid'].isin(current_df.sid.unique())]
        raw_df = pd.concat([current_df,last_df])
    raw_df = raw_df.sort_values(by=['memoryid','rankid','bankid','row','col','offset'])
    raw_df['socketid'] = raw_df['memoryid'].apply(lambda x: 0 if x < 12 else 1)
    raw_df['channelid'] = raw_df['memoryid'].apply(lambda x: int((x % 12) / 2))
    raw_df['dimmid'] = raw_df['memoryid'].apply(lambda x : x % 2)
    num_per_server = raw_df.groupby('sid')['memoryid'].count().reset_index(name='counter')
    small_server = num_per_server[num_per_server['counter'] < 100000]
    large_server = num_per_server[num_per_server['counter'] >= 100000]
    final_df = pd.DataFrame()
    global large_df
    for sid in large_server.sid.unique()[0:1]:
        large_df = raw_df[raw_df['sid'] == sid]
        cnt_df = large_server_component_features_main(sid,[start,end,freq])
        file_name = './backup/' + str(month) + '/P1_server_' + sid
        cnt_df.to_pickle(file_name)
        final_df = pd.concat([final_df,cnt_df])
    raw_df = raw_df[raw_df['sid'].isin(small_server.sid.unique())]
    small_df=apply_parallel_group(raw_df.groupby(['sid']),component_features,64,[start,end,freq])
    final_df = pd.concat([final_df,small_df])
    return final_df
import warnings
warnings.filterwarnings('ignore')
component_df=component_features_main(df,1,offset_all_time('0001-01-01 00:00:00'),offset_all_time('0001-01-01 00:00:00'),offset_all_time('0001-02-01 00:00:00'),int(pd.Timedelta(minutes=5).total_seconds()))

#### generate features
def statistical_features(name, df, args):
    start = args[0]
    end = args[1]
    wind_size = args[2]
    split = range(start, end, wind_size)
    columns_name = ['sid','predict_time']
    for metric in ['socket_count','socket_mean','socket_median','socket_std','channel_count','channel_mean','channel_median','channel_std','bank_count','bank_mean','bank_median','bank_std','row_count','row_mean','row_median','row_std','col_count','col_mean','col_median','col_std','cell_count','cell_mean','cell_median','cell_std']:
        feature = '5min' + str('_') + metric
        columns_name.append(feature)
    res_list = []
    for i in split:
        right_bound = i + wind_size
        left_bound_5m = i
        left_bound_1mon = right_bound - pd.Timedelta(days=30).total_seconds()
        df_check=df[(df['offset'] >= i) & (df['offset'] < right_bound)]
        if len(df_check) > 0:
            df_all = df[(df['offset'] >= left_bound_1mon) & (df['offset'] < right_bound)]
            df_all['socketid'] = df_all['memoryid'].apply(lambda x: 0 if x < 12 else 1)
            df_all['channelid'] = df_all['memoryid'].apply(lambda x: int((x % 12) / 2))
            df_5m = df_all[(df_all['offset'] >= left_bound_5m) & (df_all['offset'] < right_bound)]
            value = [name,right_bound]
            for cnt_df in [df_5m]:
                res_socket=cnt_df.groupby(by=['socketid'])['offset'].count().reset_index(name='counts')
                res_channel=cnt_df.groupby(by=['socketid','channelid'])['offset'].count().reset_index(name='counts')
                res_bank=cnt_df.groupby(by=['socketid','channelid','bankid'])['offset'].count().reset_index(name='counts')
                res_row=cnt_df.groupby(by=['socketid','channelid','bankid','row'])['offset'].count().reset_index(name='counts')
                res_column=cnt_df.groupby(by=['socketid','channelid','bankid','col'])['offset'].count().reset_index(name='counts')
                res_cell=cnt_df.groupby(by=['socketid','channelid','bankid','row','col'])['offset'].count().reset_index(name='counts')
                tmp = [len(res_socket), res_socket.counts.mean(), res_socket.counts.std(), res_socket.counts.median(),
                       len(res_channel),res_channel.counts.mean(), res_channel.counts.std(), res_channel.counts.median(),
                       len(res_bank),res_bank.counts.mean(), res_bank.counts.std(), res_bank.counts.median(),
                       len(res_row),res_row.counts.mean(), res_row.counts.std(), res_row.counts.median(),
                       len(res_column),res_column.counts.mean(), res_column.counts.std(), res_column.counts.median(),
                       len(res_cell),res_cell.counts.mean(), res_cell.counts.std(), res_cell.counts.median()
                       ]
                value = value + copy.deepcopy(tmp)
            res_list.append(dict(zip(columns_name, value)))

    res_df=pd.DataFrame(res_list).fillna(0)
    if (len(res_df) > 0):
        res_df.columns=columns_name

    return res_df

def large_server_statistical_features(sid,args):
    start =args[0]
    end = args[1]
    wind_size = args[2]
    split = range(start, end, wind_size)
    columns_name=['sid','marker']
    res_list = []
    for i in split:
        right_bound = i + wind_size
        df_check = large_df[(large_df['offset'] >= i) & (large_df['offset'] < right_bound)]
        if len(df_check) > 0:
            res_list.append(dict(zip(columns_name, [sid,right_bound])))
    res_df=pd.DataFrame(res_list)
    res_df.columns=columns_name
    final_df=apply_parallel_group(res_df.groupby(['marker']),large_server_parallel_statistical_features,64, args)
    final_columns_name=['sid','predict_time']
    for metric in ['socket_count','socket_mean','socket_median','socket_std','channel_count','channel_mean','channel_median','channel_std','bank_count','bank_mean','bank_median','bank_std','row_count','row_mean','row_median','row_std','col_count','col_mean','col_median','col_std','cell_count','cell_mean','cell_median','cell_std']:
        feature = '5min' + str('_') + metric
        final_columns_name.append(feature)
    final_df=final_df.fillna(0)
    final_df.columns = final_columns_name
    return final_df

def large_server_parallel_statistical_features(name, df, args):
    sid = df.iloc[0]['sid']
    right_bound = df.iloc[0]['marker']
    left_bound_5m = right_bound - pd.Timedelta(minutes=5).total_seconds()
    left_bound_1mon = right_bound - pd.Timedelta(days=30).total_seconds()
    df_all = large_df[(large_df['offset'] >= left_bound_1mon) & (large_df['offset'] < right_bound)]
    df_all['socketid'] = df_all['memoryid'].apply(lambda x: 0 if x < 12 else 1)
    df_all['channelid'] = df_all['memoryid'].apply(lambda x: int((x % 12) / 2))
    df_all['dimmid'] = df_all['memoryid'].apply(lambda x : x % 2)
    df_5m = df_all[(df_all['offset'] >= left_bound_5m) & (df_all['offset'] < right_bound)]
    value = [sid,right_bound]
    for cnt_df in[df_5m]:
        res_socket=cnt_df.groupby(by=['socketid'])['offset'].count().reset_index(name='counts')
        res_channel=cnt_df.groupby(by=['socketid','channelid'])['offset'].count().reset_index(name='counts')
        res_bank=cnt_df.groupby(by=['socketid','channelid','bankid'])['offset'].count().reset_index(name='counts')
        res_row=cnt_df.groupby(by=['socketid','channelid','bankid','row'])['offset'].count().reset_index(name='counts')
        res_column=cnt_df.groupby(by=['socketid','channelid','bankid','col'])['offset'].count().reset_index(name='counts')
        res_cell=cnt_df.groupby(by=['socketid','channelid','bankid','row','col'])['offset'].count().reset_index(name='counts')
        tmp = [len(res_socket), res_socket.counts.mean(), res_socket.counts.std(), res_socket.counts.median(),
                       len(res_channel),res_channel.counts.mean(), res_channel.counts.std(), res_channel.counts.median(),
                       len(res_bank),res_bank.counts.mean(), res_bank.counts.std(), res_bank.counts.median(),
                       len(res_row),res_row.counts.mean(), res_row.counts.std(), res_row.counts.median(),
                       len(res_column),res_column.counts.mean(), res_column.counts.std(), res_column.counts.median(),
                       len(res_cell),res_cell.counts.mean(), res_cell.counts.std(), res_cell.counts.median()
        ]
        value = value + copy.deepcopy(tmp)

    columns_name = ['sid','predict_time']
    for metric in ['socket_count','socket_mean','socket_median','socket_std','channel_count','channel_mean','channel_median','channel_std','bank_count','bank_mean','bank_median','bank_std','row_count','row_mean','row_median','row_std','col_count','col_mean','col_median','col_std','cell_count','cell_mean','cell_median','cell_std']:
        feature = '5min' + str('_') + metric
        columns_name.append(feature)

    res_df = pd.DataFrame([dict(zip(columns_name, value))])
    if (len(res_df) > 0):
        res_df.columns=columns_name
    return res_df

large_df = pd.DataFrame()
def statistical_features_main(df, month, last_month, start, end, freq):
    raw_df = pd.DataFrame()
    if month == 1:
        raw_df = df[(df['offset'] >= start) & (df['offset'] < end)]
    else:
        current_df = df[(df['offset'] >= start) & (df['offset'] < end)]
        last_df = df[(df['offset'] >= last_month) & (df['offset'] < start)]
        last_df = last_df[last_df['sid'].isin(current_df.sid.unique())]
        raw_df = pd.concat([current_df,last_df])
    raw_df = raw_df.sort_values(by=['memoryid','rankid','bankid','row','col','offset'])
    raw_df['socketid'] = raw_df['memoryid'].apply(lambda x: 0 if x < 12 else 1)
    raw_df['channelid'] = raw_df['memoryid'].apply(lambda x: int((x % 12) / 2))
    num_per_server = raw_df.groupby('sid')['memoryid'].count().reset_index(name='counter')
    small_server = num_per_server[num_per_server['counter'] < 100000]
    large_server = num_per_server[num_per_server['counter'] >= 100000]
    final_df = pd.DataFrame()
    global large_df
    for sid in large_server.sid.unique()[0:1]:
        large_df = raw_df[raw_df['sid'] == sid]
        cnt_df = large_server_statistical_features(sid,[start,end,freq])
        file_name = './backup/' + str(month) + '/P1_server_' + sid
        cnt_df.to_pickle(file_name)
        final_df = pd.concat([final_df,cnt_df])
    raw_df = raw_df[raw_df['sid'].isin(small_server.sid.unique())]
    small_df=apply_parallel_group(raw_df.groupby(['sid']),statistical_features,64,[start,end,freq])
    final_df = pd.concat([final_df,small_df])
    return final_df
import warnings
warnings.filterwarnings('ignore')
statistical=statistical_features_main(df,1,offset_all_time('0001-01-01 00:00:00'),offset_all_time('0001-01-01 00:00:00'),offset_all_time('0001-02-01 00:00:00'),int(pd.Timedelta(minutes=5).total_seconds()))


def one_hot_encoding_features(df):
    static_df = pd.read_csv('../data/inventory.csv')
    for model in ['A1','A2','B1','B2','B3','C1','C2']:
        model_set = static_df[static_df['DRAM_model'] == model].sid.unique()
        df['model_' + model] = df['sid'].apply(lambda x: 1 if x in model_set else 0)
    for dimm_num in [8,12,16,18]:
        dimm_num_set = static_df[static_df['DIMM_number'] == dimm_num].sid.unique()
        df['dimm_num_' + str(dimm_num)] = df['sid'].apply(lambda x: 1 if x in dimm_num_set else 0)
    for server in ['M1','M2','M3','M4']:
        server_set = static_df[static_df['server_manufacturer'] == server].sid.unique()
        df['manufacturer_' + server] = df['sid'].apply(lambda x: 1 if x in server_set else 0)
    return df

if __name__ == '__main__':
    try:
        os.mkdir("features/")
    except Exception as e:
        print(e)

    df = load_data() ## load all data
    last_month = [offset_all_time('0001-01-01 00:00:00'),offset_all_time('0001-01-01 00:00:00'),offset_all_time('0001-02-01 00:00:00'),offset_all_time('0001-03-01 00:00:00'),offset_all_time('0001-04-01 00:00:00'),offset_all_time('0001-05-01 00:00:00'),offset_all_time('0001-06-01 00:00:00'),offset_all_time('0001-07-01 00:00:00')]
    start = [offset_all_time('0001-01-01 00:00:00'),offset_all_time('0001-02-01 00:00:00'),offset_all_time('0001-03-01 00:00:00'),offset_all_time('0001-04-01 00:00:00'),offset_all_time('0001-05-01 00:00:00'),offset_all_time('0001-06-01 00:00:00'),offset_all_time('0001-07-01 00:00:00'),offset_all_time('0001-08-01 00:00:00')]
    end = [offset_all_time('0001-02-01 00:00:00'),offset_all_time('0001-03-01 00:00:00'),offset_all_time('0001-04-01 00:00:00'),offset_all_time('0001-05-01 00:00:00'),offset_all_time('0001-06-01 00:00:00'),offset_all_time('0001-07-01 00:00:00'),offset_all_time('0001-08-01 00:00:00'),offset_all_time('0001-09-01 00:00:00')]
    freq_name = {pd.Timedelta(minutes=5):'5m',pd.Timedelta(minutes=30):'30m',pd.Timedelta(hours=1):'1h',pd.Timedelta(days=5):'1d'}
    for freq in [pd.Timedelta(minutes=5),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=5)]:
        for i in range(8):
            counter_df = counter_features_main(df,i+1,last_month[i],start[i],end[i], int(freq.total_seconds()))
            component_df = component_features_main(df,i+1,last_month[i],start[i],end[i],int(freq.total_seconds()))
            statistical_df = statistical_features_main(df,i+1,last_month[i],start[i],end[i],int(freq.total_seconds()))
            res_df = pd.merge([counter_df, component_df], on = ['sid','predict_time'], how='inner')
            res_df = pd.merge([res_df, statistical_df], on = ['sid','predict_time'], how='inner')
            res_df = one_hot_encoding_features(res_df)
            res_df.to_csv('./features/features_' + freq_name[freq] + '_month_' + str(i+1) + '.csv', index=False)


