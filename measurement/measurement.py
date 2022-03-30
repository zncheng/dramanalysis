import os
import sys
import numpy as np
import pandas as pd
import warnings

def overall_distribution(df_log, df_tickets):
    total_server = 258993 ## total distinc num of servers
    monthly_server_population = [185961,190428,196497,198658,202245,205709,213998,236471]
    monthly_server_with_error = []
    monthly_server_with_failure = []
    counter = 0
    for i in ['0001-01','0001-02','0001-03','0001-04','0001-05','0001-06','0001-07','0001-08']:
        frac_error_server = len(df_log[df_log['month'] == i].sid.unique()) / monthly_server_population[counter]
        monthly_server_with_error.append(frac_error_server)
        frac_failed_server = len(df_tickets[df_tickets['month'] == i].sid.unique()) / monthly_server_population[counter]
        monthly_server_with_failure.append(frac_failed_server)
        counter = counter + 1
    ff = open('./result/overall_distribution.txt',"w")
    print("Monthly fraction of server with errors: mean ",np.mean(monthly_server_with_error), file=ff)
    print(monthly_server_with_error,file=ff)
    print("Fraction of server with errorr in eight month ",len(df_log.sid.unique()) / total_server, file=ff)
    print("Monthly fraction of server with failures: mean",np.mean(monthly_server_with_failure), file=ff)
    print(monthly_server_with_failure,file=ff)
    print("Overall error rate in eight month ",len(df_log.sid.unique()) / total_server)
    print("Average error rate: ", np.mean(monthly_server_with_error))
    print("Average failure rate: ", np.mean(monthly_server_with_failure))
    print("Overall distribution analysis done!")
    f.close()

def compute_time_diff(df_tickets_log):
    base ={'01':0,'02':31,'03':61,'04':92,'05':123,'06':152,'07':183,'08':213}
    df_tickets_log['error_date'] = df_tickets_log['error_time'].str[5:7]
    df_tickets_log['error_date'] = df_tickets_log['error_date'].apply(lambda x: base[x])
    df_tickets_log['failed_date'] = df_tickets_log['failed_time'].str[5:7]
    df_tickets_log['failed_date'] = df_tickets_log['failed_date'].apply(lambda x: base[x])
    df_tickets_log['error_time_in_day'] = df_tickets_log['error_time'].str[8:10].astype(int)
    df_tickets_log['failed_time_in_day'] = df_tickets_log['failed_time'].str[8:10].astype(int)
    df_tickets_log['error_time_offset'] = pd.to_datetime(df_tickets_log['error_time'].str[11:20])
    df_tickets_log['failed_time_offset'] = pd.to_datetime(df_tickets_log['failed_time'].str[11:20])
    df_tickets_log['time_diff'] = (df_tickets_log['failed_time_offset'] - df_tickets_log['error_time_offset']).dt.total_seconds() + (df_tickets_log['failed_time_in_day'] + df_tickets_log['failed_date'] - df_tickets_log['error_time_in_day'] - df_tickets_log['error_date']) * 24 * 3600

def predictable_analysis(df_tickets_log, tickets):   ### Finding 2
    num_total = {}
    for i in [1,2,3]: ## for each type of tickets
        num_total[i] = len(tickets[tickets['failure_type'] == i])
    f = open('./result/predictale_analysis.txt','w')
    time_index = ["1m","5m","10m","20m","30m","1h","1d","7d","30d"]
    print("time\tUE-driven\tCE-driven\tMisc",file=f)
    count = 0
    for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
        row_res = str(time_index[count])
        df_tm = df_tickets_log[df_tickets_log['time_diff'] > tm.total_seconds()]
        for typ in [1,2,3]:
            row_res = row_res + '\t' + "{:.6f}".format(len(df_tm[df_tm['failure_type'] == typ].sid.unique()) / num_total[typ])
        print(row_res,file=f)
        count = count + 1
    f.close()
    print("Predictable analsyis done!")

def num_ce_analysis(df_tickets_log):   ### Finding 3
    f = open('./result/num_ce_analysis.txt','w')
    time_index = ["1m","5m","10m","20m","30m","1h","1d","7d","30d"]
    print("time\tUE-driven\tCE-driven\tMisc",file=f)
    count = 0
    for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
        row_res = str(time_index[count])
        df_tm = df_tickets_log[df_tickets_log['time_diff'] > tm.total_seconds()]
        for typ in [1,2,3]:
            df_typ = df_tm[df_tm['failure_type'] == typ].groupby('sid').error_time.count().reset_index(name='val')
            row_res = row_res + '\t' + "{:.4f}".format(df_typ['val'].mean())
        print(row_res,file=f)
        count = count + 1
    f.close()
    print("Average umber of CE per failure analsyis done!")

def mtbe_analysis(df_tickets_log):   ### Finding 4
    f = open('./result/mtbe_analysis.txt','w')
    time_index = ["1m","5m","10m","20m","30m","1h","1d","7d","30d"]
    print("time\tUE-driven\tCE-driven\tMisc",file=f)
    count = 0
    for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
        row_res = str(time_index[count])
        df_tm = df_tickets_log[df_tickets_log['time_diff'] > tm.total_seconds()]
        for typ in [1,2,3]:
            df_typ = df_tm[df_tm['failure_type'] == typ].sort_values(by=['sid','error_time'])
            df_typ['res'] = df_typ.groupby('sid')['time_diff'].diff()
            df_typ = df_typ.dropna()
            df_typ['res'] = -df_typ['res']
            df_res = df_typ.groupby('sid')['res'].mean().reset_index(name='MTBE')
            row_res = row_res + '\t' + "{:.4f}".format(df_res['MTBE'].median() / 60.0)
        print(row_res,file=f)
        count = count + 1
    f.close()
    print("Mean time between error analsyis done!")

def component_breakdown(df):   ### Preliminary for Finding 5 and 6
    warnings.filterwarnings('ignore')
    df['socketid'] = df['memoryid'].apply(lambda x: 0 if x < 12 else 1)
    df['channelid'] = df['memoryid'].apply(lambda x: int((x % 12) / 2))
    df['dimmid'] = df['memoryid'].apply(lambda x : x % 2)
    res_ce_num = {}
    res_sid_num = {}

    ## socket failures
    grouped = df.groupby(['sid','socketid','channelid']).error_time.count().reset_index(name='num')
    df_channel=grouped.groupby(['sid','socketid']).channelid.count().reset_index(name='channel_count')
    df_error=grouped.groupby(['sid','socketid']).num.sum().reset_index(name='error_count')
    df_channel_error = pd.merge(df_channel,df_error,how='inner',on=['sid','socketid'])
    df_socket_result= df_channel_error[(df_channel_error['error_count'] > 1000) & (df_channel_error['channel_count'] > 1)]
    df=pd.merge(df,df_socket_result,how='left',on=['sid','socketid'])
    df_res=df[(df['error_count'] > 1000) & (df['channel_count'] > 1)]
    for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
        df_tm = df_res[df_res['time_diff'] > tm.total_seconds()]
        res_ce_num[tm] = str(len(df_tm))
        res_sid_num[tm] = str(len(df_tm.sid.unique()))
    df=df[~((df['error_count'] > 1000) & (df['channel_count'] > 1))].drop(columns=['error_count','channel_count'])

    ## channel failures
    grouped=df.groupby(['sid','socketid','channelid','dimmid','rankid','bankid']).error_time.count().reset_index(name='num')
    df_bank=grouped.groupby(['sid','socketid','channelid','dimmid','rankid']).bankid.count().reset_index(name='bank_count')
    df_error=grouped.groupby(['sid','socketid','channelid','dimmid','rankid']).num.sum().reset_index(name='error_count')
    df_bank_error = pd.merge(df_bank,df_error,how='inner',on=['sid','socketid','channelid','dimmid','rankid'])
    df_error_total=df_bank_error.groupby(['sid','socketid','channelid']).error_count.sum().reset_index(name='total_error_count')
    df_bank_total=df_bank_error.groupby(['sid','socketid','channelid']).bank_count.sum().reset_index(name='total_bank_count')
    df_total_result=pd.merge(df_error_total,df_bank_total,how='inner',on=['sid','socketid','channelid'])
    df_channel_result= df_total_result[(df_total_result['total_error_count'] > 1000) & (df_total_result['total_bank_count'] > 1)]
    df=pd.merge(df,df_channel_result,how='left',on=['sid','socketid','channelid'])
    df_res=df[(df['total_error_count'] > 1000) & (df['total_bank_count'] > 1)]
    for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
        df_tm = df_res[df_res['time_diff'] > tm.total_seconds()]
        res_ce_num[tm] = res_ce_num[tm] + ' ' + str(len(df_tm))
        res_sid_num[tm] = res_sid_num[tm]  + ' ' + str(len(df_tm.sid.unique()))
    df=df[~((df['total_error_count'] > 1000) & (df['total_bank_count'] > 1))].drop(columns=['total_error_count','total_bank_count'])

    ## bank failures
    grouped=df.groupby(['sid','socketid','channelid','dimmid','rankid','bankid','row']).error_time.count().reset_index(name='num')
    df_row=grouped.groupby(['sid','socketid','channelid','dimmid','rankid','bankid']).row.count().reset_index(name='row_count')
    df_error=grouped.groupby(['sid','socketid','channelid','dimmid','rankid','bankid']).num.sum().reset_index(name='error_count')
    df_row_error = pd.merge(df_row,df_error,how='inner',on=['sid','socketid','channelid','dimmid','rankid','bankid'])
    df_bank_result= df_row_error[(df_row_error['error_count'] > 1000) & (df_row_error['row_count'] > 1)]
    df=pd.merge(df,df_bank_result,how='left',on=['sid','socketid','channelid','dimmid','rankid','bankid'])
    df_res=df[(df['error_count'] > 1000) & (df['row_count'] > 1)]
    for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
        df_tm = df_res[df_res['time_diff'] > tm.total_seconds()]
        res_ce_num[tm] = res_ce_num[tm] + ' ' + str(len(df_tm))
        res_sid_num[tm] = res_sid_num[tm]  + ' ' + str(len(df_tm.sid.unique()))
    df=df[~((df['error_count'] > 1000) & (df['row_count'] > 1))].drop(columns=['error_count','row_count'])

    ## row failures
    grouped=df.groupby(['sid','socketid','channelid','dimmid','rankid','bankid','row','col']).error_time.count().reset_index(name='num')
    df_column=grouped.groupby(['sid','socketid','channelid','dimmid','rankid','bankid','row']).col.count().reset_index(name='col_count')
    df_error=grouped.groupby(['sid','socketid','channelid','dimmid','rankid','bankid','row']).num.sum().reset_index(name='error_count')

    df_column_error = pd.merge(df_column,df_error,how='inner',on=['sid','socketid','channelid','dimmid','rankid','bankid','row'])
    df_row_result= df_column_error[df_column_error['col_count'] > 1]
    df=pd.merge(df,df_row_result,how='left',on=['sid','socketid','channelid','dimmid','rankid','bankid','row'])
    df_res=df[df['col_count'] > 1]
    for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
        df_tm = df_res[df_res['time_diff'] > tm.total_seconds()]
        res_ce_num[tm] = res_ce_num[tm] + ' ' + str(len(df_tm))
        res_sid_num[tm] = res_sid_num[tm]  + ' ' + str(len(df_tm.sid.unique()))
    df=df[~(df['col_count'] > 1)].drop(columns=['error_count','col_count'])

    ## column failures
    grouped=df.groupby(['sid','socketid','channelid','dimmid','rankid','bankid','row','col']).error_time.count().reset_index(name='num')
    df_row=grouped.groupby(['sid','socketid','channelid','dimmid','rankid','bankid','col']).row.count().reset_index(name='row_count')
    df_error=grouped.groupby(['sid','socketid','channelid','dimmid','rankid','bankid','col']).num.sum().reset_index(name='error_count')
    df_row_error = pd.merge(df_row,df_error,how='inner',on=['sid','socketid','channelid','dimmid','rankid','bankid','col'])
    df_col_result= df_row_error[df_row_error['row_count'] > 1]
    df=pd.merge(df,df_col_result,how='left',on=['sid','socketid','channelid','dimmid','rankid','bankid','col'])
    df_res=df[df['row_count'] > 1]
    for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
        df_tm = df_res[df_res['time_diff'] > tm.total_seconds()]
        res_ce_num[tm] = res_ce_num[tm] + ' ' + str(len(df_tm))
        res_sid_num[tm] = res_sid_num[tm]  + ' ' + str(len(df_tm.sid.unique()))
    df=df[~(df['row_count'] > 1)].drop(columns=['error_count','row_count'])

    ## cell failures
    df=df.sort_values(by=['sid','socketid','channelid','dimmid','rankid','bankid','row','col','error_time'])
    df['time_res']=df.groupby(['sid','socketid','channelid','dimmid','rankid','bankid','row','col'])['time_diff'].diff().fillna(-70)
    df['time_res']=-df['time_res']
    df_cell=df.groupby(['sid','socketid','channelid','dimmid','rankid','bankid','row','col'])['time_res'].min().reset_index(name='min_diff')
    df=pd.merge(df,df_cell,how='left',on=['sid','socketid','channelid','dimmid','rankid','bankid','row','col'])
    df_res=df[df['min_diff'] <= 60]
    for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
        df_tm = df_res[df_res['time_diff'] > tm.total_seconds()]
        res_ce_num[tm] = res_ce_num[tm] + ' ' + str(len(df_tm))
        res_sid_num[tm] = res_sid_num[tm]  + ' ' + str(len(df_tm.sid.unique()))

    ## random errors failure
    df_res=df[df['min_diff'] > 60]
    for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
        df_tm = df_res[df_res['time_diff'] > tm.total_seconds()]
        res_ce_num[tm] = res_ce_num[tm] + ' ' + str(len(df_tm))
        res_sid_num[tm] = res_sid_num[tm]  + ' ' + str(len(df_tm.sid.unique()))

    return res_ce_num, res_sid_num

def component_breakdown_main(df):
    df_res = df.loc[:,['sid','memoryid','rankid','bankid','row','col','error_time','failure_type','time_diff']]
    num_ce_res = {}
    num_sid_res = {}
    for typ in [1,2,3]:
        df_type = df_res[df_res['failure_type'] == typ]
        ce, sid= component_breakdown(df_type)
        num_ce_res[typ] = ce
        num_sid_res[typ] = sid
    return num_ce_res, num_sid_res

def frac_failure_per_component(sid_res, prediction_window, idx):   ## Finding 5 Part 1
    time_index = ["1m","5m","10m","20m","30m","1h","1d","7d","30d"]
    component_index = ['socket', 'channel','bank','row','column','cell','random']
    type_index = ["ue","ce","misc"]
    f = open("./result/frac_failure_per_compoent_" + str(time_index[idx]) + ".txt","w")
    print("component\tresult\tfailure_type",file=f)
    ## total number of failures for each failures type and each prediction window from Finding 2
    total_num = { 1: [547,483,463,422,393,357,190,133,97],
                  2: [792,733,668,532,388,361,257,216,166],
                  3: [760,758,752,744,739,723,307,233,172]
                }
    for typ in [1,2,3]:
        lst = sid_res[typ][prediction_window].split()
        for i in range(len(lst)):
            print(component_index[i]+"\t"+str(int(lst[i])/total_num[typ][idx])+"\t"+type_index[typ-1],file=f)
    f.close()
    print("Fraction of failures per component analysis for window " + time_index[idx] + " done!")

def frac_ce_per_component(num_ce, time, idx):  ## Finding 5 Part 2
    time_index = ["1m","5m","10m","20m","30m","1h","1d","7d","30d"]
    component_index = ['socket', 'channel','bank','row','column','cell','random']
    type_index = ["ue","ce","misc"]
    f = open("./result/frac_ce_per_compoent_" + str(time_index[idx]) + ".txt","w")
    print("component\tvalue\tfailure_type",file=f)
    for typ in [1,2,3]:
        lst = num_ce[typ][time].split()
        total_num = 0
        for i in range(len(lst)):
            total_num = total_num + int(lst[i])
        for i in range(len(lst)):
            print(component_index[i]+"\t"+str(int(lst[i])/total_num)+"\t"+type_index[typ-1], file=f)
    print("Fraction of CEs per component analysis for window " + time_index[idx] + " done!")
    f.close()

def error_breakdown_by_component(num_ce,failure_type): ### Finding 6
    component_index = ["socket","channel","bank","row","column","cell","random"]
    time_index = ["1m","5m","10m","20m","30m","1h","1d","7d","30d"]
    type_index = ["ue","ce","misc"]
    total_num = [269733,6897677,4181363]
    f = open("./result/failures_" + type_index[failure_type-1] + "_breakdown.txt","w")
    print("time\tvalue\tcomponent",file=f)
    counter = 1
    for time in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
        lst = num_ce[failure_type][time].split()
        ## focus on channel, bank, row, column
        for i in [1,2,3,4]:
            print(time_index[counter-1]+"\t"+str(int(lst[i])/total_num[failure_type-1])+"\t"+component_index[i], file=f)
        counter = counter + 1
    f.close()
    print('Error breakdown by component analysis for failures type ' + type_index[failure_type-1] + ' done!')

def hardware_configuration_impact_analysis(df_res, factor, idx):   ### Finding 7, 8, 9
    total_failures = {1:567, 2:809, 3:761} ## overall number of failures of each types
    time_index = ["1m","5m","10m","20m","30m","1h","1d","7d","30d"]
    type_index = ["ue","ce","misc"]
    factors = [['A1','A2','B1','B2','B3','C1','C2'],
             [8, 12,16,24],
             ['M1','M2','M3','M4']]
    dimm_num_name = {8:'8-dimm',12:'12-dimm',16:'16-dimm',24:'24-dimm'}
    for typ in [1,2,3]:
        f = open("./result/" + factor + "_" + type_index[typ-1] + "_breakdown.txt","w")
        print('time\tfraction\tconfiguration', file = f)
        df_final = df_res[df_res['failure_type'] == typ]
        counter = 1
        for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
            df_tm = df_final[df_final['time_diff'] > tm.total_seconds()]
            for i in factors[idx]:
                value = len(df_tm[df_tm[factor] == i].sid.unique()) / total_failures[typ]
                if i == 8 or i == 12 or i == 16 or i == 24:
                    print(time_index[counter-1] + '\t' + str(value) + '\t' + dimm_num_name[i], file = f)
                else:
                    print(time_index[counter-1] + '\t' + str(value) + '\t' + str(i), file = f)
            counter = counter + 1
        f.close()

def failure_rate_breakdown(df_res, factor):   ## Finding 7 additional
    model_population = [77670,71641,38006,514,10540,32534,15549]
    dimm_population = [35676,151988,7085,51705]
    server_population = [58161,127818,52931,7544]
    df_final = df_res.drop_duplicates('sid')
    res = []
    idx = 0;
    if factor == "DRAM_model":
        for mod in ['A1','A2','B1','B2','B3','C1','C2']:
            res.append(len(df_final[df_final['DRAM_model'] == mod])/model_population[idx] * 100)
            idx = idx + 1
    elif factor == 'DIMM_number':
        for dimm in [8,12,16,24]:
            res.append(len(df_final[df_final['DIMM_number'] == dimm])/ dimm_population[idx] * 100)
            idx = idx + 1
    elif factor == 'server_manufacturer':
        for ser in ['M1','M2','M3','M4']:
            res.append(len(df_final[df_final['server_manufacturer'] == ser])/ server_population[idx] * 100)
            idx = idx + 1
    print(factor + " failure rate breakdown: ", res)


def failure_number_breakdown(df_res, factor):   ## Finding 7 additional
    model_population = [77670,71641,38006,514,10540,32534,15549]
    dimm_population = [35676,151988,7085,51705]
    server_population = [58161,127818,52931,7544]
    df_final = df_res.drop_duplicates('sid')
    res = []
    idx = 0;
    if factor == "DRAM_model":
        for mod in ['A1','A2','B1','B2','B3','C1','C2']:
            res.append(len(df_final[df_final['DRAM_model'] == mod]))
            idx = idx + 1
    elif factor == 'DIMM_number':
        for dimm in [8,12,16,24]:
            res.append(len(df_final[df_final['DIMM_number'] == dimm]))
            idx = idx + 1
    elif factor == 'server_manufacturer':
        for ser in ['M1','M2','M3','M4']:
            res.append(len(df_final[df_final['server_manufacturer'] == ser]))
            idx = idx + 1
    print(factor + " failure number breakdown: ", res)

def read_scrubbing_analysis(df_res):  ### Finding 10
    type_index = ["ue","ce","misc"]
    type_population = {1:547, 2:792, 3:760}
    time_index = ["1m","5m","10m","20m","30m","1h","1d","7d","30d"]
    f1 = open("./result/read_error_mean.txt","w")
    f2 = open("./result/scrub_error_mean.txt","w")
    for ff in [f1, f2]:
        print('time\tvalue\tfailure_type',file=ff)
    for typ in [1,2,3]:
        counter = 1
        df_type = df_res[df_res['failure_type'] == typ]
        for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
            df_tm = df_type[df_type['time_diff'] > tm.total_seconds()]
            df_read = df_tm[df_tm['error_type'] == 1].groupby('sid')['error_time'].count().reset_index(name='error_num')
            df_scrb = df_tm[df_tm['error_type'] == 2].groupby('sid')['error_time'].count().reset_index(name='error_num')
            print(time_index[counter-1] + '\t' + str(df_read['error_num'].mean()) + '\t' + type_index[typ -1], file = f1)
            print(time_index[counter-1] + '\t' + str(df_scrb['error_num'].mean()) + '\t' + type_index[typ -1], file = f2)
            counter = counter + 1
    for ff in [f1, f2]:
        ff.close()
    print("Read/scrubbing error analysis done!")

def soft_hard_analysis(df_res):   ### Finding 11
    time_index = ["1m","5m","10m","20m","30m","1h","1d","7d","30d"]
    df_final = df_res.sort_values(by=['sid','memoryid','rankid','bankid','row','col','error_time'])
    df_first = df_final.drop_duplicates(['sid','memoryid','rankid','bankid','row','col'],keep='first').rename(columns={'time_diff':'first_time'})
    df_last = df_final.drop_duplicates(['sid','memoryid','rankid','bankid','row','col'],keep='last').rename(columns={'time_diff':'last_time'})
    df_cell = df_first.merge(df_last, on = ['sid','memoryid','rankid','bankid','row','col'], how='inner')
    df_cell['time_res'] = (df_cell['first_time'] - df_cell['last_time']) / 3600 / 24
    df_cell['is_hard'] = df_cell['time_res'].apply(lambda x: 1 if np.round(x) >= 1 else 0)  ## if 0 soft error, else hard errors
    df_final = df_final.merge(df_cell.loc[:,['sid','memoryid','rankid','bankid','row','col','is_hard']], on = ['sid','memoryid','rankid','bankid','row','col'], how='inner')
    type_index = ["ue","ce","misc"]
    f1 = open("./result/hard_error_mean.txt","w")
    f2 = open("./result/soft_error_mean.txt","w")
    for ff in [f1, f2]:
        print('time\tvalue\tfailure_type',file=ff)
    for typ in [1,2,3]:
        counter = 1
        df_type = df_final[df_final['failure_type'] == typ]
        for tm in [pd.Timedelta(minutes=1),pd.Timedelta(minutes=5),pd.Timedelta(minutes=10),pd.Timedelta(minutes=20),pd.Timedelta(minutes=30),pd.Timedelta(hours=1),pd.Timedelta(days=1),pd.Timedelta(weeks=1),pd.Timedelta(days=30)]:
            df_tm = df_type[df_type['time_diff'] > tm.total_seconds()]
            df_hard_error = df_tm[df_tm['is_hard'] == 1].groupby(['sid'])['memoryid'].count().reset_index(name='error_num')
            df_soft_error = df_tm[df_tm['is_hard'] == 0].groupby(['sid'])['memoryid'].count().reset_index(name='error_num')
            print(time_index[counter-1] + '\t' + str(df_hard_error['error_num'].mean()) + '\t' + type_index[typ -1], file = f1)
            print(time_index[counter-1] + '\t' + str(df_soft_error['error_num'].mean()) + '\t' + type_index[typ -1], file = f2)
            counter = counter + 1
    for ff in [f1, f2]:
        ff.close()
    print("hard/soft error analysis done!")

if __name__ == '__main__':
    prefix_dir = sys.argv[1]
    try:
        os.mkdir("result/")
    except Exception as e:
        print(e)
    
    ## Load raw data
    df_mcelog = pd.read_csv(prefix_dir + 'mcelog.csv')
    df_inventory = pd.read_csv(prefix_dir + 'inventory.csv')
    df_tickets = pd.read_csv(prefix_dir + 'trouble_tickets.csv')

    ## remove write errors
    df_mcelog = df_mcelog[df_mcelog['error_type'] < 3]
    df_log = pd.merge(df_mcelog,df_inventory,how='inner',on='sid')
    df_log['month'] = df_log['error_time'].apply(lambda x: x[0:7])
    df_tickets['month'] = df_tickets['failed_time'].apply(lambda x: x[0:7])

    overall_distribution(df_log,df_tickets)   ## Finding 1

    ## merge tickets and mcelog
    df_tickets_log = df_log[df_log['sid'].isin(df_tickets.sid.unique())]
    df_tickets_log = df_tickets_log.reset_index(drop=True)
    tickets = df_tickets[df_tickets['failure_type'] > 0]
    df_tickets_log = df_tickets_log.merge(tickets.loc[:,['sid','failed_time','failure_type']],how='inner', on='sid')
    
    ## compute time diff
    compute_time_diff(df_tickets_log)
    df_tickets_log = df_tickets_log.drop(columns={'error_date', 'failed_date','error_time_in_day', 'failed_time_in_day','error_time_offset','failed_time_offset'})

    predictable_analysis(df_tickets_log, tickets) ## Finding 2
    
    num_ce_analysis(df_tickets_log)  ## Finding 3

    mtbe_analysis(df_tickets_log) ## Finding 4

    num_ce, num_res=component_breakdown_main(df_tickets_log)  ## Finding 5
    frac_failure_per_component(num_res,pd.Timedelta(minutes=5),1)
    frac_ce_per_component(num_ce,pd.Timedelta(minutes=5),1)

    hardware_configuration_impact_analysis(df_tickets_log,'DRAM_model', 0)  ## Finding 6
    failure_rate_breakdown(df_tickets_log,'DRAM_model') ## Finding 6

    hardware_configuration_impact_analysis(df_tickets_log,'DIMM_number', 1)  ## Finding 7

    hardware_configuration_impact_analysis(df_tickets_log,'server_manufacturer', 2)  ## Finding 8

    read_scrubbing_analysis(df_tickets_log)  ## Finding 9

    soft_hard_analysis(df_tickets_log)  ## Finding 10
