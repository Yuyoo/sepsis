# -*- coding: utf-8 -*-
# 对开始进入ICU的患者体征指标的进行数据预处理

import pandas as pd
import numpy as np

# 年龄处理
def processage(df):
    df['age'] = df['age'].map(lambda x: round(x, 0))
    df.drop(['hadm_id','icustay_id'],axis=1,inplace=True)
    return df


'''
时间区间合并
将正时间轴的所有测量时间点四舍五入到最近邻整小时点
负时间的所有测量时间点都当做0时间点处理
取离整时间点距离最近的测量时间点的测量值作为整点的测量值
如果同一时间点有多个离整时间点最近的测量时间点，则对这些点的测量值取平均数作为整点的测量值 
'''


def timebin0(df, para):
    # col_round = lambda x: round(x,0)
    # 将测量时间与入ICU时间相减得到的
    df['hours'] = df['intime_hours'].map(lambda x: 0 if x < 0 else round(x, 0))  # 将'intime_hours'中的值四舍五入取整
    df['time_dot'] = (df['intime_hours'] - df['hours']).map(lambda x: abs(x))  # 求'intime_hours'与整点的距离值

    df_dot = df.groupby([df['subject_id'], df['hours']])[['time_dot']].min()  # 求距离整点的最近的'intime_hours'的series
    df_dot = df_dot.reset_index(drop=False)  # 去除层次化索引，转化为普通二维dataframe，以便两表联结
    df = pd.merge(df, df_dot, on=['subject_id', 'hours', 'time_dot'], how='inner')  # 内联

    # df['hours'] = df['hours'].map(lambda x: 0 if x<=0 else x)    #处理时间轴上的负值，将时间轴上负值向上取为0

    df_mean = df.groupby(['subject_id', 'hours', 'time_dot'])[para].mean()  # 对距离整点相同的两个时间点的测量值求均值
    df_mean = df_mean.reset_index(drop=False)  # 构造包含'subject_id', 'hours','time_dot',para（均值）属性的一张新表
    df = df.drop([para, 'intime_hours'], axis=1)  # 舍弃原来的测量值para以及intime可能可以唯一标志某一series的量，以便两张表的内联
    df = pd.merge(df, df_mean, on=['subject_id', 'hours', 'time_dot'], how='inner')  # 将两张表内联
    df.drop_duplicates(inplace=True)
    df = df.drop(['time_dot'], axis=1)
    df = df.drop(['hadm_id'], axis=1)
    df = df.drop(['icustay_id'], axis=1)

    # 重排列索引顺序
    df = df.reindex(columns=['subject_id', 'hours', para])

    return df


'''
对时间进行插值操作（一）
单表构造时间序列
构造每个人与时间轴长度的字典
根据字典key-value，构造矩阵
根据矩阵构造dataframe，并与原dataframe联结
'''


def fill_time(df):
    # df = pd.read_csv('output/hr_timebin.csv')
    print df
    patientArray = df['subject_id'].unique()  # 取出所有患者的subject_id
    dict_pA = {}  # 存储患者测量体征时间长度的字典
    matrixList = []  # 存储每个患者测量时间点的矩阵列表
    for patient in patientArray:
        dict_pA[patient] = df['hours'][df['subject_id'] == patient].max()  # 取出患者id对应的时间区间的最大值
        # 生成时间长度的一维id向量，如array([ 9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.])
        sidArray = np.linspace(patient, patient, dict_pA[patient] + 1)
        # 生成从零开始的时间长度的一维增量向量，如array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
        timeArray = np.linspace(0, dict_pA[patient], dict_pA[patient] + 1)

        timeMatrix = np.array((sidArray, timeArray))  # 联结成矩阵
        timeMatrix = timeMatrix.transpose()  # 转置
        matrixList.append(timeMatrix)

    matrixAll = np.concatenate(matrixList)  # 将列表中的每个患者的矩阵联结成所有患者的矩阵
    timedf = pd.DataFrame(matrixAll, columns=['subject_id', 'hours'])  # 将矩阵转换为dataframe
    # 将原数据表与生成的无缺省的时间轴数据表外联结
    df = pd.merge(df, timedf, how='right', left_on=['subject_id', 'hours'], right_on=['subject_id', 'hours'], sort=True)
    df = df
    print df
    print timedf
    return df


'''
对时间进行插值操作（二）
联表构造时间序列，保证同一个人的所有测量参数的时间序列是一致的
'''


def fill_time2(df,timemax):
    # df = pd.read_csv('entericudata/heart_rate.csv')
    # df = timebin0(df, 'heart_rate')
    # print df
    df.drop_duplicates(inplace=True)
    # print df
    matrixList= []
    patientlist = timemax.keys()
    for patient in patientlist:
        sidArray = np.linspace(patient, patient, timemax[patient] + 1)
        timeArray = np.arange(timemax[patient]+1)
        timeMatrix = np.array((sidArray, timeArray))  # 联结成矩阵
        timeMatrix = timeMatrix.transpose()  # 转置
        matrixList.append(timeMatrix)
    matrixAll = np.concatenate(matrixList)  # 将列表中的每个患者的矩阵联结成所有患者的矩阵
    timedf = pd.DataFrame(matrixAll, columns=['subject_id', 'hours'])  # 将矩阵转换为dataframe
    # 将原数据表与生成的无缺省的时间轴数据表外联结
    df = pd.merge(df, timedf, how='right', left_on=['subject_id', 'hours'], right_on=['subject_id', 'hours'], sort=True)
    # print timedf
    return df



'''
求出所有8张表中每个测量指标的测量时间的最大值
'''
def max_max(df0,df1,df2,df3,df4,df5,df6,df7):
    df_list = []
    '''
    df0 = pd.read_csv('entericudata/blood_oxygen_saturation.csv')
    df0 = timebin0(df0, 'blood_oxygen_saturation')
    df1 = pd.read_csv('entericudata/heart_rate.csv')
    df1 = timebin0(df1, 'heart_rate')
    df2 = pd.read_csv('entericudata/ph.csv')
    df2 = timebin0(df2, 'ph')
    df3 = pd.read_csv('entericudata/pulse_pressure.csv')
    df3 = timebin0(df3, 'pulse_pressure')
    df4 = pd.read_csv('entericudata/respiration_rate.csv')
    df4 = timebin0(df4, 'respiration_rate')
    df5 = pd.read_csv('entericudata/sbp.csv')
    df5 = timebin0(df5, 'sbp')
    df6 = pd.read_csv('entericudata/temperature.csv')
    df6 = timebin0(df6, 'temperature')
    df7 = pd.read_csv('entericudata/wbc.csv')
    df7 = timebin0(df7, 'wbc')
    '''

    df_list.append(df0)
    df_list.append(df1)
    df_list.append(df2)
    df_list.append(df3)
    df_list.append(df4)
    df_list.append(df5)
    df_list.append(df6)
    df_list.append(df7)

    slist = []

    for df in df_list:
        patientArray = df['subject_id'].unique()
        dict_pA = {}    # 存储每个患者测量体征时间长度的字典
        for patient in patientArray:
            dict_pA[patient] = df['hours'][df['subject_id'] == patient].max()
        s = pd.Series(dict_pA,name='maxtime')
        s = s.reset_index()
        slist.append(s)

    df_big = pd.concat(slist)
    df_time = df_big.groupby(df_big['index'])['maxtime'].max()
    timemax = df_time.to_dict()
    # df_time = df_time.reset_index(drop=False)
    # df_time.rename(columns={'index':'subject_id'}, inplace = True)
    return timemax


'''
插值函数
'''
def fill_value(df,method):
    if method == '后项插值':
        df = df.groupby('subject_id', as_index=False).fillna(method='bfill')
        df = df.fillna(method='ffill')
    elif method == '前项插值':
        df = df.groupby('subject_id', as_index=False).fillna(method='ffill')
        df = df.fillna(method='bfill')

    return df


'''
滑窗函数
'''
def win_mean(df):
    df0 = df.groupby('subject_id', as_index=False)[
        ['blood_oxygen_saturation', 'heart_rate', 'ph', 'pulse_pressure', 'respiration_rate', 'sbp', 'temperature',
         'wbc']].rolling(window=5, center=False).mean()
    df0.reset_index(drop=True,inplace=True)
    df['blood_oxygen_saturation'] = df0['blood_oxygen_saturation']
    df['heart_rate'] = df0['heart_rate']
    df['ph'] = df0['ph']
    df['pulse_pressure'] = df0['pulse_pressure']
    df['respiration_rate'] = df0['respiration_rate']
    df['sbp'] = df0['sbp']
    df['temperature'] = df0['temperature']
    df['wbc'] = df0['wbc']
    return df

#def win_

def main():
    '''
    df_age = pd.read_csv('entericudata/age.csv')
    df_bos = pd.read_csv('entericudata/blood_oxygen_saturation.csv')
    df_hr = pd.read_csv('entericudata/heart_rate.csv')
    df_ph = pd.read_csv('entericudata/ph.csv')
    df_pp = pd.read_csv('entericudata/pulse_pressure.csv')
    df_rr = pd.read_csv('entericudata/respiration_rate.csv')
    df_sbp = pd.read_csv('entericudata/sbp.csv')
    df_temp = pd.read_csv('entericudata/temperature.csv')
    df_wbc = pd.read_csv('entericudata/wbc.csv')

    df_age = processage(df_age)

    df_bos = timebin0(df_bos,'blood_oxygen_saturation')
    df_hr = timebin0(df_hr,'heart_rate')
    df_ph = timebin0(df_ph,'ph')
    df_pp = timebin0(df_pp,'pulse_pressure')
    df_rr = timebin0(df_rr,'respiration_rate')
    df_sbp = timebin0(df_sbp,'sbp')
    df_temp = timebin0(df_temp,'temperature')
    df_wbc = timebin0(df_wbc,'wbc')

    timemax = max_max(df_bos,df_hr,df_ph,df_pp,df_rr,df_sbp,df_temp,df_wbc)

    df_bos = fill_time2(df_bos,timemax)
    df_hr = fill_time2(df_hr,timemax)
    df_ph = fill_time2(df_ph,timemax)
    df_pp = fill_time2(df_pp,timemax)
    df_rr = fill_time2(df_rr,timemax)
    df_sbp = fill_time2(df_sbp,timemax)
    df_temp = fill_time2(df_temp,timemax)
    df_wbc = fill_time2(df_wbc,timemax)

    df = pd.merge(df_bos,df_hr)
    df = pd.merge(df,df_ph)
    df = pd.merge(df,df_pp)
    df = pd.merge(df,df_rr)
    df = pd.merge(df,df_sbp)
    df = pd.merge(df,df_temp)
    df = pd.merge(df,df_wbc)

    df.to_csv('output/tofillvalue.csv', index=False)
    '''
    df = pd.read_csv('output/filled.csv')
    win_mean(df)
    #df.reset_index(drop=False,inplace=True)
    df.to_csv('output/mean.csv',index=False)
    #print df
    #df_hr = timebin0(df_hr, 'heart_rate')
    #fill_time2()
    # df_hr.to_csv('output/hr_timebin.csv',index=False)
    #df_hr = fill_time(df_hr)
    #df_hr.to_csv('output/hr_fulltime.csv', index=False)



if __name__ == '__main__':
    main()
