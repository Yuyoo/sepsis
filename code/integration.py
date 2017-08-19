# -*- coding:utf-8 -*-

import pandas as pd
import pickle

def load(df, truelist, df_SIRS):
    sArray = df['subject_id'].unique()  # 取出subject_id
    xlist = []
    ylist = []
    for sid in sArray:
        sbp = df[df['subject_id'] == sid]['sbp']
        pp = df[df['subject_id'] == sid]['pulse pressure']
        hr = df[df['subject_id'] == sid]['heart rate']
        t = df[df['subject_id'] == sid]['temperature']
        rr = df[df['subject_id'] == sid]['respiration rate']
        wbc = df[df['subject_id'] == sid]['wbc']
        ph = df[df['subject_id'] == sid]['pH']
        bos = df[df['subject_id'] == sid]['blood oxygen saturation']
        age = df[df['subject_id'] == sid]['age']
        feature = zip(sbp, pp, hr, t, rr, wbc, ph, bos, age)
        if sid in truelist:
            feature = feature[:df_SIRS['time'][df_SIRS['subject_id'] == sid].values[0]]
        xlist.append(feature)

        if sid in truelist:
            ylist.append(1)
        else:
            ylist.append(0)

    return xlist, ylist







def dropdata(df, df1, df12):
    slistAll = list(df['subject_id'].unique())
    slist1 = list(df1['subject_id'])
    slist12 = list(df12['subject_id'])
    slist2drop = [item for item in slist1 if item not in slist12]
    slist2use = [item for item in slistAll if item not in slist2drop]
    df = df[df['subject_id'].isin(slist2use)]
    return df, slist2use, slist12


'''
# 定义样本标签
def deflabel(slist2use, slist12):
    truelist = slist12
    falselist = [item for item in slist2use if item not in slist12]
    return truelist, falselist
'''

if __name__ == '__main__':
    df = pd.read_csv('../nine_data.csv', header=None,
                     names=['subject_id', 'time', 'sbp', 'pulse pressure', 'heart rate', 'temperature',
                            'respiration rate', 'wbc', 'pH', 'blood oxygen saturation', 'age'])
    df12 = pd.read_csv('../label/gold_std_patient.txt', header=None, names=['subject_id'])
    df1 = pd.read_csv('../label/gold_std1_patient.txt', header=None, names=['subject_id'])
    df2 = pd.read_csv('../label/gold_std2_patient.txt', header=None, names=['subject_id'])
    df_SIRS = pd.read_csv('../label/std_5hsirs_time.txt', sep='\t', index_col=False, header=None, names=['subject_id', 'time'])
    df, plist, truelist = dropdata(df, df1, df12)
    xlist, ylist = load(df,truelist,df_SIRS)
    out1 = open('data.pkl', 'wb')
    out2 = open('label.pkl', 'wb')
    pickle.dump(xlist, out1)
    pickle.dump(ylist, out2)
