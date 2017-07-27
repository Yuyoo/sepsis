#coding=utf-8
import pandas as pd



def plotTime(df):
    df = df[['subject_id','intime_hours','heart_rate']]
    patientArray = df['subject_id'].unique()
    p0 = df[['intime_hours', 'heart_rate']][df['subject_id'] == patientArray[0]]
    #p1 = df[['intime_hours', 'heart_rate']][df['subject_id'] == patientArray[1]]
    #p2 = df[['intime_hours', 'heart_rate']][df['subject_id'] == patientArray[2]]
    #p3 = df[['intime_hours', 'heart_rate']][df['subject_id'] == patientArray[3]]
    #p4 = df[['intime_hours', 'heart_rate']][df['subject_id'] == patientArray[4]]
    p0.plot(x = 'intime_hours',y='heart_rate',kind = 'scatter')
    print 'success'


def main():
    df = pd.read_csv('entericudata/heart_rate.csv')
    plotTime(df)

if __name__ == '__main__':
    main()