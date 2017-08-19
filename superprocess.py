# coding = utf-8
import pandas as pd
import numpy as np


def main():
    mat1 = np.loadtxt('nine_items.txt')
    mat2 = np.loadtxt('forward_sirs.txt')
    df1 = pd.DataFrame(mat1, columns=['subject_id', 'time', 'sbp', 'pulse pressure', 'heart rate', 'temperature',
                                      'respiration rate', 'wbc', 'pH', 'blood oxygen saturation', 'age'])
    df2 = pd.DataFrame(mat2, columns=['subject_id', 'time', 'sirs1', 'sirs2', 'sirs3', 'sirs4', 'score', 'singlesirs',
                                      'sirs'])
    df2 = df2[['subject_id', 'time', 'sirs']]
    df1 = df1.merge(df2, on=['subject_id', 'time'])
    print df1


if __name__ == '__main__':
    main()
