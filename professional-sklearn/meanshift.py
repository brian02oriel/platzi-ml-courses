import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == '__main__':
    df = pd.read_csv("./datasets/candy.csv")
    X = df.drop(columns=['competitorname'], axis=1)

    meanshift = MeanShift().fit(X)

    df['group'] = meanshift.labels_
    print(df.head(5))