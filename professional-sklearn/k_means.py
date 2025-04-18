import pandas as pd

from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('./datasets/candy.csv')
    X = df.drop(columns=['competitorname'], axis=1)
    
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print(len(kmeans.cluster_centers_))
    print('='*64)
    print(kmeans.predict(X))

    df['group'] = kmeans.predict(X)
    print(df)