import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df_heart = pd.read_csv("./datasets/heart.csv")

    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']

    df_features = StandardScaler().fit_transform(df_features)

    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)
    
    kpca = KernelPCA(n_components=4, kernel='linear')
    kpca.fit(X_train)

    kpca_X_train = kpca.transform(X_train)
    kpca_X_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')
    logistic.fit(kpca_X_train, y_train)
    print(f'Score: {logistic.score(kpca_X_test, y_test)}')

