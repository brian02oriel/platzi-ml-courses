import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df_heart = pd.read_csv("./datasets/heart.csv")

    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']

    df_features = StandardScaler().fit_transform(df_features)

    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    pca = PCA(n_components=3)
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    # X = n_components | Y = variance
    #plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.show()

    logistic = LogisticRegression(solver='lbfgs')

    pca_X_train = pca.transform(X_train)
    pca_X_test = pca.transform(X_test)

    logistic.fit(pca_X_train, y_train)
    print(f"Score PCA: {logistic.score(pca_X_test, y_test)}")

    ipca_X_train = pca.transform(X_train)
    ipca_X_test = pca.transform(X_test)

    logistic.fit(ipca_X_train, y_train)
    print(f"Score IPCA: {logistic.score(ipca_X_test, y_test)}")

