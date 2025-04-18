import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    df_heart = pd.read_csv('./datasets/heart.csv')
    X = df_heart.drop(columns=['target'], axis=1)
    y = df_heart['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    knn_model = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    print("="*64)
    print(accuracy_score(knn_pred, y_test))

    bag_model = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_pred = bag_model.predict(X_test)
    print("="*64)
    print(accuracy_score(bag_pred, y_test))

    boost_model = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    boost_pred = boost_model.predict(X_test)
    print("="*64)
    print(accuracy_score(boost_pred, y_test))
