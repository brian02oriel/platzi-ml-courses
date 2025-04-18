import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    df = pd.read_csv('./datasets/felicidad.csv')
    X = df.drop(columns=['country', 'rank', 'score'], axis=1)
    y = df['score']


    reg = RandomForestRegressor()
    params = {
        'n_estimators': range(4, 16),
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': range(2, 11)
    }

    rand_est = RandomizedSearchCV(reg, params, n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X, y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(rand_est.predict(X.loc[[0]]))
