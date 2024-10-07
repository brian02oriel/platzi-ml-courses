import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score, KFold

if __name__=="__main__":
    df = pd.read_csv('./datasets/felicidad.csv')
    X = df.drop(columns=['country', 'score'])
    y = df['score']

    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(df):
        print(train)
        print(test)