import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./datasets/felicidad.csv')
    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model_linear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = model_linear.predict(X_test)

    model_lasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = model_lasso.predict(X_test)

    model_ridge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = model_ridge.predict(X_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print(f'Linear loss: {linear_loss}')

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print(f'Lasso loss: {lasso_loss}')

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print(f'Ridge loss: {ridge_loss}')

    print("=="*32)
    print(f'Coef Lasso: {model_lasso.coef_}')

    print("=="*32)
    print(f'Coef ridge: {model_ridge.coef_}')