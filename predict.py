import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def prepare_data(df, forecast_col, forecast_out, test_size):
    label = df[forecast_col].shift(-forecast_out)  # criando nova coluna chamada label com as ultimas 5 linhas sao NaN
    X = np.array(df[[forecast_col]])  # criando o array de recursos
    X = preprocessing.scale(X)  # processando o array de recursos
    X_lately = X[-forecast_out:]  # criando a coluna que eu quero usar posteriormente no método de previsao
    X = X[:-forecast_out]  # X que irá conter o treinamento e o teste
    label.dropna(inplace=True)  # eliminando valores NaN
    y = np.array(label)  # atribuindo Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0)  # validação cruzada

    response = [X_train, X_test, Y_train, Y_test, X_lately]

    return response


df = pd.read_csv("precos.csv")

#df = df[df.symbol == "GOOG"]

forecast_col = 'close'
forecast_out = 5
test_size = 0.2

X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size)
learner = LinearRegression()  # inicializando o modelo de regressão linear

learner.fit(X_train, Y_train)  # treinamento do modelo de regressão linear

score = learner.score(X_test, Y_test)  # testando o modelo de regressão linear
forecast = learner.predict(X_lately)  # Conjunto que conterá os dados de previsão
response = {}  # criando o objeto json
response['test_score'] = score
response['forecast_set'] = forecast

print(response)
