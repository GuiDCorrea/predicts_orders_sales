import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


cols_lstm = ['valores']


if df['valores'].dtype != float:
    df['valores'] = df['valores'].str.replace(',', '.').astype(float)

scaler_lstm = MinMaxScaler(feature_range=(0, 1))
df_scaled = df[cols_lstm].values
df_scaled = scaler_lstm.fit_transform(df_scaled)

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X_lstm, y_lstm = create_sequences(df_scaled, seq_length)

train_size = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]


model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

model_lstm.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=32, verbose=1)


cols_regression = ['vendaid', 'pessoaId', 'numerodepequenasemicro', 'ocupacao', 'salarios', 'valoradicionado', 'TAXACRESCIMENTO10', 'TAXACRESCIMENTOATE40', 'TAXACRESCIMENTOATE50', 'RENDAFAMILIAR40commenoresrendimentosA', 'RENDAFAMILIAR10commaioresrendimentosB', 'RENDAFAMILIAR20coMmenoresrendimentosC', 'RENDAFAMILIAR20commaioresrendimentosD', 'cidade', 'UF']

# Tratar as colunas utilizando o One-Hot Encoding
df_regression_encoded = pd.get_dummies(df[cols_regression], drop_first=True)

scaler_regression = StandardScaler()
df_scaled_regression = scaler_regression.fit_transform(df_regression_encoded)

X_regression_train, X_regression_test, y_regression_train, y_regression_test = train_test_split(df_scaled_regression, df['valores'], test_size=0.2, random_state=42)

# Treina o modelo de regressão
regressor = RandomForestRegressor()

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_regression_train, y_regression_train)

best_regressor = grid_search.best_estimator_
y_pred_regression = best_regressor.predict(X_regression_test)

# Avalia os modelos
y_pred_lstm = model_lstm.predict(X_test_lstm)
y_pred_lstm = scaler_lstm.inverse_transform(y_pred_lstm)

mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
r2_lstm = r2_score(y_test_lstm, y_pred_lstm)

mse_regression = mean_squared_error(y_regression_test, y_pred_regression)
mae_regression = mean_absolute_error(y_regression_test, y_pred_regression)
r2_regression = r2_score(y_regression_test, y_pred_regression)

# Resultados
print("Resultados LSTM:")
print("Mean Squared Error:", mse_lstm)
print("Mean Absolute Error:", mae_lstm)
print("R-squared:", r2_lstm)

print("\nResultados Regressão:")
print("Mean Squared Error:", mse_regression)
print("Mean Absolute Error:", mae_regression)
print("R-squared:", r2_regression)
