import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


data = pd.read_csv('seu_arquivo.csv')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('seu_arquivo.csv')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('seu_arquivo.csv')

def prepare_data(df):
   
    cols_lstm = ['valores', 'unidade', 'UF', 'regiaogeografica', 'cidade']
    cols_regression = ['valores', 'unidade', 'UF', 'regiaogeografica', 'cidade', 'status', 'quantidade', 'emissao']
    
   
    df_lstm = df[cols_lstm]
    df_regression = df[cols_regression]
    df_regression_encoded = pd.get_dummies(df_regression, drop_first=True)
    

    X_lstm = df_lstm.drop(columns=['valores'])
    y_lstm = df_lstm['valores']
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
    
    X_regression = df_regression_encoded.drop(columns=['valores'])
    y_regression = df_regression_encoded['valores']
    X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)
    
    return X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, X_train_regression, X_test_regression, y_train_regression, y_test_regression


X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, X_train_regression, X_test_regression, y_train_regression, y_test_regression = prepare_data(data)

def build_lstm_model(input_dim):
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(input_dim, 1)))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')
    return model_lstm

def build_regression_model():
    regressor = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(regressor, param_grid, cv=TimeSeriesSplit(n_splits=5), scoring=make_scorer(mean_squared_error, greater_is_better=False))
    return grid_search

def train_and_evaluate_models(X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, X_train_regression, X_test_regression, y_train_regression, y_test_regression):
    # Treina e avalia o modelo LSTM
    input_dim = X_train_lstm.shape[1]
    model_lstm = build_lstm_model(input_dim)
    model_lstm.fit(X_train_lstm.values[:, :, np.newaxis], y_train_lstm, epochs=100, batch_size=16, verbose=1)
    
    # Treina e avalia modelo de regressão
    regressor = build_regression_model()
    regressor.fit(X_train_regression, y_train_regression)
    
    return model_lstm, regressor


def suggest_new_store_locations(data, lstm_model, regressor_model):
    # SELECIONA as características relevantes para a sugestão de cidades
    features = ['numerodepequenasemicro', 'ocupacao', 'salarios', 'valoradicionado',
                'TAXACRESCIMENTO10', 'TAXACRESCIMENTOATE40', 'TAXACRESCIMENTOATE50',
                'RENDAFAMILIAR40commenoresrendimentosA', 'RENDAFAMILIAR10commaioresrendimentosB',
                'RENDAFAMILIAR20coMmenoresrendimentosC', 'RENDAFAMILIAR20commaioresrendimentosD',
                'regiaogeografica', 'bairro', 'quantidade']
    
    
    new_data = data[features]  
    
   
    lstm_predictions = lstm_model.predict(new_data.values[:, :, np.newaxis])
    regressor_predictions = regressor_model.predict(new_data)
    
   
    suggestions_df = pd.DataFrame({
        'Características': new_data.index,
        'Previsão_LSTM': lstm_predictions,
        'Previsão_Regressor': regressor_predictions
    })
    
    return suggestions_df


def main():
    
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, X_train_regression, X_test_regression, y_train_regression, y_test_regression = prepare_data(data)
    
  
    lstm_model, regressor_model = train_and_evaluate_models(X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, X_train_regression, X_test_regression, y_train_regression, y_test_regression)
    
    
    suggestions_df = suggest_new_store_locations(data, lstm_model, regressor_model)
    
    
    suggestions_dict = suggestions_df.to_dict(orient='records')
    
    return suggestions_dict

if __name__ == "__main__":
    suggestions = main()
    print(suggestions)
