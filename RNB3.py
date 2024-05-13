import yfinance as yf
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

ids_df = pd.read_excel("C:/users/brand/onedrive/escritorio/ID empresas.xlsx")  # Ajusta el nombre del archivo según sea necesario

# Definir la ventana de tiempo para predecir el siguiente precio
window_size = 30

# Crear una lista para almacenar los resultados
results = []

# Iterar sobre los diferentes tickers
for ticker in ids_df['Ids']:
    try:
        
        def create_dataset(data, window_size):
             X, y = [], []
             for i in range(len(data) - window_size):
                 X.append(data.iloc[i:(i + window_size), 0].values)  # Tomamos solo la columna 'Adj Close'
                 y.append(data.iloc[i + window_size, 0])  # Obtenemos el siguiente precio
             return X, y
        # Descargar datos de Yahoo Finance para el ticker actual
        data = yf.download(ticker, start="2023-03-30", end="2024-03-30")
        
        # Crear conjuntos de datos para entrenamiento y prueba
        X, y = create_dataset(data[['Adj Close']], window_size)
        
        # Convertir a DataFrame de pandas
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Ajustar la forma de los datos de entrada
        X_train = tf.convert_to_tensor(X_train.values)
        X_test = tf.convert_to_tensor(X_test.values)
        
        # Definir el modelo de la red neuronal
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(window_size,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        
        # Compilar el modelo
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Entrenar el modelo durante 50 épocas
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Realizar predicciones en los últimos 15 días
        last_fifteen_days_data = data.iloc[-(window_size + 15):][['Adj Close']]
        X_last_fifteen_days, y_last_fifteen_days = create_dataset(last_fifteen_days_data, window_size)
        
        # Convertir a DataFrame de pandas
        X_last_fifteen_days = pd.DataFrame(X_last_fifteen_days)
        
        # Ajustar la forma de los datos de entrada
        X_last_fifteen_days = tf.convert_to_tensor(X_last_fifteen_days.values)
        
        # Realizar predicciones
        predictions_last_fifteen_days = model.predict(X_last_fifteen_days)
        
        # Calcular los errores absolutos entre los valores reales y las predicciones
        errors = abs(y_last_fifteen_days - predictions_last_fifteen_days.flatten())
        
        # Calcular los errores absolutos porcentuales
        absolute_percentage_errors = (errors / y_last_fifteen_days) * 100
        
        # Calcular el error absoluto porcentual medio (MAPE)
        mape = absolute_percentage_errors.mean()
        
        # Guardar los resultados en la lista
        results.append({'ID': ticker, 'MAPE': mape})
        
        # Imprimir el ID y el MAPE
        print("ID:", ticker, "MAPE:", mape)
    
    except Exception as e:
        print("Error al procesar el ticker", ticker, ":", str(e))
        continue

# Crear un DataFrame a partir de los resultados
results_df = pd.DataFrame(results)

# Guardar los resultados en un archivo CSV
results_df.to_csv("resultados.csv", index=False)

# Imprimir los resultados
print(results_df)