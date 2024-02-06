import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath, window_size, test_size=0.1, drop_columns=["Year"]):
        self.filepath = filepath
        self.window_size = window_size
        self.test_size = test_size
        self.drop_columns = drop_columns
        self.scaler = MinMaxScaler() 

    def load_data(self):
        # Load the dataset
        data = pd.read_csv(self.filepath)

        # Drop the columns
        if self.drop_columns:
            data = data.drop(columns=self.drop_columns)

        # Normalize the data
        data_normalized = self.scaler.fit_transform(data)

        # Create sequences
        X, y = self.create_sequences(data_normalized)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=False)

        return X_train, X_test, y_train, y_test
    
    def create_sequences(self, data):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i])
            y.append(data[i])
        X = np.array(X)
        y = np.array(y)

        # Reshape X to fit the model input shape
        X = X.reshape(X.shape[0], X.shape[1], data.shape[1])

        return X, y
