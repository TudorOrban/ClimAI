from load_data import DataLoader
from model_definition import ClimateModel

# Configuration
filepath = 'ZonAnn.Ts+dSST.csv'
window_size = 5

# Initialize DataLoader and load data
data_loader = DataLoader(filepath, window_size)
X_train, X_test, y_train, y_test = data_loader.load_data()
scaler = data_loader.scaler

# Initialize and train model
input_shape = (X_train.shape[1], X_train.shape[2])
output_dim = y_train.shape[1]
model = ClimateModel(input_shape, output_dim)
history = model.fit(X_train, y_train, X_test, y_test, scaler)

# Log history
print(history.history)