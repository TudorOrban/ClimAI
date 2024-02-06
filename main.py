from load_data import DataLoader

# Configuration
filepath = 'ZonAnn.Ts+dSST.csv'
window_size = 5

# Initialize DataLoader and load data
data_loader = DataLoader(filepath, window_size)
X_train, X_test, y_train, y_test = data_loader.load_data()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)