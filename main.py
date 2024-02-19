from load_data import DataLoader
from model_definition import ClimateModel
from plotter import Plotter

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

# Plot actual vs predicted and loss
plotter = Plotter()
plotter.plot_actual_vs_predicted(model.actuals, model.predictions, data_loader.column_names)
plotter.plot_loss(history)

# Generate future predictions
last_window = X_test[-1]
future_steps = 30
predictions_future = model.predict_future(last_window, window_size=window_size, scaler=data_loader.scaler, future_steps=future_steps)

# Plot future predictions
plotter.plot_all_predictions(
    model.actuals,
    model.predictions,
    predictions_future,
    data_loader.column_names,
    start_year=1880,
    last_year=2017
)
