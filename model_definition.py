from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import numpy as np

from config import model_config


class ClimateModel:
    def __init__(self, input_shape, output_dim):
        self.model = Sequential([
            LSTM(model_config['lstm_units'][0], activation='tanh', return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(model_config['l2_regularization'][0])),
            Dropout(model_config['dropout_rates'][0]),
            LSTM(model_config['lstm_units'][1], activation='tanh', kernel_regularizer=l2(model_config['l2_regularization'][1])),
            Dropout(model_config['dropout_rates'][1]),
            Dense(output_dim)
        ])
        self.compile_model()

    def compile_model(self):
        self.model.compile(optimizer=Adam(learning_rate=model_config['learning_rate']), loss='mean_squared_error')

    def lr_schedule(self, epoch, lr):
        if epoch > 0 and epoch % model_config['lr_decay_epoch_interval'] == 0:
            return lr * model_config['lr_decay_factor']
        return lr

    def fit(self, X_train, y_train, X_test, y_test, scaler):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=model_config['patience']),
            LearningRateScheduler(self.lr_schedule)
        ]
        history = self.model.fit(
            X_train, y_train,
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            validation_split=model_config['validation_split'],
            callbacks=callbacks
        )
        self.post_training(X_train, y_train, X_test, y_test, scaler)
        return history
    
    def post_training(self, X_train, y_train, X_test, y_test, scaler):
        # Generate predictions for the training and testing sets
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)

        # Invert the predictions to original scale
        train_predictions = scaler.inverse_transform(train_predictions)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_train_actual = scaler.inverse_transform(y_train)
        y_test_actual = scaler.inverse_transform(y_test)

        # Store predictions and actuals for plotting or evaluation
        self.predictions = np.concatenate((train_predictions, test_predictions), axis=0)
        self.actuals = np.concatenate((y_train_actual, y_test_actual), axis=0)

    def predict_future(self, last_window, window_size, scaler, future_steps=30):
        predictions_future = []
        last_window_sequence = last_window.reshape(1, window_size, -1)

        for i in range(future_steps):
            # Make prediction using the last window sequence
            prediction = self.model.predict(last_window_sequence)[0]
            print(f'Prediction before scaling at step {i}: {prediction}') 
        
            # Invert the scale of the prediction before using it further
            prediction_inverted = scaler.inverse_transform(prediction.reshape(1, -1))[0]
            print(f'Prediction after scaling at step {i}: {prediction_inverted}')
        
            predictions_future.append(prediction_inverted)

            # Update the last window sequence with the new prediction
            last_window_sequence = np.append(last_window_sequence[:, 1:, :], prediction.reshape(1, 1, -1), axis=1)
            print(f'Updated last_window_sequence at step {i}: {last_window_sequence}')

        return np.array(predictions_future)

