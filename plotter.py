import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self):
        pass

    @staticmethod
    def plot_actual_vs_predicted(y_actual, predictions, column_names, start_year=1880):
        """
        Plot actual vs predicted data for each column
        
        Parameters:
        - y_actual: Actual data values, numpy array.
        - predictions: Predicted data values, numpy array.
        - column_names: Names of the data columns for labeling.
        - start_year: The starting year for the actual data.
        """
        plt.figure(figsize=(10, 6))
        years_actual = np.arange(start_year, start_year + y_actual.shape[0])
        years_pred = np.arange(start_year, start_year + predictions.shape[0])
        
        # Limit to first three columns
        y_actual = y_actual[:, :3]
        predictions = predictions[:, :3]
        column_names = column_names[:3]

        for i, col_name in enumerate(column_names):
            plt.plot(years_actual, y_actual[:, i], label=f'{col_name} - Actual', linestyle='-', linewidth=2)
            plt.plot(years_pred, predictions[:, i], label=f'{col_name} - Predicted', linestyle='--', linewidth=2)
        
        plt.title('Actual vs Predicted')
        plt.xlabel('Year')
        plt.ylabel('Values')
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_loss(history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_all_predictions(y_actual, predictions, predictions_future, column_names, start_year=1880, last_year=2022):
        """
        Plot actual data, model predictions, and future predictions in one graph.

        Parameters:
        - y_actual: Actual data values, numpy array.
        - predictions: Model predictions data, numpy array.
        - predictions_future: Future predictions data, numpy array.
        - column_names: Names of the data columns for labeling.
        - start_year: The starting year for the actual data.
        - last_year: The last year of the actual data.
        """
        plt.figure(figsize=(16, 8))

        # Limit to first three columns
        y_actual = y_actual[:, :3]
        predictions = predictions[:, :3]
        predictions_future = predictions_future[:, :3]
        column_names = column_names[:3]

        years_actual = np.arange(start_year, start_year + y_actual.shape[0])
        years_pred = np.arange(start_year, start_year + predictions.shape[0])
        future_years = np.arange(last_year + 1, last_year + 1 + predictions_future.shape[0])
        print(years_actual.shape, years_pred.shape, future_years.shape)

        colors = plt.cm.tab10.colors  

        for i, col_name in enumerate(column_names):
            color = colors[i % len(colors)]
            plt.plot(years_actual, y_actual[:, i], label=f'{col_name} - Actual', linestyle='-', linewidth=2, color=color)
            plt.plot(years_pred, predictions[:, i], linestyle='--', linewidth=2, color=color)
            plt.plot(future_years, predictions_future[:, i], label=f'{col_name} - Future', linestyle=':', linewidth=2, color=color)

        plt.title('Actual vs Predicted vs Future Predictions')
        plt.xlabel('Year')
        plt.ylabel('Values')
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()
        plt.show()