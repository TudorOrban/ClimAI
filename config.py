model_config = {
    "lstm_units": [1000, 1000],
    "dropout_rates": [0.35, 0.35],
    "l2_regularization": [0.01, 0.013],
    "learning_rate": 0.001,
    "epochs": 300,
    "batch_size": 64,
    "validation_split": 0.15,
    "patience": 20,
    "lr_decay_factor": 0.9,
    "lr_decay_epoch_interval": 20,
}
