model_config = {
    "lstm_units": [400, 400],
    "dropout_rates": [0.35, 0.35],
    "l2_regularization": [0.005, 0.01],
    "learning_rate": 0.0012,
    "epochs": 300,
    "batch_size": 64,
    "validation_split": 0.15,
    "patience": 20,
    "lr_decay_factor": 0.9,
    "lr_decay_epoch_interval": 20,
}
