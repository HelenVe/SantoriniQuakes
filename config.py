# Model Hyperparameters
input_size = 1 # Magnitude is the only feature
hidden_size = 100
num_layers = 1 # Number of LSTM layers
output_size = 1 # Predicting a single magnitude value for each future timestep
look_back_steps = 24  # 6 hours of historical data (24 * 15min)
forecast_horizon = 4  # Predict next 1 hour (4 * 15min)
