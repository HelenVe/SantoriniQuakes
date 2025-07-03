import joblib
import numpy as np
import os
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader
from model import Seq2SeqAutoencoder
from config import input_size, hidden_size, num_layers, output_size, look_back_steps, forecast_horizon

current_directory = os.path.dirname(os.getcwd())

X_test = torch.load(os.path.join(current_directory, 'Data/Χ_test.pt'))
y_test = torch.load(os.path.join(current_directory, 'Data/y_test.pt'))
scaler = joblib.load('minmax_scaler.pkl')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


# Load Model
model = Seq2SeqAutoencoder(input_size, hidden_size, num_layers, output_size, look_back_steps, forecast_horizon)
model.load_state_dict(torch.load('model.pt'))
model.to(device)
model.eval()
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_size = 0.8

print("\nEvaluating the model on test data...")
model.eval()  # Set model to evaluation mode
test_predictions_scaled = []
test_actuals_scaled = []

with torch.no_grad():  # Disable gradient calculation
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        test_predictions_scaled.append(outputs.cpu().numpy())  # Move to CPU and convert to np
        test_actuals_scaled.append(targets.cpu().numpy())

# Concatenate all batches
test_predictions_scaled = np.concatenate(test_predictions_scaled, axis=0)
test_actuals_scaled = np.concatenate(test_actuals_scaled, axis=0)

# Reshape for inverse_transform
y_test_original = scaler.inverse_transform(test_actuals_scaled.reshape(-1, forecast_horizon))
y_pred_original = scaler.inverse_transform(test_predictions_scaled.reshape(-1, forecast_horizon))

# Calculate metrics for overall performance (flatten all predictions and actuals)
rmse_overall = np.sqrt(mean_squared_error(y_test_original.flatten(), y_pred_original.flatten()))
mae_overall = mean_absolute_error(y_test_original.flatten(), y_pred_original.flatten())

print(f"\nOverall Test RMSE: {rmse_overall:.4f}")
print(f"Overall Test MAE: {mae_overall:.4f}")

# Calculate metrics for each step in the forecast horizon
print("\nMetrics for each forecast step:")
for i in range(forecast_horizon):
    step_rmse = np.sqrt(mean_squared_error(y_test_original[:, i], y_pred_original[:, i]))
    step_mae = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])
    print(f"  Step {i + 1} (15min ahead): RMSE={step_rmse:.4f}, MAE={step_mae:.4f}")
