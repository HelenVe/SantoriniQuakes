import os
import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from config import input_size, hidden_size, num_layers, output_size, look_back_steps, forecast_horizon

from model import Seq2SeqAutoencoder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

current_directory = os.path.dirname(os.getcwd())

df = pd.read_csv(os.path.join(current_directory, 'Data\catalogue.csv'))
df['Time'] = pd.to_datetime(df['Time'])
df_resampled = df.set_index('Time')
mean_magnitudes_15min = df_resampled['Magnitude'].resample('15min').mean().ffill()# forward fills
mean_magnitudes_filled = mean_magnitudes_15min.rolling(window=4, min_periods=1).mean().fillna(0) # take last 4 measurements

data = np.array(mean_magnitudes_filled).reshape(-1, 1)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
joblib.dump(scaler, 'minmax_scaler.pkl')

def create_sequences(data, look_back_steps, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - look_back_steps - forecast_horizon + 1):
        seq_in = data[i:(i + look_back_steps)] # shape (look_back_steps, 1)
        seq_out = data[(i + look_back_steps):(i + look_back_steps + forecast_horizon)] # shape (forecast_horizon, 1)
        X.append(seq_in)
        y.append(seq_out)
    return np.array(X), np.array(y)

X, y = create_sequences(data, look_back_steps, forecast_horizon)
train_size = int(len(X) * 0.8) # 80% for training, 20% for testing
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

print(f"\nTrain set shapes: X_train_np {X_train.shape}, y_train_np {y_train.shape}")
print(f"Test set shapes: X_test_np {X_test.shape}, y_test_np {y_test.shape}")

# convert to Pytorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

torch.save(X_train, os.path.join(current_directory, 'Data\X_train.pt'))
torch.save(y_train, os.path.join(current_directory, 'Data\y_train.pt'))
torch.save(X_test, os.path.join(current_directory, 'Data\X_test.pt'))
torch.save(y_test, os.path.join(current_directory, 'Data\y_test.pt'))

batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = Seq2SeqAutoencoder(input_size, hidden_size, num_layers, output_size, look_back_steps, forecast_horizon).to(device)
model_save_path = os.path.join(os.getcwd(), 'model.pt')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"\nModel architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

num_epochs = 30
print(f"\nTraining the model on {device}...")
train_losses = []
val_losses = []

# ----------- Training ---------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_save_path)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # --- Validation Loop ---
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_running_loss += loss.item() * inputs.size(0)

    epoch_val_loss = val_running_loss / len(test_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}')

print("\nTraining complete.")

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss During Training (PyTorch)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

os.makedirs(os.path.join(current_directory, 'Data\TrainingPlots'), exist_ok=True)
plt.savefig(os.path.join(current_directory, r'Data\TrainingPlots\train_loss.png'))