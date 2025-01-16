from typing import Tuple, Dict, Any
import torch
from microgrid_controller import OptimizationLayer, Controller
import pandas as pd
import numpy as np

from microgrid import MicroGridSystem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset

from loss import MSELoss, TotalCostLoss

import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import logging
logging = logging.getLogger('pytorch')

# Imposta il device per la GPU
device = torch.device("cuda")
logging.info(f"Using device: {device}")

VARIABLES = ['Ps', 'Pl', 'Pr', 'Pg', 'eps']

# Leggi i dati dai file CSV
pv_data = pd.read_csv('./dataset.csv', usecols=[1]).squeeze()
load_data = pd.read_csv('./dataset.csv', usecols=[2]).squeeze()

class TimeseriesDataset(Dataset):
    """ Dataset per serie temporali con finestre di look-back e prediction. """
    def __init__(self, df, lookback, horizon):
        self.data = df
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx+self.lookback].values
        y = self.data.iloc[idx+self.lookback:idx+self.lookback+self.horizon].values
        return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)

# Carica e normalizza i dati
df = pd.DataFrame({'pv': pv_data, 'load': load_data})
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
scaler = MinMaxScaler()
train_normalized = scaler.fit_transform(train_df)
test_normalized = scaler.transform(test_df)
train_normalized_df = pd.DataFrame(train_normalized, columns=df.columns)
test_normalized_df = pd.DataFrame(test_normalized, columns=df.columns)

train_dataset = TimeseriesDataset(train_normalized_df, lookback=168, horizon=24)
test_dataset = TimeseriesDataset(test_normalized_df, lookback=168, horizon=24)

loader_train = DataLoader(train_dataset, batch_size=16, shuffle=False)
loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create the MicroGridSystem and Controller
microgrid = MicroGridSystem(sample_time=1, self_discharge=0.95, initial_state=200, grid_power_min_max=[-4000.0, 4000.0])
optimization_layer = OptimizationLayer(prediction_horizon=24,
                                       A_matrix=microgrid.get_model_matrices()[0].to(device),
                                       B_matrix=microgrid.get_model_matrices()[1].to(device),
                                       charge_min_max=microgrid.charge_min_max,
                                       storage_power_min_max=microgrid.storage_power_min_max,
                                       grid_power_min_max=microgrid.grid_power_min_max)
controller = Controller(optimization_layer, scaler, device=device).to(device)

# Training setup
optimizer = torch.optim.Adam(controller.parameters(), lr=0.01, weight_decay=0.0)
mse_loss = MSELoss().to(device)
total_cost_loss = TotalCostLoss().to(device)
epoch_losses = []

# Training loop
for epoch in tqdm(range(3), desc="Training Epoch"):
    total_loss = 0
    for inputs, targets in loader_train:
        optimizer.zero_grad()
        predictions, (Ps_in, Ps_out, Pl, Pr, eps, s), Pg_hat = controller(inputs, controller.current_state[:, -1])
        real_Pg = controller.Pg_with_real_values(targets)
        loss = mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(loader_train)
    epoch_losses.append(average_loss)
    logging.info(f'Epoch {epoch + 1}: Average Loss {average_loss:.4f}')

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.show()

# Evaluation on test data
controller.eval()
inputs, targets = next(iter(loader_test))
with torch.no_grad():
    predictions, _, _ = controller(inputs, controller.current_state[:, -1])
    pv_predictions, load_predictions = controller.denormalize(predictions)
    pv_targets, load_targets = controller.denormalize(targets.view(predictions.shape))

mse = torch.nn.functional.mse_loss(pv_predictions, pv_targets)
logging.info(f"MSE between PV predictions and PV targets: {mse.item()}")

# Plotting results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(pv_targets.cpu().numpy().flatten(), label='Actual PV', linestyle='-', color='blue')
plt.plot(pv_predictions.cpu().numpy().flatten(), label='Predicted PV', linestyle='-', color='red')
plt.title('PV Predictions vs Actual')
plt.xlabel('Timestep Index')
plt.ylabel('PV Value')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(load_targets.cpu().numpy().flatten(), label='Actual Load', linestyle='-', color='blue')
plt.plot(load_predictions.cpu().numpy().flatten(), label='Predicted Load', linestyle='-', color='red')
plt.title('Load Predictions vs Actual')
plt.xlabel('Timestep Index')
plt.ylabel('Load Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
