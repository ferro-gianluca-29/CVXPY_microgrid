

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
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Carica i dati
df = pd.DataFrame({
    'pv': pv_data,
    'load': load_data,
})

def normalize_data(train_data, test_data):
    scaler = MinMaxScaler()
    train_normalized = scaler.fit_transform(train_data)
    test_normalized = scaler.transform(test_data)
    return train_normalized, test_normalized, scaler

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle = False)  

train_normalized, test_normalized, scaler = normalize_data(train_df, test_df)

train_normalized_df = pd.DataFrame(train_normalized, columns=df.columns)
test_normalized_df = pd.DataFrame(test_normalized, columns=df.columns)

lookback = 168  # finestre di input
horizon = 24    # finestre di output

train_dataset = TimeseriesDataset(train_normalized_df, lookback, horizon)
test_dataset = TimeseriesDataset(test_normalized_df, lookback, horizon)

# DataLoader
batch_size_train = 16
batch_size_test = 1

loader_train = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False)
loader_test = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)


# Create the plant model
microgrid = MicroGridSystem(
                            sample_time=1,
                            self_discharge=0.95,
                            initial_state=200,
                            grid_power_min_max=[-4000.0, 4000.0])

# Create the NN and MPC optimiser
optimization_layer = OptimizationLayer(prediction_horizon=24,
                                        A_matrix=microgrid.get_model_matrices()[0],
                                        B_matrix=microgrid.get_model_matrices()[1],
                                        charge_min_max=microgrid.charge_min_max,
                                        storage_power_min_max=microgrid.storage_power_min_max,
                                        grid_power_min_max=microgrid.grid_power_min_max)

controller = Controller(optimization_layer, scaler)



num_epochs = 100

optimizer = torch.optim.Adam(controller.parameters(), lr=0.01, weight_decay=0.0)

# Imposta il modello in modalità training
controller.train()

# Crea un'istanza della tua classe loss personalizzata
mse_loss = MSELoss()

total_cost_loss = TotalCostLoss()

tbar = tqdm(range(num_epochs))

epoch_losses = []  # Lista per conservare la loss media di ogni epoca

for epoch in tbar:

    total_loss = 0  # Inizializza l'accumulatore di loss per l'epoca
    num_batches = 0  # Contatore per il numero di batch, per calcolare la media
    
    # Loop per alimentare il modello
    for inputs, targets in loader_train:

        #🎈🎈🎈🎈🎈🎈🎈 FORWARD PASS 🎈🎈🎈🎈🎈🎈🎈🎈

        # Chiama (implicitamente) il metodo `forward` sull'istanza del controller
        predictions, (Ps_in, Ps_out, Pl, Pr, eps, s) = controller(inputs, controller.current_state[:, -1])

        #🎈🎈🎈🎈🎈🎈🎈 CALCOLO DELLA LOSS 🎈🎈🎈🎈🎈🎈🎈

        regression_loss_training = mse_loss(predictions, targets)
        #task_loss_training = total_cost_loss()
        gradient_loss = regression_loss_training

        #🎈🎈🎈🎈🎈🎈🎈 BACKWARD PASS 🎈🎈🎈🎈🎈🎈🎈

        optimizer.zero_grad()  # Azzera i gradienti accumulati
        gradient_loss.backward() # Backpropagation per calcolare i gradienti
        optimizer.step()  # Aggiorna i pesi del modello

        #print("Loss:", gradient_loss.item())

        # Aggiorna lo stato con il nuovo stato calcolato dalla rete
        #controller.update_state(s)

        # Aggiorna il total loss e il conteggio dei batch
        total_loss += gradient_loss.item()
        num_batches += 1

    # Calcola la loss media per l'epoca e aggiorna la barra di progresso
    average_loss = total_loss / num_batches
    epoch_losses.append(average_loss)  # Aggiungi la loss media alla lista
    tbar.set_description(f"Epoch {epoch + 1}/{num_epochs} Average Loss: {average_loss:.4f}")


plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.show()


## 🎈🎈🎈🎈🎈🎈 TEST 🎈🎈🎈🎈🎈🎈🎈

# Assicurati che il modello sia in modalità evaluation
controller.eval()

# Prendiamo solo il primo batch per il test
inputs, targets = next(iter(loader_test))

# Disabilitiamo il calcolo dei gradienti
with torch.no_grad():
    # Effettua la predizione
    predictions, _ = controller(inputs, controller.current_state[:, -1])

# Denormalizza le previsioni di pv e load
pv_predictions, load_predictions = controller.denormalize(predictions)

targets = targets.reshape(predictions.shape)
                            
pv_targets, load_targets = controller.denormalize(targets)

# Plotting per "pv"
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)  # Primo subplot per "pv"
plt.plot(pv_targets.cpu().numpy().flatten(), label='Actual PV', linestyle='-', color='blue')
plt.plot(pv_predictions.cpu().numpy().flatten(), label='Predicted PV', linestyle='-', color='red')
plt.title('PV Predictions vs Actual')
plt.xlabel('Timestep Index')
plt.ylabel('PV Value')
plt.legend()
plt.grid(True)

# Plotting per "load"
plt.subplot(1, 2, 2)  # Secondo subplot per "load"
plt.plot(load_targets.cpu().numpy().flatten(), label='Actual Load', linestyle='-', color='blue')
plt.plot(load_predictions.cpu().numpy().flatten(), label='Predicted Load', linestyle='-', color='red')
plt.title('Load Predictions vs Actual')
plt.xlabel('Timestep Index')
plt.ylabel('Load Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


