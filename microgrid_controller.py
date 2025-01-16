import torch
import torch.nn as nn
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpylayers.torch import CvxpyLayer
from typing import Tuple

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class OptimizationLayer(nn.Module):
    def __init__(self, prediction_horizon: int,
                 A_matrix: torch.Tensor,
                 B_matrix: torch.Tensor,
                 charge_min_max: Tuple[float, float],
                 storage_power_min_max: Tuple[float, float],
                 grid_power_min_max: Tuple[float, float]):
        super(OptimizationLayer, self).__init__()
        self.device = torch.device('cuda')  # Imposta il dispositivo CUDA
        A_matrix = A_matrix.to(self.device)
        B_matrix = B_matrix.to(self.device)
        
        # Verifica che i limiti min siano inferiori ai limiti max
        assert charge_min_max[0] < charge_min_max[1], "The minimum charge must be smaller than the maximum charge."
        assert storage_power_min_max[0] < storage_power_min_max[1], "The minimum storage power must be smaller than the maximum storage power."
        assert grid_power_min_max[0] < grid_power_min_max[1], "The minimum grid power must be smaller than the maximum grid power."

        self._prediction_horizon = prediction_horizon

        # Definisci variabili e parametri
        s = Variable((1, prediction_horizon + 1))
        Ps_in = Variable((1, prediction_horizon))
        Ps_out = Variable((1, prediction_horizon))
        Pl = Variable((1, prediction_horizon))
        Pr = Variable((1, prediction_horizon))
        eps = Variable((1, prediction_horizon))
        
        price = Parameter((1, prediction_horizon))
        load = Parameter((1, prediction_horizon))
        pv = Parameter((1, prediction_horizon))
        s0 = Parameter((1, 1))

        # Definisci funzione obiettivo e vincoli
        objective = 0.0
        constraints = [s[:, 0] == s0]
        for k in range(prediction_horizon):
            objective += -1 * price[0, k] * (Pr[:, k] - Pl[:, k] - (Ps_in[:, k] - Ps_out[:, k])) + 10000 * eps[:, k]
            constraints += [
                s[:, k + 1] == A_matrix @ s[:, k] + B_matrix[0, 0] * Ps_in[:, k] + B_matrix[0, 1] * Ps_out[:, k],
                s[:, k] >= charge_min_max[0],
                s[:, k] <= charge_min_max[1],
                Ps_in[:, k] >= 0,
                Ps_in[:, k] <= storage_power_min_max[1],
                Ps_out[:, k] >= 0,
                Ps_out[:, k] <= storage_power_min_max[1],
                Pl[:, k] >= load[0, k] - eps[:, k],
                Pl[:, k] <= load[0, k] + eps[:, k],
                Pr[:, k] >= pv[0, k] - eps[:, k],
                Pr[:, k] <= pv[0, k] + eps[:, k],
                Pr[:, k] - Pl[:, k] - Ps_in[:, k] + Ps_out[:, k] >= grid_power_min_max[0],
                Pr[:, k] - Pl[:, k] - Ps_in[:, k] + Ps_out[:, k] <= grid_power_min_max[1]
            ]

        problem = Problem(Minimize(objective), constraints)
        self.optimization_layer = CvxpyLayer(problem, parameters=[price, load, pv, s0], variables=[s, Ps_in, Ps_out, Pl, Pr, eps])

    def forward(self, price_tariff: torch.Tensor, load_hat: torch.Tensor, pv_hat: torch.Tensor, initial_state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size = load_hat.shape[0]
        device = self.device  # Utilizza il dispositivo GPU predefinito

        # Prepara tensori per le variabili di output su GPU
        s = torch.zeros((batch_size, self._prediction_horizon + 1), device=device, dtype=torch.float64)
        Ps_in = torch.zeros((batch_size, self._prediction_horizon), device=device, dtype=torch.float64)
        Ps_out = torch.zeros((batch_size, self._prediction_horizon), device=device, dtype=torch.float64)
        Pl = torch.zeros((batch_size, self._prediction_horizon), device=device, dtype=torch.float64)
        Pr = torch.zeros((batch_size, self._prediction_horizon), device=device, dtype=torch.float64)
        eps = torch.zeros((batch_size, self._prediction_horizon), device=device, dtype=torch.float64)

        # Calcola soluzioni per ogni input nel batch
        for i in range(batch_size):
            s[i, :], Ps_in[i, :], Ps_out[i, :], Pl[i, :], Pr[i, :], eps[i, :] = self.optimization_layer(
                price_tariff[i:i+1, :], load_hat[i:i+1, :], pv_hat[i:i+1, :], initial_state[i:i+1, 0:1]
            )

        return Ps_in, Ps_out, Pl, Pr, eps, s



class Controller(nn.Module):
    """This class is used to define the controller for the system.

    The controller is a neural network that estimates the parameters of the optimization problem
    and uses the predicted parameters to solve the optimization problem.

    Args:
        optimization_layer (OptimizationLayer): The optimization layer.
        dataset: The dataset to be used.
    """

    def __init__(self,
                 optimization_layer: OptimizationLayer, scaler: MinMaxScaler,
                 ) -> None:
        super(Controller, self).__init__()


        #🚫🚫🚫🚫🚫🚫   AGGIORNA IL CODICE IN MODO DA DARLI IN INPUT SENZA RIDEFINIRLI QUA!🚫🚫🚫🚫🚫

        self._lookback_window = 168
        self._prediction_horizon = 24
        self._batch_size_train = 16

        ### 🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫🚫


        storage_data = pd.read_csv('./storage_data.csv')
        self.efficiency = storage_data['charge_discharge_efficiency'][0]

        self.charge_min_max = [storage_data['capacity_min_kWh'][0].item(),
                               storage_data['capacity_max_kWh'][0].item()]
        self.storage_power_min_max = [storage_data['power_min_kW'][0].item(),
                                      storage_data['power_max_kW'][0].item()]
        self.grid_power_min_max = [-4000.0, 4000.0]
        assert self.charge_min_max[0] <= self.charge_min_max[
            1], f"`charge_min_max` must be in [charge_min, charge_max], Got {self.charge_min_max}."
        assert self.storage_power_min_max[0] <= self.storage_power_min_max[
            1], f"`storage_power_min_max` must be in [power_min, power_max], Got {self.storage_power_min_max}."
        assert self.grid_power_min_max[0] <= self.grid_power_min_max[
            1], f"`grid_power_min_max` must be in [power_min, power_max], Got {self.grid_power_min_max}."

        self._self_discharge = 0.0042
        self._sample_time = 1

        self.initial_state = torch.tensor([200], dtype=torch.float64).reshape(-1, 1)

        self.current_state = self.initial_state[0] * torch.ones(self._batch_size_train, self._lookback_window,
                                                          self.get_state_dimension(), device='cpu', dtype=torch.float64)

        self._hidden_dim = 64
        self._num_layers = 1

        self._input_dim = 2
        self._output_dim = 2

        self.scaler = scaler

        #####  🎈🎈🎈🎈🎈 RETE LSTM PER IL FORECASTING 🎈🎈🎈🎈🎈🎈🎈

        self._lstm = nn.LSTM(self._input_dim, self._hidden_dim,
                             self._num_layers, batch_first=True, bias=True)

        self._fc = nn.Linear(
            self._hidden_dim, self._output_dim*self._prediction_horizon)
        
        ########### 🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈 ################

        self.optimization_layer = optimization_layer


    def _create_price_tariff(self):
        price = 40
        return price
    
    def denormalize(self, predictions):

        """ Denormalize the predictions for load and pv output """

        # Cambia la forma dei dati per l'applicazione dello scaler
        data_shape = predictions.shape
        predictions_reshaped = predictions.view(-1, 2)
        predictions_np = predictions_reshaped.cpu().detach().numpy()
        
        # Applica lo scaler per denormalizzare
        denorm_predictions_np = self.scaler.inverse_transform(predictions_np)
        
        # Converti di nuovo in tensori e rimetti nella forma originale
        denorm_predictions = torch.tensor(denorm_predictions_np, device=predictions.device)
        denorm_predictions = denorm_predictions.view(data_shape)
        
        # Separa pv_hat e load_hat
        pv_hat = denorm_predictions[:, :, 0]
        load_hat = denorm_predictions[:, :, 1]
        
        return pv_hat, load_hat
        

    def forward(self,
                inputs: torch.Tensor,
                initial_state: torch.Tensor
                ) -> Tuple[torch.Tensor,
                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of the controller.

        It returns the predictions for the features with respect to the `prediction_horizon`. These
        are the inputs of the optimization problem. The output then also contains the optimal solution
        of the optimization problem.
        """

        inputs = inputs.to(torch.float32)
        predictions, _ = self._lstm(inputs)
        predictions = self._fc(predictions[:, -1, :])
        predictions = predictions.to(torch.float64)
        predictions = predictions.reshape(predictions.shape[0],
                                          self._prediction_horizon,
                                          self._output_dim)
        
        #pv_hat = predictions[:, :, 0]  # Prendi la prima colonna per pv
        #load_hat = predictions[:, :, 1]  # Prendi la seconda colonna per load

        predictions = predictions.clone().detach().double().requires_grad_()

        pv_hat, load_hat = self.denormalize(predictions)
        pv_hat = pv_hat.clone().detach().double().requires_grad_()
        load_hat = load_hat.clone().detach().double().requires_grad_()

        self.price_tariff = torch.full_like(
            pv_hat, fill_value=self._create_price_tariff(),
            dtype=torch.float64, requires_grad=True
        )

        self.initial_state = initial_state.clone().detach().double().requires_grad_()

        Ps_in, Ps_out, Pl, Pr, eps, s = self.optimization_layer(
            self.price_tariff, load_hat, pv_hat, self.initial_state
        )
        Pg_hat = self.compute_Pg(Ps_in, Ps_out, Pr, Pl)

        return predictions, (Ps_in, Ps_out, Pl, Pr, eps, s), Pg_hat
    
    def update_state(self, s):
        """
        Aggiorna lo stato corrente con l'ultimo stato di carica della batteria.

        Args:
            s (torch.Tensor): Tensor contenente gli stati di carica della batteria per ogni timestep.
        """
        # Assumiamo che s abbia la forma [batch_size, prediction_horizon + 1]
        # e che vogliamo conservare l'ultimo stato di carica per ogni batch.
        self.current_state = s[:, -1].unsqueeze(1).float()  # Conserva l'ultimo stato di carica e lo adatta per il prossimo batch

    def get_state_dimension(self) -> int:
        """Return the state dimension of the system."""
        A_matrix = self.get_plant_matrices()[0]
        return A_matrix.shape[0]
    
    def get_plant_matrices(self, device: torch.device = torch.device("cpu")
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is used to compute the matrices of the real plant.

        This is the function that the user has to implement.
        """
        A_matrix = (1 - self._self_discharge) * \
            torch.ones((1, 1), dtype=torch.float64, device=device)
        
        b_in = self._sample_time * self.efficiency * torch.ones((1, 1), dtype=torch.float64, device=device)
        b_out = -1*self._sample_time * (1/self.efficiency) * torch.ones((1, 1), dtype=torch.float64, device=device)
        B_matrix = torch.cat((b_in, b_out),1)
        return A_matrix, B_matrix
    
    def compute_Pg(self, Ps_in: torch.Tensor,
                   Ps_out: torch.Tensor,
                   Pr: torch.Tensor,
                   Pl: torch.Tensor) -> torch.Tensor:
        """Compute the power exchanged with the utility grid as:
            `Pg = Pr - Ps_in + Ps_out - Pl`

        Args:
            Ps_in (torch.Tensor): Power input to the storage system.
            Ps_out (torch.Tensor): Power output to the storage system.
            Pr (torch.Tensor): Power exchanged with the renewable generator.
            Pl (torch.Tensor): Power exchanged with the load.
        """
        return Pr - Ps_in + Ps_out - Pl
    
    def Pg_with_real_values(self, targets):

        targets = targets.reshape(targets.shape[0],
                                          self._prediction_horizon,
                                          self._output_dim)
        
        real_pv, real_load = self.denormalize(targets)

        Ps_in, Ps_out, Pl, Pr, eps, s = self.optimization_layer(
            self.price_tariff, real_load, real_pv, self.initial_state)
        
        real_Pg = self.compute_Pg(Ps_in, Ps_out, Pr, Pl)

        return real_Pg
        
class Controller(nn.Module):
    """This class defines the controller for the system.
    
    Args:
        optimization_layer (OptimizationLayer): The optimization layer.
        scaler (MinMaxScaler): Scaler for normalizing/denormalizing data.
        initial_state (torch.Tensor): Initial state of the storage.
        lookback_window (int): Number of past observations to consider.
        prediction_horizon (int): Number of future steps to predict.
        batch_size_train (int): Batch size for training.
        device (torch.device): Device to run the model computations.
    """
    def __init__(self, optimization_layer, scaler, initial_state, lookback_window, prediction_horizon, batch_size_train, device):
        super(Controller, self).__init__()
        self.optimization_layer = optimization_layer.to(device)
        self.scaler = scaler
        self.initial_state = initial_state.to(device)
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.batch_size_train = batch_size_train
        self.device = device
        
        # LSTM Network settings
        self._hidden_dim = 64
        self._num_layers = 1
        self._input_dim = 2  # Assuming two features per input
        self._output_dim = 2  # Assuming two outputs

        # LSTM and Fully Connected layer
        self._lstm = nn.LSTM(self._input_dim, self._hidden_dim, self._num_layers, batch_first=True).to(device)
        self._fc = nn.Linear(self._hidden_dim, self._output_dim * self.prediction_horizon).to(device)

    def _create_price_tariff(self):
        """Create a fixed price tariff."""
        return 40  # Fixed price for simplicity

    def denormalize(self, predictions):
        """Denormalize the predictions for load and PV output using the stored scaler."""
        data_shape = predictions.shape
        predictions_reshaped = predictions.view(-1, 2).cpu().numpy()  # Move data to CPU for scaler
        denorm_predictions_np = self.scaler.inverse_transform(predictions_reshaped)
        denorm_predictions = torch.tensor(denorm_predictions_np, device=self.device).view(data_shape)
        pv_hat, load_hat = denorm_predictions[:, :, 0], denorm_predictions[:, :, 1]
        return pv_hat, load_hat

    def forward(self, inputs, initial_state):
        """Forward pass of the controller."""
        inputs = inputs.to(self.device).float()
        initial_state = initial_state.to(self.device).double()
        predictions, _ = self._lstm(inputs)
        predictions = self._fc(predictions[:, -1, :])
        predictions = predictions.to(torch.float64)
        predictions = predictions.reshape(predictions.shape[0], self.prediction_horizon, self._output_dim)
        pv_hat, load_hat = self.denormalize(predictions)
        price_tariff = torch.full_like(pv_hat, fill_value=self._create_price_tariff(), dtype=torch.float64)

        Ps_in, Ps_out, Pl, Pr, eps, s = self.optimization_layer(price_tariff, load_hat, pv_hat, initial_state)
        Pg_hat = self.compute_Pg(Ps_in, Ps_out, Pr, Pl)
        return predictions, (Ps_in, Ps_out, Pl, Pr, eps, s), Pg_hat

    def update_state(self, s):
        """Update the current state with the last battery charge state."""
        self.current_state = s[:, -1].unsqueeze(1).to(self.device).float()

    def compute_Pg(self, Ps_in, Ps_out, Pr, Pl):
        """Compute the power exchanged with the utility grid."""
        return Pr - Ps_in + Ps_out - Pl

    def get_plant_matrices(self):
        """Compute the matrices of the real plant."""
        A_matrix = (1 - self._self_discharge) * torch.ones((1, 1), dtype=torch.float64, device=self.device)
        b_in = self._sample_time * self.efficiency * torch.ones((1, 1), dtype=torch.float64, device=self.device)
        b_out = -self._sample_time * (1 / self.efficiency) * torch.ones((1, 1), dtype=torch.float64, device=self.device)
        B_matrix = torch.cat((b_in, b_out), dim=1)
        return A_matrix, B_matrix

    def get_state_dimension(self):
        """Get the dimension of the system state based on plant matrices."""
        A_matrix = self.get_plant_matrices()[0]
        return A_matrix.shape[0]

    