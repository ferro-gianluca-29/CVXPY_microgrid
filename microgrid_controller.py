import torch
import torch.nn as nn
import argparse
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpylayers.torch import CvxpyLayer
from typing import Tuple, Dict, Any
import pandas as pd

from sklearn.preprocessing import MinMaxScaler




class OptimizationLayer(nn.Module):
    """This class is used to define the optimization problem.

    Args:
        prediction_horizon (int): Prediction horizon.
        A_matrix (torch.Tensor): A matrix of the storage dynamics.
        B_matrix (torch.Tensor): B matrix of the storage dynamics.
        charge_min_max (Tuple[float, float]): Minimum and maximum storage charge.
        storage_power_min_max (Tuple[float, float]): Minimum and maximum storage power.
        grid_power_min_max (Tuple[float, float]): Minimum and maximum grid power.

    The optimization problem is defined as a neural network that uses the parameters
    predicted by the controller to solve the optimization problem.
    """

    def __init__(self, prediction_horizon: int,
                 A_matrix: torch.Tensor,
                 B_matrix: torch.Tensor,
                 charge_min_max: Tuple[float, float],
                 storage_power_min_max: Tuple[float, float],
                 grid_power_min_max: Tuple[float, float]) -> None:
        super(OptimizationLayer, self).__init__()
        # Check that min < max
        assert charge_min_max[0] < charge_min_max[
            1], f"The minimum charge must be smaller than the maximum charge. Got {charge_min_max[0]} and {charge_min_max[1]}."
        assert storage_power_min_max[0] < storage_power_min_max[
            1], f"The minimum storage power must be smaller than the maximum storage power. Got {storage_power_min_max[0]} and {storage_power_min_max[1]}."
        assert grid_power_min_max[0] < grid_power_min_max[
            1], f"The minimum grid power must be smaller than the maximum grid power. Got {grid_power_min_max[0]} and {grid_power_min_max[1]}."

        self._prediction_horizon = prediction_horizon

        # Define the variables
        s = Variable((1, prediction_horizon+1))  # Storage charge
        Ps_in = Variable((1, prediction_horizon))  # Storage power in
        Ps_out = Variable((1, prediction_horizon))  # Storage power out
        Pl = Variable((1, prediction_horizon))  # Load power
        Pr = Variable((1, prediction_horizon))  # Renewable generator power
        eps = Variable((1, prediction_horizon))  # Soft constraint

        # Define the parameters
        price = Parameter((1, prediction_horizon))
        load = Parameter((1, prediction_horizon))
        pv = Parameter((1, prediction_horizon))
        s0 = Parameter((1, 1))

        # Define the objective function and constraints
        objective = 0.0
        constraints = []
        for k in range(prediction_horizon):
            objective += -1 * price[0, k] * \
                (Pr[:, k]-Pl[:, k] - (Ps_in[:, k] - Ps_out[:, k])) + 10000*(eps[:, k])
            constraints += [
                s[:, k+1] == A_matrix @ s[:, k] +
                B_matrix[0,0] * Ps_in[:, k] + B_matrix[0,1] * Ps_out[:, k],  # Storage dynamics
                s[:, k] >= charge_min_max[0],  # Storage min charge limit
                s[:, k] <= charge_min_max[1],  # Storage max charge limit
                # Storage min power limit
                Ps_in[:, k] <= storage_power_min_max[1],
                Ps_in[:, k] >= 0,
                Ps_out[:, k] <= storage_power_min_max[1],
                Ps_out[:, k] >= 0, 
                Pl[:, k] >= load[0, k] - eps[:, k],  # Load lower bounds
                Pl[:, k] <= load[0, k] + eps[:, k],  # Load upper bounds
                Pr[:, k] >= pv[0, k] - eps[:, k],  # Renewable lower bounds
                Pr[:, k] <= pv[0, k] + eps[:, k],  # Renewable upper bounds
                Pr[:, k] - Pl[:, k] - \
                Ps_in[:, k] + Ps_out[:, k] >= grid_power_min_max[0],  # Max grid power
                Pr[:, k] - Pl[:, k] - \
                Ps_in[:, k] + Ps_out[:, k] <= grid_power_min_max[1]  # Min grid power
            ]
        constraints += [s[:, 0] == s0]  # Feedback constraint

        # Define the problem
        problem = Problem(Minimize(objective), constraints)
        self.optimization_layer = CvxpyLayer(problem,
                                              parameters=[price, load, pv, s0],
                                              variables=[s, Ps_in, Ps_out, Pl, Pr, eps])

    def forward(self, price_tariff: torch.Tensor,
                load_hat: torch.Tensor,
                pv_hat: torch.Tensor,
                initial_state: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the optimization problem.

        Args:
            price_tariff (torch.Tensor): Price teriff.
            load_hat (torch.Tensor): Predicted load.
            pv_hat (torch.Tensor): Predicted pv.
            initial_state (torch.Tensor): Initial state of the storage.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The solution of the optimization problem.
        """
        batch_size = load_hat.shape[0]
        device = load_hat.device
        assert price_tariff.shape[0] == load_hat.shape[
            0], f"The batch size of the price tariff and load predictions must be the same. Got {price_tariff.shape[0]} and {load_hat.shape[0]}."
        assert price_tariff.shape[0] == pv_hat.shape[
            0], f"The batch size of the price tariff and pv predictions must be the same. Got {price_tariff.shape[0]} and {pv_hat.shape[0]}."

        s = torch.zeros(
            (batch_size, self._prediction_horizon+1), device=device, dtype=torch.float64)
        Ps_in = torch.zeros((batch_size, self._prediction_horizon),
                         device=device, dtype=torch.float64)
        Ps_out = torch.zeros((batch_size, self._prediction_horizon),
                         device=device, dtype=torch.float64)
        Pl = torch.zeros((batch_size, self._prediction_horizon),
                         device=device, dtype=torch.float64)
        Pr = torch.zeros((batch_size, self._prediction_horizon),
                         device=device, dtype=torch.float64)
        eps = torch.zeros(
            (batch_size, self._prediction_horizon), device=device, dtype=torch.float64)
        
        # Compute solution for each input in the batch
        # per ogni finestra del batch calcolo le decisioni
        for i in range(batch_size):
            s[i, :], Ps_in[i, :], Ps_out[i, :], Pl[i, :], Pr[i, :], eps[i, :] = \
                self.optimization_layer(price_tariff[i:i+1, :], load_hat[i:i+1, :],
                                         pv_hat[i:i+1, :], initial_state[i:i+1, 0:1])

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


        #🚫🚫🚫🚫🚫🚫   AGGIORNA IL CODICE IN MODO DA DARLI IN INPUT SENZA DEFINIRLI QUA!🚫🚫🚫🚫🚫

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

        self.initial_state = torch.tensor([200], dtype=torch.float32).reshape(-1, 1)

        self.current_state = self.initial_state[0] * torch.ones(self._batch_size_train, self._lookback_window,
                                                          self.get_state_dimension(), device='cpu', dtype=torch.float32)

        self._hidden_dim = 48
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
        predictions, _ = self._lstm(inputs)
        predictions = self._fc(predictions[:, -1, :])
        predictions = predictions.reshape(predictions.shape[0],
                                          self._prediction_horizon,
                                          self._output_dim)
        
        #pv_hat = predictions[:, :, 0]  # Prendi la prima colonna per pv
        #load_hat = predictions[:, :, 1]  # Prendi la seconda colonna per load

        # Denormalizzo le previsioni prima di darle in input al layer di ottimizzazione
        pv_hat, load_hat = self.denormalize(predictions)
        
        price_tariff = torch.zeros_like(pv_hat)
        price_tariff.fill_(self._create_price_tariff())

        # Assicura che initial_state sia sempre 2D
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(1)

        initial_state = initial_state.float()


        # Passaggio dei valori al layer di ottimizzazione
        Ps_in, Ps_out, Pl, Pr, eps, s = self.optimization_layer(
            price_tariff, load_hat, pv_hat, initial_state)

        return predictions, (Ps_in, Ps_out, Pl, Pr, eps, s)
    
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

    