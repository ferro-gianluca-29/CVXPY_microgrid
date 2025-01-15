from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import torch
import argparse
import pandas as pd

import logging
logging = logging.getLogger('pytorch_lightning')


class StateSpaceSystem(ABC):
    """Abstract class for state space models.

    The plant is described by the following state space model:
        `x(t+1) = A x(t) + B u(t)`

    where `x(t)` is the state vector, `u(t)` is the input vector and `A`, `B` are the state space matrices.
    The plant is simulated by the method plant_simulation that computes the next state `x(t+1)`
    given the current state `x(t)` and the input `u(t)`.

    Args:
        sample_time (int): Sample time of the system. 
        initial_state (List[float]): Initial value for each state of the system.
    """

    def __init__(self, sample_time: int, initial_state: List[float]):
        self._sample_time = sample_time
        self._initial_state = initial_state

    @abstractmethod
    def _get_plant_matrices(self, device: torch.device = torch.device("cpu")
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the state space matrices A, B of the real plant.

        These matrices represent the real system dynamics and
        are used to simulate the plant.
        """
        pass

    @abstractmethod
    def get_model_matrices(self, device: torch.device = torch.device("cpu")
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the state space matrices A, B of the system model.

        These matrices represent the model of the system that is used 
        in the controller.
        """
        pass

    def get_initial_state(self) -> torch.Tensor:
        """Get the initial system state."""
        return torch.tensor(self._initial_state, dtype=torch.float64).reshape(-1, 1)

    def plant_simulation(self, x_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """This method is used to simulate the state space system.

        For one time step: `x(t+1) = A x(t) + B u(t)`
        """
        A_matrix, B_matrix = self._get_plant_matrices(x_t.device)
        return torch.matmul(A_matrix, x_t) + torch.matmul(B_matrix, u_t)

    def get_state_dimension(self) -> int:
        """Return the state dimension of the system."""
        A_matrix = self._get_plant_matrices()[0]
        return A_matrix.shape[0]

    def get_input_dimension(self) -> int:
        """Return the input dimension of the system."""
        B_matrix = self._get_plant_matrices()[1]
        return B_matrix.shape[1]


class MicroGridSystem(StateSpaceSystem):
    """This class is used to create the model of the microgrid system.

    Args:
        self_discharge (float): Self discharge rate of the storage system
                                0 <= self_discharge <= 1,
                                self_discharge = 0 means no self discharge
        grid_power_min_max (Tuple[float, float]): Minimum and maximum power that can be injected into the grid.
    """

    def __init__(self,
                 self_discharge: float,
                 grid_power_min_max: Tuple[float, float], **kwargs: Any) -> None:
        StateSpaceSystem.__init__(self, **kwargs)
        assert 0 <= self_discharge <= 1, f"`self_discharge` must be in [0, 1], Got {self_discharge}."
        storage_data = pd.read_csv("./storage_data.csv")
        self._self_discharge = self_discharge
        self.efficiency = storage_data['charge_discharge_efficiency'][0]
        assert 0 <= self.efficiency <= 1, f"`efficiency` must be in [0, 1], Got {self.efficiency}."
        self.charge_min_max = [storage_data['capacity_min_kWh'][0].item(),
                               storage_data['capacity_max_kWh'][0].item()]
        self.storage_power_min_max = [storage_data['power_min_kW'][0].item(),
                                      storage_data['power_max_kW'][0].item()]
        self.grid_power_min_max = grid_power_min_max
        assert self.charge_min_max[0] <= self.charge_min_max[
            1], f"`charge_min_max` must be in [charge_min, charge_max], Got {self.charge_min_max}."
        assert self.storage_power_min_max[0] <= self.storage_power_min_max[
            1], f"`storage_power_min_max` must be in [power_min, power_max], Got {self.storage_power_min_max}."
        assert self.grid_power_min_max[0] <= self.grid_power_min_max[
            1], f"`grid_power_min_max` must be in [power_min, power_max], Got {self.grid_power_min_max}."

    def _get_plant_matrices(self, device: torch.device = torch.device("cpu")
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

    def get_model_matrices(self, device: torch.device = torch.device("cpu")
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is used to compute the matrices of the state space model.

        This is the function that the user has to implement.
        To use the same matrices for the real plant and the model set: 
        A_matrix, B_matrix = self._get_plant_matrices()
        Alternatively, the user can implement a different model as:
            A_matrix = ...
            B_matrix = ...
        """
        A_matrix, B_matrix = self._get_plant_matrices(device)
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
    

    def compute_next_state(self, current_state, Ps_in, Ps_out, Pr, Pl):
        """
        Calcola il prossimo stato del sistema basandosi sulle potenze in entrata e uscita della batteria,
        e le potenze dei generatori rinnovabili e dei carichi.

        Args:
            current_state (torch.Tensor): Stato corrente della batteria.
            Ps_in (torch.Tensor): Potenza immessa nella batteria.
            Ps_out (torch.Tensor): Potenza prelevata dalla batteria.
            Pr (torch.Tensor): Potenza generata dai rinnovabili.
            Pl (torch.Tensor): Potenza consumata dai carichi.

        Returns:
            torch.Tensor: Prossimo stato della batteria.
        """
        A_matrix, B_matrix = self.get_model_matrices()
        # Calcolo del prossimo stato usando la dinamica lineare del sistema.
        next_state = A_matrix @ current_state + B_matrix[:, 0:1] * Ps_in - B_matrix[:, 1:2] * Ps_out
        return next_state