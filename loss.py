from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch

import logging
logging = logging.getLogger('pytorch_lightning')


class MSELoss(nn.Module):
    """This class is used to create a mean squared error loss function."""

    def __init__(self) -> None:
        super(MSELoss, self).__init__()
        self.loss = F.mse_loss

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """This method is used to perform a forward pass through the loss function.
        
        Args:
            y_hat (torch.Tensor): The predicted values.
            y (torch.Tensor): The true values.
            weight (Optional[torch.Tensor], optional): The weight vector. Defaults to None.
        """
        device = y_hat.device  # Assume y_hat is already on GPU
        batch_size, _, _ = y_hat.shape
        loss = 0.0
        for i in range(batch_size):
            loss_per_sample = self.loss(y_hat[i], y[i], reduction='none')
            if weight is not None:
                loss_per_sample = loss_per_sample * weight[:, None].to(device)
            loss += loss_per_sample.sum()
        loss = loss / batch_size
        return loss

    def get_weight(self, prediction_horizon: int, exp_decay: float,
                   device: torch.device) -> torch.Tensor:
        """Create a weight vector for the loss function, different for each prediction horizon."""
        return exp_decay ** torch.arange(0, prediction_horizon, device=device)
    

class TotalCostLoss(nn.Module):
    """Class for calculating MSE loss between real and predicted total cost."""

    def __init__(self) -> None:
        super(TotalCostLoss, self).__init__()
        self.mse_loss = MSELoss()

    def forward(self, Pg_hat: torch.Tensor, Pg: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """Calculate MSE loss between real cost and predicted cost using given price and production vectors."""

        device = Pg_hat.device  # Assume Pg_hat is already on GPU
        batch_size, _ = Pg_hat.shape
        loss = 0.0

        for i in range(batch_size):
            real_cost = total_cost(price=price.to(device), Pg=Pg[i].to(device))
            predicted_cost = total_cost(price=price.to(device), Pg=Pg_hat[i].to(device))
            loss_per_sample = (real_cost - predicted_cost) ** 2
            loss += loss_per_sample

        loss = loss / batch_size
        return loss
            
def total_cost(price: torch.Tensor, Pg: torch.Tensor) -> torch.Tensor:
    """This function computes the total cost of the power generation.
    Args:
        price (torch.Tensor): electricity price in [€/MWh].
        Pg (torch.Tensor): power generation in [kW].
    Returns:
        float: total cost in [€] (sample time assumed to be 1h).
    """
    return -1*torch.sum(price * Pg) / 1000  # [€]
