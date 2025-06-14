# experiments/linear_regression.py

from typing import Dict, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ignite.metrics import Metric, MeanSquaredError
import numpy as np

from hpo_framework.base_handler import BaseExperimentHandler


class LinearRegressionModel(nn.Module):
    """간단한 선형 회귀 모델"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class LinearRegressionHandler(BaseExperimentHandler):
    """
    선형 회귀 실험을 위한 구체적인 핸들러 구현.
    BaseExperimentHandler의 모든 추상 메서드를 구현합니다.
    """

    def get_model(self) -> nn.Module:
        model_args = self.params['static']['model']
        return LinearRegressionModel(
            input_dim=model_args['input_dim'],
            output_dim=model_args['output_dim']
        )

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        data_args = self.params['static']['data']
        batch_size = self.params['optimize']['training']['batch_size']

        X = np.random.randn(data_args['n_samples'], data_args['n_features']).astype(np.float32)
        true_weights = np.random.randn(data_args['n_features'], 1).astype(np.float32)
        true_bias = np.random.randn(1).astype(np.float32)
        y = (X @ true_weights + true_bias + np.random.randn(data_args['n_samples'], 1) * data_args['noise']).astype(np.float32)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        split_idx = int(data_args['n_samples'] * data_args['train_split'])
        train_dataset = TensorDataset(X_tensor[:split_idx], y_tensor[:split_idx])
        val_dataset = TensorDataset(X_tensor[split_idx:], y_tensor[split_idx:])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        optimizer_name = self.params['optimize']['optimizer']['optimizer_name']
        lr = self.params['optimize']['training']['lr']

        if optimizer_name == "Adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "RMSprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            momentum = self.params['optimize']['optimizer']['sgd_momentum']
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def get_loss_fn(self) -> Callable:
        return nn.MSELoss()

    def get_metrics(self) -> Dict[str, Metric]:
        # config.yaml의 'metric_to_optimize'에 지정된 'val_mse'를 키로 사용
        return {"val_mse": MeanSquaredError()}

    def get_train_step_fn(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: Callable) -> Callable:
        def train_step(engine, batch):
            model.train()
            optimizer.zero_grad()
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            return loss.item()
        return train_step

    def get_eval_step_fn(self, model: nn.Module) -> Callable:
        def eval_step(engine, batch):
            model.eval()
            with torch.no_grad():
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                y_pred = model(x)
            return y_pred, y
        return eval_step
