
# -*- coding: utf-8 -*-
"""
Created on  Jun 15 2025

@author: sac
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 재사용 가능한 프레임워크에서 runner를 임포트
from sac_optuna.runner import run_experiment


# ===================================================================
# 1. 사용자 정의 컴포넌트 (모델, 데이터, 옵티마이저 등)
#	- 이 부분만 자신의 문제에 맞게 수정하면 됩니다.
# ===================================================================

class LinearRegressionModel(nn.Module):
	"""간단한 선형 회귀 모델"""

	def __init__(self, input_dim, output_dim):
		super(LinearRegressionModel, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		return self.linear(x)


def get_model(args: dict) -> nn.Module:
	"""설정값(args)을 바탕으로 모델 객체를 생성하여 반환합니다."""
	model_args = args['static']['model']
	return LinearRegressionModel(
		input_dim=model_args['input_dim'],
		output_dim=model_args['output_dim']
	)


def get_dataloaders(args: dict) -> (DataLoader, DataLoader):
	"""설정값(args)을 바탕으로 훈련/검증 데이터로더를 생성하여 반환합니다."""
	data_args = args['static']['data']
	batch_size = args['optimize']['training']['batch_size']

	X = np.random.randn(data_args['n_samples'], data_args['n_features']).astype(np.float32)
	true_weights = np.random.randn(data_args['n_features'], 1).astype(np.float32)
	true_bias = np.random.randn(1).astype(np.float32)
	y = (X @ true_weights + true_bias + np.random.randn(data_args['n_samples'], 1) * data_args['noise']).astype(
		np.float32)

	X_tensor = torch.from_numpy(X)
	y_tensor = torch.from_numpy(y)

	split_idx = int(data_args['n_samples'] * data_args['train_split'])
	train_dataset = TensorDataset(X_tensor[:split_idx], y_tensor[:split_idx])
	val_dataset = TensorDataset(X_tensor[split_idx:], y_tensor[split_idx:])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, val_loader


def get_optimizer(model: nn.Module, args: dict) -> torch.optim.Optimizer:
	"""모델과 설정값(args)을 바탕으로 옵티마이저 객체를 생성하여 반환합니다."""
	optimizer_name = args['optimize']['optimizer']['optimizer_name']
	lr = args['optimize']['training']['lr']

	if optimizer_name == "Adam":
		return torch.optim.Adam(model.parameters(), lr=lr)
	elif optimizer_name == "RMSprop":
		return torch.optim.RMSprop(model.parameters(), lr=lr)
	elif optimizer_name == "SGD":
		momentum = args['optimize']['optimizer']['sgd_momentum']
		return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	else:
		raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_loss_fn(args: dict):
	"""설정값(args)을 바탕으로 손실 함수를 생성하여 반환합니다."""
	# 이 예제에서는 고정된 MSE Loss를 사용합니다.
	return nn.MSELoss()


# ===================================================================
# 2. 실험 실행
# ===================================================================

if __name__ == "__main__":
	# 설정 파일 경로
	CONFIG_PATH = "config.yaml"

	# 프레임워크의 run_experiment 함수에 사용자 정의 컴포넌트들을 전달하여 실행
	run_experiment(
		config_path=CONFIG_PATH,
		get_model_fn=get_model,
		get_data_fn=get_dataloaders,
		get_optimizer_fn=get_optimizer,
		get_loss_fn=get_loss_fn
	)

