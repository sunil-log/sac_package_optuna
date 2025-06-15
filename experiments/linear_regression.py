# experiments/linear_regression.py

from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# 모델 정의는 그대로 유지
class LinearRegressionModel(nn.Module):
	"""간단한 선형 회귀 모델"""

	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.linear = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		return self.linear(x)


def single_session(cfg: Dict[str, Any]) -> float:
	"""
	단일 하이퍼파라미터 설정(`cfg`)을 사용하여 전체 훈련 및 평가 파이프라인을 실행하고,
	최적화 대상 점수를 반환합니다.

	Args:
		cfg (Dict[str, Any]): 정적 파라미터와 샘플링된 하이퍼파라미터를 포함하는 딕셔너리.

	Returns:
		float: 검증 데이터셋에 대한 최종 MSE 점수.
	"""
	# --- 1. 설정값 추출 ---
	trial = cfg['trial']  # Optuna trial 객체
	static_params = cfg['static']
	optim_params = cfg['optimize']

	device = torch.device(static_params['device'])

	# 데이터 관련 설정
	data_args = static_params['data']
	n_samples = data_args['n_samples']
	n_features = data_args['n_features']

	# 모델 관련 설정
	model_args = static_params['model']

	# 훈련 관련 설정
	training_args = optim_params['training']
	lr = training_args['lr']
	batch_size = training_args['batch_size']
	max_epochs = training_args['max_epochs']

	# 옵티마이저 관련 설정
	optimizer_args = optim_params['optimizer']
	optimizer_name = optimizer_args['optimizer_name']

	# --- 2. 데이터 준비 ---
	X = np.random.randn(n_samples, n_features).astype(np.float32)
	true_weights = np.random.randn(n_features, 1).astype(np.float32)
	true_bias = np.random.randn(1).astype(np.float32)
	y = (X @ true_weights + true_bias + np.random.randn(n_samples, 1) * data_args['noise']).astype(np.float32)

	X_tensor = torch.from_numpy(X)
	y_tensor = torch.from_numpy(y)
	dataset = TensorDataset(X_tensor, y_tensor)

	train_size = int(data_args['train_split'] * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	# --- 3. 모델, 손실함수, 옵티마이저 생성 ---
	model = LinearRegressionModel(
		input_dim=model_args['input_dim'],
		output_dim=model_args['output_dim']
	).to(device)

	loss_fn = nn.MSELoss()

	if optimizer_name == "Adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	elif optimizer_name == "RMSprop":
		optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
	elif optimizer_name == "SGD":
		momentum = optimizer_args['sgd_momentum']
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	else:
		raise ValueError(f"Unknown optimizer: {optimizer_name}")

	# --- 4. 훈련 및 검증 루프 실행 ---
	print(f"\n--- Trial {trial.number}: Starting ---")
	print(f"Params: {trial.params}")

	for epoch in range(max_epochs):
		# 훈련
		model.train()
		for x_batch, y_batch in train_loader:
			x_batch, y_batch = x_batch.to(device), y_batch.to(device)

			optimizer.zero_grad()
			y_pred = model(x_batch)
			loss = loss_fn(y_pred, y_batch)
			loss.backward()
			optimizer.step()

		# 검증
		model.eval()
		val_loss = 0
		with torch.no_grad():
			for x_val, y_val in val_loader:
				x_val, y_val = x_val.to(device), y_val.to(device)
				y_pred_val = model(x_val)
				val_loss += loss_fn(y_pred_val, y_val).item()

		avg_val_loss = val_loss / len(val_loader)
		print(f"Trial {trial.number} - Epoch {epoch + 1}/{max_epochs} - Val MSE: {avg_val_loss:.4f}")

		# Optuna Pruning
		trial.report(avg_val_loss, epoch)
		if trial.should_prune():
			print(f"Trial {trial.number} pruned at epoch {epoch + 1}.")
			raise optuna.exceptions.TrialPruned()

	print(f"--- Trial {trial.number}: Finished. Final Val MSE: {avg_val_loss:.4f} ---")

	# --- 5. 최종 점수 반환 ---
	# Optuna는 이 반환값을 사용하여 하이퍼파라미터를 최적화합니다.
	return avg_val_loss
