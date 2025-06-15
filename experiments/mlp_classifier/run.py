
# -*- coding: utf-8 -*-
"""
Created on  Jun 15 2025

@author: sac
"""

# experiments/mlp_classifier/run.py

from typing import TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import optuna

if TYPE_CHECKING:
	from hpo_framework.runner import DotDict


# --- 모델 정의 ---
class MLPClassifier(nn.Module):
	"""간단한 다층 퍼셉트론 분류 모델"""

	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout_rate),
			nn.Linear(hidden_dim, output_dim)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layers(x)


# --- 단일 세션 함수 ---
def single_session(cfg: "DotDict") -> float:
	"""단일 하이퍼파라미터 설정으로 MLP 분류기 훈련 및 평가를 실행합니다."""
	# --- 1. 설정값 추출 ---
	trial = cfg.trial
	device = torch.device(cfg.static.device)
	data_args = cfg.static.data
	model_args = cfg.static.model
	training_args = cfg.optimize.training
	optimizer_args = cfg.optimize.optimizer

	# --- 2. 데이터 준비 ---
	X, y = make_classification(
		n_samples=data_args.n_samples,
		n_features=model_args.input_dim,
		n_informative=data_args.n_informative,
		n_redundant=0,
		random_state=42
	)
	X = X.astype(np.float32)
	# BCEWithLogitsLoss를 위해 타겟을 float으로 변경하고 차원을 맞춤
	y = y.astype(np.float32).reshape(-1, 1)

	dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
	train_size = int(data_args.train_split * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	train_loader = DataLoader(train_dataset, batch_size=training_args.batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=training_args.batch_size)

	# --- 3. 모델, 손실함수, 옵티마이저 생성 ---
	model = MLPClassifier(
		input_dim=model_args.input_dim,
		hidden_dim=model_args.hidden_dim,
		output_dim=model_args.output_dim,
		dropout_rate=training_args.dropout_rate
	).to(device)

	loss_fn = nn.BCEWithLogitsLoss()
	optimizer = getattr(torch.optim, optimizer_args.optimizer_name)(model.parameters(), lr=training_args.lr)

	# --- 4. 훈련 및 검증 루프 ---
	print(f"\n--- Trial {trial.number}: Starting ---")
	print(f"Params: {trial.params}")

	final_val_loss = float('inf')
	for epoch in range(training_args.max_epochs):
		model.train()
		for x_batch, y_batch in train_loader:
			x_batch, y_batch = x_batch.to(device), y_batch.to(device)
			optimizer.zero_grad()
			y_pred_logits = model(x_batch)
			loss = loss_fn(y_pred_logits, y_batch)
			loss.backward()
			optimizer.step()

		model.eval()
		val_loss = 0
		all_preds = []
		all_labels = []
		with torch.no_grad():
			for x_val, y_val in val_loader:
				x_val, y_val = x_val.to(device), y_val.to(device)
				y_pred_val_logits = model(x_val)
				val_loss += loss_fn(y_pred_val_logits, y_val).item()

				preds = torch.sigmoid(y_pred_val_logits) > 0.5
				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(y_val.cpu().numpy())

		avg_val_loss = val_loss / len(val_loader)
		accuracy = accuracy_score(all_labels, all_preds)
		print(f"Trial {trial.number} - Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f} - Accuracy: {accuracy:.4f}")

		# Optuna Pruning
		trial.report(avg_val_loss, epoch)
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()
		final_val_loss = avg_val_loss

	print(f"--- Trial {trial.number}: Finished. Final Val Loss: {final_val_loss:.4f} ---")

	# study의 direction이 'minimize'이므로 validation loss를 반환
	return final_val_loss
