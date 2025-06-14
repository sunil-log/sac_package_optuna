
# -*- coding: utf-8 -*-
"""
Created on  Jun 15 2025

@author: sac
"""

# sac_optuna/ignite_trainer.py

import torch
from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError
from ignite.contrib.handlers.optuna import OptunaPruningHandler


def run_trial(
		trial,
		args,
		model,
		optimizer,
		loss_fn,
		train_loader,
		val_loader
) -> float:
	"""
	Ignite를 사용하여 단일 trial의 학습 및 평가를 실행합니다.
	"""
	device = torch.device(args['static']['device'])
	model.to(device)

	# 1. Ignite Engine 정의
	def train_step(engine, batch):
		model.train()
		optimizer.zero_grad()
		x, y = batch[0].to(device), batch[1].to(device)
		y_pred = model(x)
		loss = loss_fn(y_pred, y)
		loss.backward()
		optimizer.step()
		return loss.item()

	def eval_step(engine, batch):
		model.eval()
		with torch.no_grad():
			x, y = batch[0].to(device), batch[1].to(device)
			y_pred = model(x)
		return y_pred, y

	trainer = Engine(train_step)
	evaluator = Engine(eval_step)

	# 평가 지표(Metric) 부착
	# TODO: 이 부분은 문제에 맞게 수정이 필요할 수 있습니다.
	# 현재는 MSE로 고정되어 있습니다.
	MeanSquaredError().attach(evaluator, "metric")

	# 2. 핸들러(Handler) 부착
	# Pruning 핸들러: 성능이 낮은 trial을 조기 중단
	# direction이 'minimize'일 경우 metric, 'maximize'일 경우 -metric을 사용
	score_name = "metric"
	pruning_handler = OptunaPruningHandler(trial, score_name, trainer)
	evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

	# 매 에포크 종료 시 검증 및 로그 출력 핸들러
	@trainer.on(Events.EPOCH_COMPLETED)
	def log_validation_results(engine):
		evaluator.run(val_loader)
		metrics = evaluator.state.metrics
		print(f"Trial {trial.number} - Epoch {engine.state.epoch} - Val Metric: {metrics[score_name]:.4f}")

	# 3. 학습 실행
	trainer.run(train_loader, max_epochs=args['optimize']['training']['max_epochs'])

	# 4. 최종 검증 점수 반환
	final_metrics = evaluator.state.metrics
	return final_metrics[score_name]