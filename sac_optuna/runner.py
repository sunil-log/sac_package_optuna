
# -*- coding: utf-8 -*-
"""
Created on  Jun 15 2025

@author: sac
"""

# sac_optuna/runner.py

import os
from copy import deepcopy
import yaml
import optuna
from .sampler import sample_params
from .ignite_trainer import run_trial


def run_experiment(config_path, get_model_fn, get_data_fn, get_optimizer_fn, get_loss_fn):
	"""
	설정 파일을 로드하고 Optuna 최적화 전체 과정을 실행합니다.

	Args:
		config_path (str): YAML 설정 파일 경로.
		get_model_fn (callable): args를 받아 모델을 반환하는 함수.
		get_data_fn (callable): args를 받아 train/val 데이터로더를 반환하는 함수.
		get_optimizer_fn (callable): model과 args를 받아 옵티마이저를 반환하는 함수.
		get_loss_fn (callable): args를 받아 손실 함수를 반환하는 함수.
	"""
	# 1. 설정 파일 로드
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	def objective(trial: optuna.trial.Trial) -> float:
		"""단일 Trial을 실행하는 objective 함수"""
		# 1.1. 하이퍼파라미터 샘플링
		optimized_params = sample_params(trial, config["optimize"])

		# 1.2. static 설정과 결합하여 최종 args 생성
		trial_args = deepcopy(config)
		# deep_update와 같은 유틸리티를 사용하면 더 좋습니다.
		trial_args["optimize"] = optimized_params

		# 1.3. 사용자 정의 함수를 통해 실험 컴포넌트 생성
		model = get_model_fn(trial_args)
		train_loader, val_loader = get_data_fn(trial_args)
		optimizer = get_optimizer_fn(model, trial_args)
		loss_fn = get_loss_fn(trial_args)

		# 1.4. 학습 세션 실행 및 점수 반환
		try:
			score = run_trial(trial, trial_args, model, optimizer, loss_fn, train_loader, val_loader)
		except Exception as e:
			print(f"Trial {trial.number} failed with error: {e}")
			# 실패한 trial은 Pruned 처리될 수 있도록 예외를 발생
			raise optuna.exceptions.TrialPruned()

		return score

	# 2. Optuna Study 생성 또는 로드
	study_config = config['static']['study']
	db_path = study_config['db_path']
	os.makedirs(os.path.dirname(db_path), exist_ok=True)
	storage_name = f"sqlite:///{db_path}"

	study = optuna.create_study(
		study_name=study_config['study_name'],
		storage=storage_name,
		load_if_exists=True,
		direction=study_config['direction'],
		pruner=optuna.pruners.MedianPruner()  # Pruning 알고리즘 추가
	)

	# 3. 최적화 실행
	study.optimize(objective, n_trials=study_config['n_trials'])

	# 4. 결과 출력
	print("\n================== Optimization Finished ==================")
	print(f"Study: {study.study_name}")
	print(f"Number of finished trials: {len(study.trials)}")

	best_trial = study.best_trial
	print(f"Best trial value: {best_trial.value:.5f}")

	print("Best hyperparameters:")
	for key, value in best_trial.params.items():
		print(f"  - {key}: {value}")