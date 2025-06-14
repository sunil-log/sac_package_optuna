
# -*- coding: utf-8 -*-
"""
Created on  Jun 15 2025

@author: sac
"""

# sac_optuna/sampler.py
import optuna


def sample_params(trial: optuna.trial.Trial, param_space: dict) -> dict:
	"""
	YAML에 정의된 탐색 공간으로부터 하이퍼파라미터를 재귀적으로 샘플링합니다.
	"""
	params = {}
	for name, config in param_space.items():
		# 'type' 키가 있는 dict는 최종 파라미터로 간주합니다.
		if isinstance(config, dict) and "type" in config:
			param_type = config["type"]
			# Optuna의 suggest API를 동적으로 호출합니다.
			if param_type == "int":
				params[name] = trial.suggest_int(name, config["low"], config["high"], step=config.get("step", 1),
				                                 log=config.get("log", False))
			elif param_type == "float":
				params[name] = trial.suggest_float(name, config["low"], config["high"], step=config.get("step", None),
				                                   log=config.get("log", False))
			elif param_type == "categorical":
				params[name] = trial.suggest_categorical(name, config["choices"])
			else:
				raise ValueError(f"Unknown param type: {param_type}")
		# 'type' 키가 없는 dict는 중첩된 구조로 간주하고 재귀 호출합니다.
		elif isinstance(config, dict):
			params[name] = sample_params(trial, config)

	return params