# hpo_framework/runner.py

import os
from typing import Type
from copy import deepcopy
import yaml
import optuna

from .param_sampler import sample_params
from .trial_executor import run_trial
from .base_handler import BaseExperimentHandler


def merge_configs(base, new_values):
    """재귀적으로 딕셔너리를 업데이트하는 유틸리티 함수."""
    for k, v in new_values.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = merge_configs(base[k], v)
        else:
            base[k] = v
    return base

def run_hpo(config_path: str, handler_class: Type[BaseExperimentHandler]):
    """
    설정 파일을 로드하고 Optuna HPO(Hyperparameter Optimization) 전체 과정을 실행합니다.

    Args:
        config_path (str): YAML 설정 파일 경로.
        handler_class (Type[BaseExperimentHandler]): 사용할 실험 핸들러의 클래스.
    """
    # 1. 설정 파일 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    def objective(trial: optuna.trial.Trial) -> float:
        """단일 Trial을 실행하는 objective 함수"""
        # 1.1. 하이퍼파라미터 샘플링
        optimized_params = sample_params(trial, config["optimize"])

        # 1.2. static 설정과 결합하여 최종 trial_params 생성
        trial_params = deepcopy(config)
        trial_params = merge_configs(trial_params, {"optimize": optimized_params})

        # 1.3. 핸들러 클래스를 인스턴스화
        handler = handler_class(trial_params)

        # 1.4. 학습 세션 실행 및 점수 반환
        try:
            score = run_trial(trial, handler)
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # 실패한 trial은 Pruned 처리될 수 있도록 예외를 발생
            raise optuna.exceptions.TrialPruned()

        return score

    # 2. Optuna Study 생성 또는 로드
    study_config = config['static']['study']
    db_path = study_config['db_path']
    # 디렉토리가 존재하지 않으면 생성
    if os.path.dirname(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    storage_name = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_config['study_name'],
        storage=storage_name,
        load_if_exists=True,
        direction=study_config['direction'],
        pruner=optuna.pruners.MedianPruner()  # Pruning 알고리즘
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
