# hpo_framework/runner.py

import os
from typing import Callable, Dict, Any
from copy import deepcopy
import yaml
import optuna

from .param_sampler import sample_params

def merge_configs(base: Dict[str, Any], new_values: Dict[str, Any]) -> Dict[str, Any]:
    """재귀적으로 딕셔너리를 업데이트하는 유틸리티 함수."""
    for k, v in new_values.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = merge_configs(base[k], v)
        else:
            base[k] = v
    return base

def run_hpo(config_path: str, session_fn: Callable[[Dict[str, Any]], float]):
    """
    설정 파일을 로드하고 Optuna HPO(Hyperparameter Optimization) 전체 과정을 실행합니다.

    Args:
        config_path (str): YAML 설정 파일 경로.
        session_fn (Callable[[Dict[str, Any]], float]):
            단일 실험 세션을 실행하는 함수. 하이퍼파라미터 딕셔너리를 인자로 받고,
            최적화할 점수(float)를 반환합니다.
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

        # 1.3. 프루닝을 위한 콜백 함수를 trial_params에 추가
        trial_params['trial'] = trial

        # 1.4. 사용자가 정의한 세션 함수를 실행하고 점수 반환
        try:
            score = session_fn(trial_params)
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # 실패한 trial은 Pruned 처리될 수 있도록 예외를 발생
            raise optuna.exceptions.TrialPruned()

        return score

    # 2. Optuna Study 생성 또는 로드
    study_config = config['static']['study']
    db_path = study_config['db_path']
    if os.path.dirname(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    storage_name = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_config['study_name'],
        storage=storage_name,
        load_if_exists=True,
        direction=study_config['direction'],
        pruner=optuna.pruners.MedianPruner()
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

