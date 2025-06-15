# hpo_framework/runner.py

import os
from typing import Callable, Dict, Any
from copy import deepcopy
import yaml
import optuna

from .param_sampler import sample_params

class DotDict(dict):
    """
    점(.) 표기법으로 접근 가능한 딕셔너리 클래스.
    중첩된 딕셔너리도 재귀적으로 변환합니다.
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")


def merge_configs(base: Dict[str, Any], new_values: Dict[str, Any]) -> Dict[str, Any]:
    """재귀적으로 딕셔너리를 업데이트하는 유틸리티 함수."""
    for k, v in new_values.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = merge_configs(base[k], v)
        else:
            base[k] = v
    return base

def _create_pruner(config: Dict[str, Any]) -> optuna.pruners.BasePruner:
    """설정 파일에 기반하여 Pruner 객체를 생성합니다."""
    pruner_type = config.get("type", "MedianPruner").lower()
    pruner_args = {k: v for k, v in config.items() if k != "type"}

    if pruner_type == "medianpruner":
        return optuna.pruners.MedianPruner(**pruner_args)
    elif pruner_type == "hyperbandpruner":
        return optuna.pruners.HyperbandPruner(**pruner_args)
    elif pruner_type == "none" or pruner_type is None:
        return optuna.pruners.NopPruner() # Pruning을 수행하지 않음
    else:
        raise ValueError(f"지원하지 않는 Pruner 타입입니다: {config.get('type')}")


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
        trial_params['trial'] = trial

        # 1.3. 사용자가 정의한 세션 함수를 실행하고 점수 반환
        try:
            # session_fn에 전달하기 직전, DotDict으로 감싸줍니다.
            score = session_fn(DotDict(trial_params))
        except optuna.exceptions.TrialPruned as e:
            # Pruning으로 인한 예외는 그대로 다시 발생시켜 Optuna가 처리하도록 합니다.
            raise e
        except Exception as e:
            # 그 외 모든 예외는 에러로 기록하고 trial을 실패(FAILED) 상태로 만듭니다.
            print(f"Trial {trial.number} failed with an unexpected error: {e}")
            # 예외를 다시 발생시켜 Optuna가 trial을 FAILED로 처리하도록 합니다.
            raise e

        return score

    # 2. Optuna Study 생성 또는 로드
    study_config = config['static']['study']
    db_path = study_config['db_path']
    if os.path.dirname(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    storage_name = f"sqlite:///{db_path}"

    pruner = _create_pruner(study_config.get("pruner", {}))

    study = optuna.create_study(
        study_name=study_config['study_name'],
        storage=storage_name,
        load_if_exists=True,
        direction=study_config['direction'],
        pruner=pruner
    )

    # 3. 최적화 실행
    study.optimize(objective, n_trials=study_config['n_trials'])

    # 4. 결과 출력
    print("\n================== Optimization Finished ==================")
    print(f"Study: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")

    # 실패하지 않은 trial 중에서 최적의 결과를 찾습니다.
    try:
        best_trial = study.best_trial
        print(f"Best trial value: {best_trial.value:.5f}")

        print("Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  - {key}: {value}")
    except ValueError:
        print("모든 Trial이 실패하여 최적의 하이퍼파라미터를 찾을 수 없습니다.")

