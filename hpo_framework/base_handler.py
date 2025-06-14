# hpo_framework/base_handler.py

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.metrics import Metric


class BaseExperimentHandler(ABC):
    """
    실험 정의를 위한 추상 베이스 클래스 (Abstract Base Class).

    이 클래스는 사용자가 자신의 실험을 프레임워크에 통합하기 위해 구현해야 하는
    메서드들의 인터페이스를 정의합니다. 프레임워크는 이 핸들러를 통해 모델,
    데이터, 학습 로직 등 실험에 필요한 모든 구성요소를 얻습니다.
    """

    def __init__(self, trial_params: dict):
        """
        Args:
            trial_params (dict): Optuna trial에 의해 샘플링된 파라미터와
                                 정적 파라미터를 포함하는 전체 설정 딕셔너리.
        """
        self.params = trial_params
        self.device = torch.device(self.params['static']['device'])

    @abstractmethod
    def get_model(self) -> nn.Module:
        """
        설정(self.params)에 기반하여 모델 객체를 생성하고 반환합니다.
        """
        pass

    @abstractmethod
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        설정(self.params)에 기반하여 훈련 및 검증 데이터로더를 생성하고 반환합니다.
        """
        pass

    @abstractmethod
    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        모델과 설정(self.params)에 기반하여 옵티마이저를 생성하고 반환합니다.
        """
        pass

    @abstractmethod
    def get_loss_fn(self) -> Callable:
        """
        설정(self.params)에 기반하여 손실 함수를 생성하고 반환합니다.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Metric]:
        """
        평가에 사용할 Ignite Metric 딕셔너리를 반환합니다.
        key는 메트릭 이름, value는 Ignite Metric 객체입니다.
        config.yaml의 'metric_to_optimize'에 명시된 key가 반드시 포함되어야 합니다.
        """
        pass

    @abstractmethod
    def get_train_step_fn(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: Callable) -> Callable:
        """
        Ignite Trainer Engine에 사용될 train_step 함수를 반환합니다.
        이 함수는 단일 훈련 배치를 처리하는 로직을 포함합니다.

        Args:
            model (nn.Module): 훈련할 모델.
            optimizer (torch.optim.Optimizer): 옵티마이저.
            loss_fn (Callable): 손실 함수.

        Returns:
            Callable: Ignite Engine이 호출할 `(engine, batch) -> loss` 형태의 함수.
        """
        pass

    @abstractmethod
    def get_eval_step_fn(self, model: nn.Module) -> Callable:
        """
        Ignite Evaluator Engine에 사용될 eval_step 함수를 반환합니다.
        이 함수는 단일 평가 배치를 처리하는 로직을 포함합니다.

        Args:
            model (nn.Module): 평가할 모델.

        Returns:
            Callable: Ignite Engine이 호출할 `(engine, batch) -> (y_pred, y)` 형태의 함수.
        """
        pass
