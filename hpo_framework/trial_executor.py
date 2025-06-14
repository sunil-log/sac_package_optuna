# hpo_framework/trial_executor.py

import optuna
from ignite.engine import Engine, Events
from ignite.contrib.handlers.optuna import OptunaPruningHandler
from .base_handler import BaseExperimentHandler


def run_trial(trial: optuna.trial.Trial, handler: BaseExperimentHandler) -> float:
    """
    Ignite를 사용하여 단일 trial의 학습 및 평가를 실행합니다.
    이 함수는 더 이상 특정 모델이나 학습 로직에 의존하지 않습니다.

    Args:
        trial (optuna.trial.Trial): 현재 Optuna trial 객체.
        handler (BaseExperimentHandler): 실험의 모든 구성요소를 제공하는 핸들러.

    Returns:
        float: 최적화 대상 평가지표의 최종 점수.
    """
    # 1. 핸들러를 통해 실험 구성요소 가져오기
    model = handler.get_model().to(handler.device)
    train_loader, val_loader = handler.get_dataloaders()
    optimizer = handler.get_optimizer(model)
    loss_fn = handler.get_loss_fn()
    metrics = handler.get_metrics()
    train_step_fn = handler.get_train_step_fn(model, optimizer, loss_fn)
    eval_step_fn = handler.get_eval_step_fn(model)

    # 2. Ignite Engine 생성
    trainer = Engine(train_step_fn)
    evaluator = Engine(eval_step_fn)

    # 3. 평가지표 부착
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # 4. 핸들러(Handler) 부착
    # Pruning 핸들러: 성능이 낮은 trial을 조기 중단
    study_config = handler.params['static']['study']
    metric_to_optimize = study_config['metric_to_optimize']

    # OptunaPruningHandler는 최적화 방향에 따라 자동으로 점수를 처리합니다.
    pruning_handler = OptunaPruningHandler(trial, metric_to_optimize, trainer)
    evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

    # 매 에포크 종료 시 검증 및 로그 출력 핸들러
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        eval_metrics = evaluator.state.metrics
        metric_value = eval_metrics[metric_to_optimize]
        print(f"Trial {trial.number} - Epoch {engine.state.epoch} - {metric_to_optimize}: {metric_value:.4f}")

    # 5. 학습 실행
    max_epochs = handler.params['optimize']['training']['max_epochs']
    trainer.run(train_loader, max_epochs=max_epochs)

    # 6. 최종 검증 점수 반환
    final_metrics = evaluator.state.metrics
    return final_metrics[metric_to_optimize]
