# Ignite-Optuna 프레임워크 프로젝트

이 프로젝트는 PyTorch Ignite와 Optuna를 연동하여 하이퍼파라미터 최적화를 수행하는 재사용 가능한 프레임워크를 제공합니다.

---

## 프로젝트 구조

```
.
├── config.yaml                     # 실험의 모든 설정을 정의하는 파일
├── run_linear_regression.py        # 실제 실험을 정의하고 실행하는 메인 스크립트
└── sac_optuna/                     # 재사용 가능한 최적화 프레임워크 패키지
    ├── __init__.py
    ├── runner.py                   # Optuna Study 생성 및 실행 관리자
    ├── ignite_trainer.py           # Ignite Engine을 사용한 학습/평가 세션 실행
    └── sampler.py                  # YAML 기반 Optuna 파라미터 샘플링 유틸리티
```

-   **sac_optuna/**: 이 디렉토리는 최적화 로직의 핵심이며, 다른 프로젝트에서도 그대로 가져와 사용할 수 있습니다.
    -   **runner.py**: `run_experiment` 함수를 통해 전체 최적화 과정을 orchestrate 합니다.
    -   **ignite\_trainer.py**: `run_trial` 함수를 통해 단일 trial의 학습/평가 과정을 Ignite로 처리합니다.
    -   **sampler.py**: `config.yaml`의 `optimize` 섹션을 파싱하여 하이퍼파라미터를 샘플링합니다.
-   **run\_linear\_regression.py**: 사용자가 자신의 모델과 데이터를 적용하는 예시 파일입니다.
    -   자신만의 `Model`, `get_dataloaders`, `get_optimizer`, `get_loss_fn` 등을 정의합니다.
    -   `sac_optuna.runner.run_experiment` 함수에 이 컴포넌트들을 전달하여 최적화를 실행합니다.
-   **config.yaml**: static 파라미터와 `optimize`할 하이퍼파라미터 탐색 공간을 정의합니다. 코드를 수정하지 않고 이 파일만 변경하여 다양한 실험을 수행할 수 있습니다.

---

## 실행 방법

1.  필요한 라이브러리를 설치합니다.
    ```bash
    pip install torch pyyaml optuna ignite sqlalchemy
    ```
2.  아래 명령어를 통해 선형 회귀 예제에 대한 하이퍼파라미터 최적화를 시작합니다.
    ```bash
    python run_linear_regression.py
    ```