# 범용 하이퍼파라미터 최적화(HPO) 프레임워크

이 프로젝트는 **PyTorch Ignite**와 **Optuna**를 사용하여, 다양한 머신러닝/딥러닝 실험에 적용할 수 있는 재사용 가능한 하이퍼파라미터 최적화(HPO) 프레임워크를 제공합니다.

---

## 핵심 디자인: 관심사 분리 (Separation of Concerns)

이 프레임워크의 핵심은 **실험 관련 코드**와 **범용 프레임워크 코드**를 명확히 분리하는 것입니다.

-   `hpo_framework/`: 일반적인 HPO 로직을 담고 있는 재사용 가능한 패키지입니다. 이 코드는 특정 모델이나 데이터에 대해 알지 못합니다.
-   `experiments/`: 사용자가 자신의 실험을 정의하는 공간입니다. `BaseExperimentHandler`를 상속받아 모델, 데이터, 학습 로직 등을 구현합니다.
-   `config.yaml`: 실험에 필요한 모든 파라미터(정적, 최적화 대상)를 코드 수정 없이 관리합니다.

### 프로젝트 구조

```
.
├── config.yaml                 # 실험 설정 파일
├── run_experiment.py           # 주 실행 스크립트
├── experiments/                # 사용자 정의 실험 디렉토리
│   ├── __init__.py
│   └── linear_regression.py    # 예제: 선형 회귀 실험 핸들러
└── hpo_framework/              # 재사용 가능한 범용 HPO 프레임워크
    ├── __init__.py
    ├── base_handler.py         # 실험 핸들러의 인터페이스(ABC)를 정의
    ├── runner.py               # Optuna Study 생성 및 실행
    ├── trial_executor.py       # Ignite를 사용해 단일 trial 실행
    └── param_sampler.py        # YAML에서 파라미터 샘플링
```

### 주요 특징

-   **명확한 계약 (Contract)**: `hpo_framework/base_handler.py`는 추상 기본 클래스(ABC)를 통해 사용자가 구현해야 할 메서드(`get_model`, `get_dataloaders` 등)를 명시합니다. 이를 통해 프레임워크와 사용자 코드 간의 인터페이스가 명확해집니다.
-   **독립성 및 재사용성**: `hpo_framework/trial_executor.py`는 특정 모델, 데이터셋, 학습 로직에 종속되지 않습니다. 모든 필요한 컴포넌트를 `ExperimentHandler` 객체로부터 동적으로 받아 Ignite Engine을 구성하므로, 다양한 PyTorch 프로젝트에 쉽게 재사용될 수 있습니다.
-   **간결성 및 응집도**: `run_experiment.py`는 단순히 `ExperimentHandler` 클래스 타입만 인자로 받아 실행됩니다. 실험의 모든 구성요소가 하나의 핸들러 클래스 안에서 관리되어 응집도가 높고 코드가 간결합니다.

---

## 새로운 실험 추가 방법

1.  `experiments/` 디렉토리 안에 `my_new_experiment.py`와 같은 새 파이썬 파일을 생성합니다.
2.  해당 파일에서 `hpo_framework.base_handler.BaseExperimentHandler`를 상속받는 `MyNewExperimentHandler` 클래스를 정의합니다.
3.  `BaseExperimentHandler`에 정의된 모든 추상 메서드들(`get_model`, `get_dataloaders`, `get_optimizer`, `get_train_step_fn` 등)을 자신의 실험에 맞게 구현합니다.
4.  `run_experiment.py` 파일에서 `from experiments.my_new_experiment import MyNewExperimentHandler`와 같이 핸들러를 임포트하고, `run_hpo` 함수의 `handler_class` 인자로 전달합니다.
5.  `config.yaml` 파일을 새로운 실험에 맞게 수정합니다.

---

## 실행 방법

#### 1. 라이브러리 설치

```bash
pip install torch pyyaml optuna ignite sqlalchemy
```

#### 2. HPO 실행

아래 명령어를 통해 선형 회귀 예제에 대한 하이퍼파라미터 최적화를 시작합니다.

```bash
python run_experiment.py
```