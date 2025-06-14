# 범용 하이퍼파라미터 최적화(HPO) 프레임워크 (Refactored)

이 프로젝트는 **Optuna**를 사용하여, 특정 훈련 프레임워크(예: PyTorch Ignite, Keras)에 종속되지 않는 범용 하이퍼파라미터 최적화(HPO) 프레임워크를 제공한다.

---

## 핵심 디자인: 명확한 책임 분리

이 프레임워크의 핵심은 HPO 로직과 실험(훈련) 로직을 완전히 분리하는 것이다.

-   `hpo_framework/`: 일반적인 HPO 로직을 담고 있는 재사용 가능한 패키지이다. 이 코드는 사용자의 모델이나 데이터, 훈련 방식에 대해 전혀 알지 못한다. 오직 `config.yaml`을 읽어 하이퍼파라미터를 샘플링하고, 사용자가 제공한 `session` 함수를 호출하여 점수를 받는 역할만 한다.
-   `experiments/`: 사용자가 자신의 실험을 `single_session` 함수로 정의하는 공간이다. 이 함수 안에서 사용자는 데이터 로딩, 모델 생성, 훈련, 평가 등 모든 과정을 완전히 제어할 수 있다.
-   `config.yaml`: 실험에 필요한 모든 파라미터(정적, 최적화 대상)를 코드 수정 없이 관리한다.
-   `run_experiment.py`: HPO를 시작하는 진입점 스크립트이다. 어떤 `session` 함수를 실행할지 결정한다.

---

## 프로젝트 구조

```text
.
├── config.yaml              # 실험 설정 파일
├── run_experiment.py        # 주 실행 스크립트
├── experiments/             # 사용자 정의 실험 디렉토리
│   └── linear_regression.py   # 예제: 선형 회귀 실험 함수
└── hpo_framework/           # 재사용 가능한 범용 HPO 프레임워크
    ├── runner.py            # Optuna Study 생성 및 실행
    └── param_sampler.py     # YAML에서 파라미터 샘플링
```

---

## 주요 특징

-   **최소한의 결합 (Loosely Coupled)**: `hpo_framework`는 `session_fn(config) -> score` 라는 간단한 함수 계약(function contract) 외에는 아무것도 요구하지 않는다. PyTorch, TensorFlow, Scikit-learn 등 어떤 라이브러리든 자유롭게 사용할 수 있다.
-   **완벽한 제어**: 사용자는 `single_session` 함수 내부의 훈련/평가 로직을 100% 직접 제어한다. 복잡한 추상 클래스를 상속받을 필요가 없다.
-   **단순성과 유연성**: 새로운 실험을 추가하는 것은 단순히 새로운 `single_session` 함수를 작성하고 `run_experiment.py`에서 이를 지정하는 것만으로 충분하다.

---

## 새로운 실험 추가 방법

1.  `experiments/` 디렉토리 안에 `my_new_experiment.py`와 같은 새 파이썬 파일을 생성한다.
2.  해당 파일에 `def single_session(cfg: dict) -> float:` 시그니처를 갖는 함수를 작성한다.
3.  함수 내에서 `cfg` 딕셔너리를 사용하여 하이퍼파라미터를 읽어오고, 데이터 로딩, 모델 생성, 훈련, 평가 로직을 모두 구현한다.
4.  함수의 마지막에는 Optuna가 최적화할 대상 점수(예: validation loss)를 반환한다.
5.  `run_experiment.py` 파일에서 `from experiments.my_new_experiment import single_session`과 같이 함수를 임포트하고, `run_hpo` 함수의 `session_fn` 인자로 전달한다.
6.  `config.yaml` 파일을 새로운 실험에 맞게 수정한다.

---

## 실행 방법

1.  **라이브러리 설치**
    ```bash
    # Ubuntu 기준
    pip install torch pyyaml optuna sqlalchemy
    ```

2.  **HPO 실행**
    아래 명령어를 통해 선형 회귀 예제에 대한 하이퍼파라미터 최적화를 시작한다.
    ```bash
    python run_experiment.py
    ```