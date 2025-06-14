# run_experiment.py

# 재사용 가능한 프레임워크에서 runner를 임포트
from hpo_framework.runner import run_hpo

# 사용자가 정의한 실험 핸들러를 임포트
# 다른 실험을 실행하려면 이 부분만 수정하면 됩니다.
from experiments.linear_regression import LinearRegressionHandler


if __name__ == "__main__":
    # 설정 파일 경로
    CONFIG_PATH = "config.yaml"

    # 프레임워크의 run_hpo 함수에 설정 파일 경로와
    # 사용할 핸들러 클래스를 전달하여 실행
    run_hpo(
        config_path=CONFIG_PATH,
        handler_class=LinearRegressionHandler
    )
