# run_experiment.py

import argparse
import os
import importlib.util
from typing import Callable, Dict, Any

from hpo_framework.runner import run_hpo


def load_experiment_session_fn(experiment_name: str) -> Callable[[Dict[str, Any]], float]:
	"""
    주어진 실험 이름에 해당하는 single_session 함수를 동적으로 로드합니다.

    Args:
        experiment_name (str): experiments 디렉토리 아래의 실험 폴더 이름.

    Returns:
        Callable: 로드된 single_session 함수.
    """
	module_path = os.path.join("experiments", experiment_name, "run.py")
	if not os.path.exists(module_path):
		raise FileNotFoundError(f"실험 파일을 찾을 수 없습니다: {module_path}")

	# 모듈을 파일 경로로부터 동적으로 로드
	spec = importlib.util.spec_from_file_location(f"experiments.{experiment_name}.run", module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"실험 모듈을 로드할 수 없습니다: {experiment_name}")

	experiment_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(experiment_module)

	# 모듈에서 single_session 함수 가져오기
	if not hasattr(experiment_module, "single_session"):
		raise AttributeError(f"'{module_path}'에 'single_session' 함수가 정의되지 않았습니다.")

	return experiment_module.single_session


def main():
	"""
    커맨드 라인 인자로 실험을 선택하고 HPO를 실행하는 메인 함수.
    """
	parser = argparse.ArgumentParser(
		description="범용 HPO 프레임워크 실행기.",
		formatter_class=argparse.RawTextHelpFormatter
	)
	parser.add_argument(
		"-e", "--experiment_name",
		type=str,
		required=True,
		help="실행할 실험의 이름 (예: linear_regression, mlp_classifier)"
	)
	args = parser.parse_args()

	# 실험 설정 파일 경로 구성
	config_path = os.path.join("experiments", args.experiment_name, "config.yaml")
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

	# 실험 함수 동적 로드
	session_fn = load_experiment_session_fn(args.experiment_name)

	# HPO 실행
	run_hpo(
		config_path=config_path,
		session_fn=session_fn
	)


if __name__ == "__main__":
	main()
