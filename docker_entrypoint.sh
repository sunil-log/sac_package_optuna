#!/bin/bash  
  
# ================  
# Environment Variable Setup  
# ================  
  
# 1. Matplotlib 임시 설정 디렉토리  
export MPLCONFIGDIR="/tmp/matplotlib-$USER"  
mkdir -p "$MPLCONFIGDIR"  
  
# 2. PYTHONPATH 설정  
# 추가할 경로들을 배열로 정의한다.  
declare -a paths_to_add=(  
    "/sac/src/sac_package_optuna"
    "/sac/src/sac_package_common"
    "/sac/src"
)  
  
# 배열의 원소들을 ':' 문자로 연결하여 하나의 문자열로 만든다.  
# IFS(Internal Field Separator)를 일시적으로 ':'로 변경하여 join 연산을 수행한다.  
joined_paths=$(IFS=:; echo "${paths_to_add[*]}")  
  
# 기존 PYTHONPATH 앞에 새로운 경로들을 추가한다.  
# 기존 PYTHONPATH가 비어있을 경우를 대비하여 :를 조건부로 추가한다.  
if [[ -n "$PYTHONPATH" ]]; then  
    export PYTHONPATH="${joined_paths}:${PYTHONPATH}"  
else  
    export PYTHONPATH="${joined_paths}"  
fi  
  
# 설정된 PYTHONPATH 확인 (디버깅용)  
echo "Updated PYTHONPATH: $PYTHONPATH"  
  
# ================  
# Execution  
# ================  
  
# cd /sac/src
cd /sac
  
# 메인 스크립트 실행  
# python run_experiment.py --experiment_name linear_regression
python run_experiment.py --experiment_name mlp_classifier

