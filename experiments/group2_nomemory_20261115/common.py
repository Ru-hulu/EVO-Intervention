"""Isolated, suite-matched, no-history evaluation of the nine SOC35 adapters."""
from pathlib import Path
from experiments.soc35_finetune_20261114.common import read, save, sha, tensor_digest

ROOT=Path('/cache/cvpzhr/setup/group2-soc35-no-memory-20261115')
OUT=Path('/cache/cvpzhr/evaluations/group2-soc35-no-memory-20261115')
TRAIN=Path('/cache/cvpzhr/setup/soc35-procvlm-robometer-e1-20261114')
SOURCE=Path('/cache/cvpzhr/datasets/soc-two-per-task-perturbations-5fps-20261114/inputs.json')
SELECTION=Path('/cache/cvpzhr/setup/robometer-procvlm-soc35-cache-20261114/selection.json')
PYTHON='/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
MODELS=('procvlm-2b','forest','robometer-4b')
SUITES=('State','Order','Count')

def job_name(model,suite):return f'{model}-{suite}'
def result_path(model,key):return OUT/model/'episodes'/(key.replace('/','__')+'.json')
