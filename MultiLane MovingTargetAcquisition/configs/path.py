import sys
sys.path.append("..")
from utils.utils import path_to_cousin

PATH_AMORT_ROOT = path_to_cousin("amort", "")
PATH_AMORT_SIM_DATASET = f"{PATH_AMORT_ROOT}data/datasets/"
PATH_AMORT_MODEL = f"{PATH_AMORT_ROOT}data/amortizer_models"
PATH_AMORT_BOARD = f"{PATH_AMORT_ROOT}data/board"
PATH_AMORT_RESULT = f"{PATH_AMORT_ROOT}data/results"

PATH_EXP_DATA = path_to_cousin("experiment", "")