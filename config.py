import numpy as np

# DATASET PARAMETERS
DATASET = 'BRATS2020' # 'BRATS2018', 'BRATS2020', 'BRATS2021'; 'CC2024'
TRAIN_DIR = ''  # 'Modify data path'
VAL_DIR = TRAIN_DIR
TEST_DIR = TRAIN_DIR
TRAIN_LIST = ''
VAL_LIST = ''
TEST_LIST = ''

MODALITY = ['falir', 't1c', 't1', 't2'] # ['falir', 't1c', 't1', 't2']; ['t1ce', 't2', 't2fs', 'dwi']
CROP_SIZE = (80, 80, 80)

BATCH_SIZE = 4
NUM_WORKERS = 8
NUM_CLASSES = 4

# GENERAL
MODEL = 'mcbtformer'
PRETRAINED = False
PRETRAINEDPATH = ''
ALPHA = 1
RESUME = ''
NUM_SEGM_EPOCHS = 1000
ITER_PER_EPOCH = 150
RFSE = 0 # region fusion start epoch
RANDOM_SEED = 1024
NEEDVAL = False
VAL_EVERY = 200  # how often to record validation scores

# OPTIMISERS' PARAMETERS
LR = 2e-4  # TO FREEZE, PUT 0
WD = 1e-4  # TO FREEZE, PUT 0
WARMUP = 20