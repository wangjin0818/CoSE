import os

PRE_TRAINED_VECTOR_PATH = 'pretrained_Vectors'
VECTOR_NAME = 'glove.840B.300d'

DATASET_PATH = 'corpus'
CKPTS_PATH = 'ckpts'
LOG_PATH = 'logs'

for directory in [PRE_TRAINED_VECTOR_PATH, CKPTS_PATH]:
    if not os.path.exists(directory): os.makedirs(directory)