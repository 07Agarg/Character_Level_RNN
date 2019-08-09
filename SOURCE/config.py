# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:55:43 2019

@author: ashima.garg
"""

import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')

NAME_FILE = os.path.join(DATA_DIR, 'names/')

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
NUM_EPOCHS = 100000
LEARNING_RATE = 0.005
BATCH_SIZE = 5

# HIDDEN UNITS
N_HIDDEN = 128
