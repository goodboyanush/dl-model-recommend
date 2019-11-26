#!/usr/bin/env python
import os
catalog_home_path = os.environ.get('CATALOG_TEMP_DIR', '..')

nlds_data_path = catalog_home_path + "/data/nlds/"
models_path = catalog_home_path + "/models/"

model_params = {
    'workers': 4,
    'batchSize': 64,
    'size': 32,
    'inpSize': 32,
    'sampleSize': 200,
    'classificationType': 'LinearSVM',
    'daisyRadius': 5,
    'daisyRings': 1,
    'daisyHistograms': 6,
    'daisyOrientations': 8,
    'lbpWidth': 3,
    'lbpHeight': 3,
    'lbpP': 16,
    'lbpR': 2,
    'lbpMethod': 'Uniform',
    'hogBlockNorm': 'L2-Hys',
    'hogNumCells': 3,
    'hogCellSize': 8,
    'hogOrientations': 9
}