#!/bin/bash

/home/llr/cms/cuisset/miniforge3/condabin/mamba activate /grid_mnt/data_cms_upgrade/cuisset/conda/envs/superclustering-dnn-cpu

# 12 hours is 43200 seconds
python3 -m dnn_training.hyperparameter_scan -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v15-sampleDump/superclsDumper_\*.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/hyperparams-v2 -D cpu -n hp-v2 --timeout 43200

