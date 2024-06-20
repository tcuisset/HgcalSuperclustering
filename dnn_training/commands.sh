# Full training
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs
mamba activate ticlRegression-gpu

python3 -m dnn_training.trainer -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v15-sampleDump/superclsDumper_\*.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/full-v2 -D cuda:1 -e 200 -b 512 --trainingLossType=binary
python3 -m dnn_training.trainer -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v15-sampleDump/superclsDumper_\*.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/full-v2 -D cuda:1 -e 200 -b 512 --trainingLossType=continuousAssociationScore

#testing single sample
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs
mamba activate ticlRegression-gpu

python3 -m dnn_training.trainer -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v13/superclsDumper_1.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/test -D cuda:1 -e 2


############## Tensorboard
mamba activate ticlRegression-gpu
tensorboard --logdir /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/full-v2






################ Hyperparameter scan
python3 -m dnn_training.hyperparameter_scan -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v15-sampleDump/superclsDumper_\*.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/hyperparams-v3 -D cuda:1 -n hp-v3 --createStudy
python3 -m dnn_training.hyperparameter_scan -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v15-sampleDump/superclsDumper_\*.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/hyperparams-v3 -D cuda:1 -n hp-v3

cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs
mamba activate ticlRegression-gpu
python3 -m dnn_training.hyperparameter_scan -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v15-sampleDump/superclsDumper_\*.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/hyperparams-v3 -D cuda:2 -n hp-v3

## dashboard
optuna-dashboard /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/hyperparams-v3/journal_optuna.log





######################### CMSSW inference


python3 -m dnn_training.cmsswInference --cmssw "/grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_1_X_2024-04-29-1100" \
        --cmsswConfig "/grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/NewElectronsD98/rereco_WPscan.py" --cmsswConfigHarvesting "/grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/NewElectronsD98/step4_WPscan.py" \
        --DQMForComparison "/grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/v2-65772-mustache/DQM/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root" \
        --cmsswInputFiles $(for i in `seq 1 30`; do echo -n "/grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/v2-65772-mustache/step3_${i}.root "; done) \
        --output "/grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/test/inference-tests/v1h" 
        --model "/grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/trained-dnn/current/supercls_v2_Apr05_14-48.onnx" 
        
        