# Full training
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs
mamba activate ticlRegression-gpu

python3 -m dnn_training.trainer -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v13/superclsDumper_\*.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/full -D cuda:1 -e 100


#testing single sample
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs
mamba activate ticlRegression-gpu

python3 -m dnn_training.trainer -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v13/superclsDumper_1.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/test -D cuda:1 -e 2



mamba activate ticlRegression-gpu
tensorboard --logdir /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/full