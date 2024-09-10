
## Setup
To make CMSSW find the models :

~~~bash
cd $LOCALRT/external/$SCRAM_ARCH
mkdir -p data/RecoHGCal/TICL/data
cd data/RecoHGCal/TICL/data
ln -s ../../../../../../../trained-dnn/current/ tf_models
~~~

Put the models in `trained-dnn/current`

For automatic inference during hyperparameter search : 
~~~
ln -s /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/models models_hyperparam_search
~~~

## Working points list
 - supercls_v2_May16_22-18-09.onnx : hyperparams-v7, best WP for energy resolution : 0.5, best for gsf efficiency : 0.05, chosen WP 0.3
 - supercls_v2_Apr05_14-48.onnx (-> supercls_v2p1) : best WP 0.3 ?

## Regression
` cp /grid_mnt/data__data.polcms/cms/sghosh/SUPERCLUSTERINGDATA/models/regressor.onnx /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_1_X_2024-06-12-2300/external/slc7_amd64_gcc12/data/RecoHGCal/TICL/data/superclustering/regression_v1.onnx`