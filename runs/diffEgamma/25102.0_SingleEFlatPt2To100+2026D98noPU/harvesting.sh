
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_0_0_pre0
cmsenv
cd - 

cd /data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7-dnn
cmsRun /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/diffEgamma/25102.0_SingleEFlatPt2To100+2026D98noPU/step4_local.py

cd /data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7-mustache
cmsRun /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/diffEgamma/25102.0_SingleEFlatPt2To100+2026D98noPU/step4_local.py

# COmapring two DQM files for differences
(
    mkdir /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7
    cd /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7
    mkdir RelMonCompare
    mkdir RelMonCompareEgamma
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7-mustache/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompare -CR --standalone --use_black_file -p --no_successes &
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7-mustache/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompareEgamma -CR --standalone --use_black_file -p -d Egamma &
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7-mustache/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompareEgamma -CR --standalone --use_black_file -p -d EgammaV &
)

# not tested :
mkdir -p /eos/user/t/tcuisset/www/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7-mustache-vs-dnn/
cp -r /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7/RelMonCompare /eos/user/t/tcuisset/www/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7-mustache-vs-dnn/
cp -r /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7/RelMonCompareEgamma /eos/user/t/tcuisset/www/supercls/diffEgamma/25102_SingleE_D98noPU/v3-1d9a7-mustache-vs-dnn/


