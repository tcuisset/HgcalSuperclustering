
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_0_0_pre0
cmsenv
cd - 

cd /data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-dnn
cmsRun /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/diffEgamma/25102.0_SingleEFlatPt2To100+2026D98PU/step4_HARVESTING_PU_local.py

cd /data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-mustache
cmsRun /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/diffEgamma/25102.0_SingleEFlatPt2To100+2026D98PU/step4_HARVESTING_PU_local.py

cd /data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-mustache-noreg
cmsRun /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/diffEgamma/25102.0_SingleEFlatPt2To100+2026D98PU/step4_HARVESTING_PU_local.py



# COmapring two DQM files for differences
# dnn vs mustache
mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7/dnn-vs-mustache
(
    cd /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7/dnn-vs-mustache
    mkdir RelMonCompare
    mkdir RelMonCompareEgamma
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-mustache/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompare -CR --standalone --use_black_file -p --no_successes &
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-mustache/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompareEgamma -CR --standalone --use_black_file -p -d Egamma &
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-mustache/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompareEgamma -CR --standalone --use_black_file -p -d EgammaV &
)

mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7/dnn-vs-mustache-noreg
(
    cd /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7/dnn-vs-mustache-noreg
    mkdir RelMonCompare
    mkdir RelMonCompareEgamma
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-mustache-noreg/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompare -CR --standalone --use_black_file -p --no_successes &
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-mustache-noreg/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompareEgamma -CR --standalone --use_black_file -p -d Egamma &
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-mustache-noreg/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v3-1d9a7-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompareEgamma -CR --standalone --use_black_file -p -d EgammaV &
)