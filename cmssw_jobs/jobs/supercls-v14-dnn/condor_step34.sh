#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_0_0_pre0
cmsenv
cd - 

cat  <<EOF >>step3_TICLv5_DNN_reduced.py

process.source.fileNames = cms.untracked.vstring('file:$INPUT_FOLDER/step2_$INDEX.root')

# process.ticlTracksterLinksSuperclustering.linkingPSet.onnxModelPath = cms.FileInPath('RecoHGCal/TICL/data/tf_models/supercls_v2.onnx')
process.ticlTracksterLinksSuperclustering.linkingPSet.onnxModelPath = cms.FileInPath('RecoHGCal/TICL/data/tf_models/$MODEL_FILENAME')
process.ticlTracksterLinksSuperclustering.linkingPSet.nnWorkingPoint = cms.double($DNN_WP)

EOF

cmsRun -n 8 step3_TICLv5_DNN_reduced.py

set +e
mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/$VERSION/
mv step3.root /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/$VERSION/step3_DNN_$INDEX.root

