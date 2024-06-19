#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_1_X_2024-05-15-1100
cmsenv
cd - 

cat  <<EOF >>rereco_ticlDumper.py

process.source.fileNames = cms.untracked.vstring('file:$INPUT_FOLDER/step3_$INDEX.root')
process.ticlTracksterLinksSuperclustering.linkingPSet.onnxModelPath = cms.FileInPath('RecoHGCal/TICL/data/tf_models/$MODEL_FILENAME')
process.ticlTracksterLinksSuperclustering.linkingPSet.nnWorkingPoint = cms.double($DNN_WP)

EOF

cmsRun -n 8 rereco_ticlDumper.py

set +e
mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/$VERSION/
mv dumper.root /grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/$VERSION/dumper_$INDEX.root
