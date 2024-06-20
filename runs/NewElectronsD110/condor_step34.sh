#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/fixedReleases/06-12-2300-superclsWed19
cmsenv
cd - 

cat  <<EOF >>step3_dnn_dumper_forReReco.py
process.source.fileNames = cms.untracked.vstring('file:$INPUT_FOLDER/step2_$INDEX.root')
EOF

cmsRun -n 8 step3_dnn_dumper_forReReco.py

set +e
mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD110/$VERSION/
mv step3.root /grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD110/$VERSION/step3_$INDEX.root
mv step3_inDQM.root /grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD110/$VERSION/step3_inDQM_$INDEX.root
mv dumper.root /grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD110/$VERSION/dumper_$INDEX.root

