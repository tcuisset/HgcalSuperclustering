#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_1_X_2024-04-29-1100
cmsenv
cd - 

cat  <<EOF >>fromStep3_superclsSampleDumper.py

process.source.fileNames = cms.untracked.vstring('file:$INPUT_FOLDER/step3_$INDEX.root')
EOF

cmsRun -n 1 fromStep3_superclsSampleDumper.py

set +e
mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/$VERSION/
mv superclsSampleDumper.root /grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/$VERSION/superclsSampleDumper_$INDEX.root
