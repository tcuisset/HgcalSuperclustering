#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_1_X_2024-04-18-2300
cmsenv
cd - 

cat  <<EOF >>step3_TICLv5_mustache.py
process.source.fileNames = cms.untracked.vstring('file:$INPUT_FOLDER/step2_$INDEX.root')
EOF

cmsRun -n 8 step3_TICLv5_mustache.py

set +e
mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/$VERSION/
mv step3.root /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/$VERSION/step3_mustache_$INDEX.root
mv step3_inDQM.root /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/$VERSION/step3_inDQM_mustache_$INDEX.root
mv histo.root /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/$VERSION/ticlDumper_mustache_$INDEX.root

