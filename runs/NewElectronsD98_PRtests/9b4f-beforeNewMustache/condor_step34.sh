#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd $CMSSW_BASE_PATH
cmsenv
cd - 

cat  <<EOF >>$PY_CONFIG_FILE
process.source.fileNames = cms.untracked.vstring('file:$INPUT_FOLDER/step2_$INDEX.root')
EOF

cmsRun -n 8 $PY_CONFIG_FILE

set +e
mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/superclsPR/NewElectronsD98/$VERSION/
#mv step3.root /grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/$VERSION/step3_$INDEX.root
mv step3.root /grid_mnt/data_cms_upgrade/cuisset/supercls/superclsPR/NewElectronsD98/$VERSION/step3_inDQM_$INDEX.root
#mv dumper.root /grid_mnt/data_cms_upgrade/cuisset/supercls/superclsPR/NewElectronsD98/$VERSION/dumper_$INDEX.root

