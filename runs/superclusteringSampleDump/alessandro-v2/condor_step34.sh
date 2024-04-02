#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_0_0_pre0
cmsenv
cd - 

cat  <<EOF >>step3_TICLv5_superclsSampleDumper.py

process.source.fileNames = cms.untracked.vstring('file:$INPUT_FOLDER/step2_$INDEX.root')


EOF

#sed -i '/# Schedule definition/a from RecoHGCal.TICL.customiseForTICLv5_cff import customiseForTICLv5; process = customiseForTICLv5(process, True, enableSuperclusteringDNN=True)' step3.py

cmsRun step3_TICLv5_superclsSampleDumper.py


set +e
mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/$VERSION/
mv superclsDumper.root /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/$VERSION/superclsDumper_$INDEX.root

