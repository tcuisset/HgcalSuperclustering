#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_0_0_pre0
cmsenv
cd - 

cat  <<EOF >>SingleElectronFlatPt2To100_cfi_GEN_SIM.py
from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randSvc.populate()
EOF
cmsRun SingleElectronFlatPt2To100_cfi_GEN_SIM.py


cat  <<EOF >>SingleElectronFlatPt2To100_cfi_GEN_SIM.py
from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randSvc.populate()
EOF
cmsRun step2_DIGI_L1TrackTrigger_L1_DIGI2RAW_HLT_PU.py

mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/$VERSION
mv step2.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/$VERSION/step3_$INDEX.root