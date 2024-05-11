#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_1_X_2024-04-29-1100
cmsenv
cd - 

cat  <<EOF >>SingleElectron.py
from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper 
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService) 
randSvc.populate() 

process.source.firstLuminosityBlock = cms.untracked.uint32($INDEX)

process.generator.PGunParameters.ParticleID = cms.vint32($ParticleId)
EOF
cmsRun SingleElectron.py

cat  <<EOF >>step2.py
from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper 
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService) 
randSvc.populate() 
EOF
cmsRun step2.py

mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/$VERSION
mv step2.root /grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/$VERSION/step2_$INDEX.root