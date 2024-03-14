#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_0_0_pre0
cmsenv
cd - 

cat  <<EOF >>step3.py
process.source.fileNames = cms.untracked.vstring('file:/grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/step2_$INDEX.root')

process.FEVTDEBUGHLToutput.outputCommands.extend([
    "keep *_ecalDrivenElectronSeeds_*_*",
    "keep *_particleFlowRecHitHGC_*_*",
    "keep *_particleFlowClusterHGCal_*_*",
    "keep *_particleFlowSuperClusterHGCal_*_*",
    "keep *_mergedSuperClustersHGC_*_*",
    "keep *_electronMergedSeeds_*_*",
    "keep *_photonCoreHGC_*_*",
    "keep *_photonsHGC_*_*",
    "keep *_electronCkfTrackCandidates_*_*",
    "keep *_electronGsfTracks_*_*",
    "keep *_pfTrack_*_*",
    "keep *_pfTrackElec_*_*",
    "keep *_particleFlowBlock_*_*",

    "drop l1t*_*_*_*",
    "drop EBDigiCollection_simEcalUnsuppressedDigis__HLT",
    "drop recoCaloClusters_hgcalMergeLayerClusters__HLT"
    ])

process.hgcalValidatorSequence.remove(process.hgcalValidatorv5)
process.hgcalTiclPFValidation.remove(process.ticlPFValidation)
process.hgcalValidation.remove(process.hgcalPFJetValidation)
EOF

sed -i '/# Schedule definition/a from RecoHGCal.TICL.customiseForTICLv5_cff import customiseForTICLv5; process = customiseForTICLv5(process, True, enableSuperclusteringDNN=True)' step3.py
cmsRun step3.py

cmsRun step4.py

set +e
mkdir -p /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/$VERSION/
mv step3.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/$VERSION/step3_$INDEX.root
mv step3_inMINIAODSIM.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/$VERSION/step3_inMINIAODSIM_$INDEX.root
mv step3_inDQM.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/$VERSION/step3_inDQM_$INDEX.root
mv histo.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/$VERSION/dumper_$INDEX.root
mv DQM_*.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98noPU/$VERSION/DQM_$INDEX.root
