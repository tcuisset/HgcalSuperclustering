#!/bin/bash
set -e

export X509_USER_PROXY=/grid_mnt/vol_home/llr/cms/cuisset/.t3/proxy.cert

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_0_0_pre0
cmsenv
cd - 

cat  <<EOF >>step3_RAW2DIGI_RECO_RECOSIM_PAT_VALIDATION_DQM_PU.py
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
EOF

sed -i '/# Schedule definition/a from RecoHGCal.TICL.customiseForTICLv5_cff import customiseForTICLv5; process = customiseForTICLv5(process, True)' step3_RAW2DIGI_RECO_RECOSIM_PAT_VALIDATION_DQM_PU.py
cmsRun /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/diffEgamma/27034.0_TTbar_14TeV+2026D103PU/step3_RAW2DIGI_RECO_RECOSIM_PAT_VALIDATION_DQM_PU.py

cmsRun /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/diffEgamma/27034.0_TTbar_14TeV+2026D103PU/step4_HARVESTING_PU.py

mkdir -p /data_cms_upgrade/cuisset/supercls/diffEgamma/$VERSION
mv step2.root /data_cms_upgrade/cuisset/supercls/diffEgamma/$VERSION/step3_$INDEX.root