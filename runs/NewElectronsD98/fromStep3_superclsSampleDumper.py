# Starts from step3 file (FEVTDEBUG), runs SuperclusteringDNNSampleDumper

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5, ticl_v5_mustache

process = cms.Process('RECOsc',Phase2C17I13M9,ticl_v5,ticl_v5_mustache)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D98Reco_cff') # needed for TICLDumper
process.load('Configuration.StandardSequences.MagneticField_cff') # needed for TICLDumper
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step3.root'),
    secondaryFileNames = cms.untracked.vstring(),
    # inputCommands=cms.untracked.vstring(
    #     # remove old superclustering (before ticlv5) so we can reuse the old samples without redoing RAW2DIGI,RECO
    #     'keep *',
    #     'drop ticlTrackstersRefss_ticlTrackstersSuperclustering_superclusteredTracksters_RECO'
    # )
)

# Output
process.TFileService = cms.Service("TFileService",
                                    fileName=cms.string("superclsSampleDumper.root")
                                    )

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3b nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T25', '')

#Setup FWK for multithreaded
process.options.numberOfThreads = 8
process.options.numberOfStreams = 0


# the name of process translates into folder name inside TFileService root file
process.superclusteringSampleDumper = cms.EDAnalyzer("SuperclusteringSampleDumper")
process.superclusteringSampleDumper_step = cms.EndPath(process.superclusteringSampleDumper)


# process.Timing = cms.Service("Timing",
#   summaryOnly = cms.untracked.bool(False),
#   useJobReport = cms.untracked.bool(True)
# )
