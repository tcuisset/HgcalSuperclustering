# Starts from step3 file (FEVTDEBUG), runs superclustering, then TICL dumper (supercls only) and SuperclusteringDNNSampleDumper

import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process('RECOsc',Phase2C17I13M9)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff') # needed for TICLDumper
process.load('Configuration.StandardSequences.MagneticField_cff') # needed for TICLDumper
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step3.root'),
    secondaryFileNames = cms.untracked.vstring()
)

# Output
process.TFileService = cms.Service("TFileService",
                                    fileName=cms.string("dumper.root")
                                    )

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3b nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

#Setup FWK for multithreaded
process.options.numberOfThreads = 20
process.options.numberOfStreams = 0


##### 
process.load("RecoHGCal.TICL.superclusteringProducer_cfi")
process.superclusteringProducer_step = cms.Path(process.superclusteringProducer)

from RecoHGCal.TICL.ticlDumper_cfi import ticlDumper
process.ticlDumper = ticlDumper.clone(
        saveLCs=False,
        saveCLUE3DTracksters=False,
        saveTrackstersMerged=False,
        saveSimTrackstersSC=False,
        saveSimTrackstersCP=False,
        saveTICLCandidate=False,
        saveSimTICLCandidate=False,
        saveTracks=False,
        saveAssociations=False,
        saveSuperclustering=True,
        saveSuperclusteringDNNScore=True,
    )
process.ticlDumper_step = cms.EndPath(process.ticlDumper)

# the name of process translates into folder name inside TFileService root file
process.superclusteringSampleDumper = cms.EDAnalyzer("SuperclusteringSampleDumper")
process.superclusteringSampleDumper_step = cms.EndPath(process.superclusteringSampleDumper)


# process.Timing = cms.Service("Timing",
#   summaryOnly = cms.untracked.bool(False),
#   useJobReport = cms.untracked.bool(True)
# )