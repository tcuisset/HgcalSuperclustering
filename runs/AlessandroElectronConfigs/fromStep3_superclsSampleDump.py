# Runs SuperclusteringSampleDumper from a step3 file 

import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process('RECOScDump',Phase2C17I13M9)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

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
                                    fileName=cms.string("superclsDump.root")
                                    )

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3b nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

#Setup FWK for multithreaded
process.options.numberOfThreads = 20
process.options.numberOfStreams = 0

# the name of process translates into folder name inside TFileService root file
process.superclusteringSampleDumper = cms.EDAnalyzer("SuperclusteringSampleDumper",
)

process.p = cms.EndPath(process.dumper)


process.Timing = cms.Service("Timing",
  summaryOnly = cms.untracked.bool(False),
  useJobReport = cms.untracked.bool(True)
)
