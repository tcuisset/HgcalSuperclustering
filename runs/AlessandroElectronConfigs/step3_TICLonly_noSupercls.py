# From step2 files run RAW2DIGI,RECO,RECOSIM then TICL validation then TICL dumper (do not save supercls output in dumper)
# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3_noSupercls -s RAW2DIGI,RECO,RECOSIM --conditions auto:phase2_realistic_T21 --datatier GEN-SIM-RECO,DQMIO --eventcontent FEVTDEBUGHLT,DQM --geometry Extended2026D88 --era Phase2C17I13M9 --filein file:step2.root --fileout file:step3.root --nThreads 20 --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process('RECO',Phase2C17I13M9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

### added
process.load('Configuration.StandardSequences.Validation_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step2.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    holdsReferencesToDeleteEarly = cms.untracked.VPSet(),
    makeTriggerResults = cms.obsolete.untracked.bool,
    modulesToCallForTryToContinue = cms.untracked.vstring(),
    modulesToIgnoreForDeleteEarly = cms.untracked.vstring(),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3_noSupercls nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.recosim_step = cms.Path(process.recosim)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.recosim_step,process.FEVTDEBUGHLToutput_step,process.DQMoutput_step)


################## Add TICL validation here
from RecoHGCal.TICL.ticlDumper_cfi import ticlDumper
# Validation
from Validation.HGCalValidation.HGCalValidator_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalRecHitMapProducer_cfi import hgcalRecHitMapProducer

# Automatic addition of the customisation function from RecoHGCal.Configuration.RecoHGCal_EventContent_cff
from RecoHGCal.Configuration.RecoHGCal_EventContent_cff import customiseHGCalOnlyEventContent

#### Fix for customizeTICLFromReco missing SimTracksterProducer
from RecoHGCal.TICL.SimTracksters_cff import *
process.ticlSimTrackstersTask = ticlSimTrackstersTask

process.TICL_ValidationProducers = cms.Task(process.hgcalRecHitMapProducer,
                                                process.lcAssocByEnergyScoreProducer,
                                                process.layerClusterCaloParticleAssociationProducer,
                                                process.scAssocByEnergyScoreProducer,
                                                process.layerClusterSimClusterAssociationProducer,
                                                process.simTsAssocByEnergyScoreProducer,
                                                process.simTracksterHitLCAssociatorByEnergyScoreProducer,
                                                process.tracksterSimTracksterAssociationLinking,
                                                process.tracksterSimTracksterAssociationPR,
                                                process.tracksterSimTracksterAssociationLinkingbyCLUE3D,
                                                process.tracksterSimTracksterAssociationPRbyCLUE3D,
                                                )
process.TICL_ValidationProducers.add(process.tracksterSimTracksterAssociationLinkingPU,
                                            process.tracksterSimTracksterAssociationPRPU)

process.TICL_Validator = cms.Task(process.hgcalValidator)

from Validation.Configuration.globalValidation_cff import globalPrevalidationTrackingOnly
process.globalPrevalidationTrackingOnly = globalPrevalidationTrackingOnly
process.TICL_Validation = cms.Path(
    process.globalPrevalidationTrackingOnly, # added by hand
    process.ticlSimTrackstersTask, # added by hand
    process.TICL_ValidationProducers,
                                    process.TICL_Validator
                                    )

### TICLDumper
process.ticlDumper = ticlDumper.clone(
    saveLCs=True,
    saveCLUE3DTracksters=True,
    saveTrackstersMerged=True,
    saveSimTrackstersSC=True,
    saveSimTrackstersCP=True,
    saveTICLCandidate=True,
    saveSimTICLCandidate=True,
    saveTracks=True,
    saveAssociations=True,
    saveSuperclustering=False,
    saveSuperclusteringDNNScore=False,
)
process.TFileService = cms.Service("TFileService",
                                    fileName=cms.string("dumper_noSupercls.root")
                                    )
process.ticlDumper_step = cms.EndPath(process.ticlDumper)

#process.superclusteringSampleDumper = cms.EDAnalyzer("SuperclusteringSampleDumper")
#process.superclusteringSampleDumper_step = cms.EndPath(process.superclusteringSampleDumper)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.recosim_step,
    ##### Added TICL here
    process.TICL_Validation,
    ######
    process.FEVTDEBUGHLToutput_step,process.DQMoutput_step,
    process.ticlDumper_step, # process.superclusteringSampleDumper_step ## added here
)

# call to customisation function customiseHGCalOnlyEventContent imported from RecoHGCal.Configuration.RecoHGCal_EventContent_cff
process = customiseHGCalOnlyEventContent(process)
process.FEVTDEBUGHLToutput.outputCommands.extend([ # for TICLDumper
    'keep *_generalTracks_*_*',
    'keep *_tofPID_t0_*',
    'keep *_mtdTrackQualityMVA_*_*',
    'keep *_tofPID_sigmat0_*',
])


from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads = 8
process.options.numberOfStreams = 0



# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
