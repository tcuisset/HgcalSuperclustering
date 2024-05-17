# step3 with ticlV5, mustache, ticlDumper
# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3_superclsDumper -s RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation,DQM:@phase2 --conditions auto:phase2_realistic_T25 --datatier GEN-SIM-RECO,DQMIO -n -1 --eventcontent FEVTDEBUGHLT,DQM --geometry Extended2026D98 --era Phase2C17I13M9 --procModifiers ticl_v5 --filein file:step2.root --fileout file:step3.root --nThreads 8
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5, ticl_v5_mustache

process = cms.Process('RECO',Phase2C17I13M9,ticl_v5,ticl_v5_mustache)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D98Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
# process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
#process.load('Configuration.StandardSequences.PATMC_cff')
#process.load('Configuration.StandardSequences.Validation_cff')
# process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
# process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

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
    annotation = cms.untracked.string('step3_superclsDumper nevts:-1'),
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

# process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
#     dataset = cms.untracked.PSet(
#         dataTier = cms.untracked.string('DQMIO'),
#         filterName = cms.untracked.string('')
#     ),
#     fileName = cms.untracked.string('file:step3_inDQM.root'),
#     outputCommands = process.DQMEventContent.outputCommands,
#     splitLevel = cms.untracked.int32(0)
# )

# Additional output definition

# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T25', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.recosim_step = cms.Path(process.recosim)

process.load("Validation.RecoTrack.TrackValidation_cff")
process.load("Validation.Configuration.hgcalSimValid_cff")
process.prevalidation_hgcal = cms.Sequence(process.tracksValidation, process.hgcalAssociators, process.ticlSimTrackstersTask)
process.prevalidation_hgcal_step = cms.Path(process.prevalidation_hgcal)

process.load("SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi")
# process.load("Validation.RecoEgamma.egammaValidation_cff")
process.prevalidation_egamma_step = cms.Path(process.simHitTPAssocProducer)
# process.validation_egamma_step = cms.Path(process.egammaValidation)

process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
# process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# TICL dumper + superclsSampleDumper
process.TFileService = cms.Service("TFileService",
                                    fileName=cms.string("dumper.root")
                                    )
process.load("RecoHGCal.TICL.ticlDumper_cff")
# remove PU associator as that takes forever and we don't use it
del process.tracksterSimTracksterAssociationLinkingPU
process.ticlDumper.associators.remove(next(x for x in process.ticlDumper.associators if x.suffix == cms.string("PU")))
process.ticlDumper_step = cms.EndPath(process.ticlDumper)

# process.load("RecoHGCal.TICL.superclusteringSampleDumper_cfi")
# process.superclusteringSampleDumper_step = cms.EndPath(process.superclusteringSampleDumper)

# needs tracksterSimTracksterAssociationLinkingPU and is slow
try:
    del process.hgcalValidator
except KeyError: pass
try:
    del process.trackValidator
except KeyError: pass
del process.trackValidatorGsfTracks
del process.trackValidatorConversion
del process.trackValidatorTPPtLess09
del process.trackValidatorTPEtaGreater2p7


# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.recosim_step,
    process.prevalidation_hgcal_step,process.prevalidation_egamma_step,
    #process.validation_egamma_step,
    process.FEVTDEBUGHLToutput_step,process.ticlDumper_step,# process.superclusteringSampleDumper_step
    )
# process.schedule.associate(process.patTask)
# from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
# associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads = 8
process.options.numberOfStreams = 0

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
# from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC 

# #call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
# process = miniAOD_customizeAllMC(process)

# End of customisation functions

# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

# HGCAL content + TICL + stuff for reRECO
process.FEVTDEBUGHLToutput.outputCommands = cms.untracked.vstring(
    'keep *_genParticles_*_*',

    # HGCAL event content
    'keep *_HGCalRecHit_*_*',
    'keep *_hgcalMergeLayerClusters_*_*',
    'keep CaloParticles_mix_*_*',
    'keep SimClusters_mix_*_*',
    'keep recoTracks_generalTracks_*_*',
    'keep recoTrackExtras_generalTracks_*_*',
    'keep SimTracks_g4SimHits_*_*',
    'keep SimVertexs_g4SimHits_*_*',
    'keep *_layerClusterSimClusterAssociationProducer_*_*',
    'keep *_layerClusterCaloParticleAssociationProducer_*_*',
    'keep *_randomEngineStateProducer_*_*',


    # TICL event content
    'keep *_ticlTrackstersCLUE3DHigh_*_RECO',

    'keep *_layerClusterSimTracksterAssociationProducer_*_*',
    'keep *_tracksterSimTracksterAssociationLinking_*_*',
    'keep *_tracksterSimTracksterAssociationPR_*_*', 
    # 'keep *_tracksterSimTracksterAssociationLinkingPU_*_*',
    # 'keep *_tracksterSimTracksterAssociationPRPU_*_*', 
    'keep *_tracksterSimTracksterAssociationLinkingbyCLUE3D_*_*',
    'keep *_tracksterSimTracksterAssociationPRbyCLUE3D_*_*', 
    
    'keep *_ticlSimTracksters_*_*',
    'keep *_ticlSimTICLCandidates_*_*',
    'keep *_ticlSimTrackstersFromCP_*_*',
    # 'keep *_tracksterSimTracksterAssociationLinkingbyCLUE3DEM_*_*',
    # 'keep *_tracksterSimTracksterAssociationPRbyCLUE3DEM_*_*',
    # 'keep *_tracksterSimTracksterAssociationLinkingSuperclustering_*_*', # for dnn only
    # 'keep *_tracksterSimTracksterAssociationPRSuperclustering_*_*', 

    'keep *_ticlCandidate_*_*',
    # 'keep *_pfTICL_*_*',

    "keep *_offlinePrimaryVerticesWithBS_*_RECO",


    "keep *_offlineBeamSpot_*_*",
)

process.Timing = cms.Service("Timing",
  summaryOnly = cms.untracked.bool(False),
  useJobReport = cms.untracked.bool(True)
)
