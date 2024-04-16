# Starts from alessandro's electrons step2, run part of TICLv5 with superclustering DNN. Only runs dumper with strictly needed stuff that is different with the DNN

# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3_TICLonly_v2 -s RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation --conditions auto:phase2_realistic_T21 --datatier GEN-SIM-RECO,DQMIO --eventcontent FEVTDEBUGHLT,DQM --geometry Extended2026D88 --era Phase2C17I13M9 --filein file:step2.root --fileout file:step3.root --nThreads 20 --no_exec
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
#process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
#process.load('Configuration.StandardSequences.PATMC_cff')
process.load('Configuration.StandardSequences.Validation_cff')
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
    annotation = cms.untracked.string('step3_TICLv5_dnn nevts:1'),
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
    splitLevel = cms.untracked.int32(0) ######### SPLITTING
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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.recosim_step = cms.Path(process.recosim)
# process.Flag_BadChargedCandidateFilter = cms.Path(process.BadChargedCandidateFilter)
# process.Flag_BadChargedCandidateSummer16Filter = cms.Path(process.BadChargedCandidateSummer16Filter)
# process.Flag_BadPFMuonDzFilter = cms.Path(process.BadPFMuonDzFilter)
# process.Flag_BadPFMuonFilter = cms.Path(process.BadPFMuonFilter)
# process.Flag_BadPFMuonSummer16Filter = cms.Path(process.BadPFMuonSummer16Filter)
# process.Flag_CSCTightHalo2015Filter = cms.Path(process.CSCTightHalo2015Filter)
# process.Flag_CSCTightHaloFilter = cms.Path(process.CSCTightHaloFilter)
# process.Flag_CSCTightHaloTrkMuUnvetoFilter = cms.Path(process.CSCTightHaloTrkMuUnvetoFilter)
# process.Flag_EcalDeadCellBoundaryEnergyFilter = cms.Path(process.EcalDeadCellBoundaryEnergyFilter)
# process.Flag_EcalDeadCellTriggerPrimitiveFilter = cms.Path(process.EcalDeadCellTriggerPrimitiveFilter)
# process.Flag_HBHENoiseFilter = cms.Path()
# process.Flag_HBHENoiseIsoFilter = cms.Path()
# process.Flag_HcalStripHaloFilter = cms.Path(process.HcalStripHaloFilter)
# process.Flag_METFilters = cms.Path(process.metFilters)
# process.Flag_chargedHadronTrackResolutionFilter = cms.Path(process.chargedHadronTrackResolutionFilter)
# process.Flag_ecalBadCalibFilter = cms.Path(process.ecalBadCalibFilter)
# process.Flag_ecalLaserCorrFilter = cms.Path(process.ecalLaserCorrFilter)
# process.Flag_eeBadScFilter = cms.Path()
# process.Flag_globalSuperTightHalo2016Filter = cms.Path(process.globalSuperTightHalo2016Filter)
# process.Flag_globalTightHalo2016Filter = cms.Path(process.globalTightHalo2016Filter)
# process.Flag_goodVertices = cms.Path(process.primaryVertexFilter)
# process.Flag_hcalLaserEventFilter = cms.Path(process.hcalLaserEventFilter)
# process.Flag_hfNoisyHitsFilter = cms.Path(process.hfNoisyHitsFilter)
# process.Flag_muonBadTrackFilter = cms.Path(process.muonBadTrackFilter)
# process.Flag_trackingFailureFilter = cms.Path(process.goodVertices+process.trackingFailureFilter)
# process.Flag_trkPOGFilters = cms.Path(~process.logErrorTooManyClusters)
# process.Flag_trkPOG_logErrorTooManyClusters = cms.Path(~process.logErrorTooManyClusters)
# process.Flag_trkPOG_manystripclus53X = cms.Path()
# process.Flag_trkPOG_toomanystripclus53X = cms.Path()
process.prevalidation_step = cms.Path(process.baseCommonPreValidation)
# process.prevalidation_step1 = cms.Path(process.globalPrevalidationTracking)
# process.prevalidation_step2 = cms.Path(process.globalPrevalidationMuons)
# process.prevalidation_step3 = cms.Path(process.globalPrevalidationJetMETOnly)
# process.prevalidation_step4 = cms.Path(process.prebTagSequenceMC)
# process.prevalidation_step5 = cms.Path(process.produceDenoms)
# process.prevalidation_step6 = cms.Path(process.globalPrevalidationHCAL)
process.prevalidation_step7 = cms.Path(process.globalPrevalidationHGCal)
process.prevalidation_step8 = cms.Path(process.prevalidation)
process.validation_step = cms.EndPath(process.baseCommonValidation)
# process.validation_step1 = cms.EndPath(process.globalValidationTrackingOnly)
# process.validation_step2 = cms.EndPath(process.globalValidationMuons)
# process.validation_step3 = cms.EndPath(process.globalValidationJetMETonly)
# process.validation_step4 = cms.EndPath(process.electronValidationSequence)
# process.validation_step5 = cms.EndPath(process.photonValidationSequence)
# process.validation_step6 = cms.EndPath(process.bTagPlotsMCbcl)
# process.validation_step7 = cms.EndPath(process.pfTauRunDQMValidation)
# process.validation_step8 = cms.EndPath(process.globalValidationHCAL)
process.validation_step9 = cms.EndPath(process.globalValidationHGCal)
# process.validation_step10 = cms.EndPath(process.globalValidationMTD)
# process.validation_step11 = cms.EndPath(process.globalValidationOuterTracker)
# process.validation_step12 = cms.EndPath(process.validationECALPhase2)
# process.validation_step13 = cms.EndPath(process.trackerphase2ValidationSource)
# process.validation_step14 = cms.EndPath(process.validation)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
#process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
from RecoHGCal.TICL.customiseForTICLv5_cff import customiseForTICLv5
#from RecoHGCal.Configuration.RecoHGCal_EventContent_cff import customiseHGCalOnlyEventContent
#process = customiseHGCalOnlyEventContent(process)
process = customiseForTICLv5(process, True, enableSuperclusteringDNN=True) #True to also run the TICLDumper

process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.recosim_step,
    #process.Flag_HBHENoiseFilter,process.Flag_HBHENoiseIsoFilter,process.Flag_CSCTightHaloFilter,process.Flag_CSCTightHaloTrkMuUnvetoFilter,process.Flag_CSCTightHalo2015Filter,process.Flag_globalTightHalo2016Filter,process.Flag_globalSuperTightHalo2016Filter,process.Flag_HcalStripHaloFilter,process.Flag_hcalLaserEventFilter,process.Flag_EcalDeadCellTriggerPrimitiveFilter,process.Flag_EcalDeadCellBoundaryEnergyFilter,process.Flag_ecalBadCalibFilter,process.Flag_goodVertices,process.Flag_eeBadScFilter,process.Flag_ecalLaserCorrFilter,process.Flag_trkPOGFilters,process.Flag_chargedHadronTrackResolutionFilter,process.Flag_muonBadTrackFilter,process.Flag_BadChargedCandidateFilter,process.Flag_BadPFMuonFilter,process.Flag_BadPFMuonDzFilter,process.Flag_hfNoisyHitsFilter,process.Flag_BadChargedCandidateSummer16Filter,process.Flag_BadPFMuonSummer16Filter,process.Flag_trkPOG_manystripclus53X,process.Flag_trkPOG_toomanystripclus53X,process.Flag_trkPOG_logErrorTooManyClusters,process.Flag_METFilters,
    #process.prevalidation_step, #process.prevalidation_step1,process.prevalidation_step2,process.prevalidation_step3,process.prevalidation_step4,process.prevalidation_step5,process.prevalidation_step6,
    #process.prevalidation_step7,
    # process.prevalidation_step8, 
    #process.validation_step,
    #process.validation_step1,process.validation_step2,process.validation_step3,process.validation_step4,process.validation_step5,process.validation_step6,process.validation_step7,
    #process.validation_step8, # removed to fix  No "CaloTPGRecord" record found in the EventSetup. in  module HcalDigisValidation/'AllHcalDigisValidation'
    #process.validation_step9,
    #process.validation_step10,process.validation_step11,process.validation_step12,process.validation_step13,process.validation_step14,
    process.FEVTDEBUGHLToutput_step,
    #process.DQMoutput_step
    )
#process.schedule.associate(process.patTask)
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

# # Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
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

# process.Timing = cms.Service("Timing",
#   summaryOnly = cms.untracked.bool(False),
#   useJobReport = cms.untracked.bool(True)
# )

# Mustache output commands
process.FEVTDEBUGHLToutput.outputCommands = cms.untracked.vstring(
    "keep *_genParticles_*_*",
    # HGCAL event content
    # 'keep *_HGCalRecHit_*_*',
    # 'keep *_hgcalMergeLayerClusters_*_*',
    # 'keep CaloParticles_mix_*_*',
    # 'keep SimClusters_mix_*_*',
    # 'keep recoTracks_generalTracks_*_*',
    # 'keep recoTrackExtras_generalTracks_*_*',
    # 'keep SimTracks_g4SimHits_*_*',
    # 'keep SimVertexs_g4SimHits_*_*',
    # 'keep *_layerClusterSimClusterAssociationProducer_*_*',
    # 'keep *_layerClusterCaloParticleAssociationProducer_*_*',
    # 'keep *_randomEngineStateProducer_*_*',
    # 'keep *_layerClusterSimTracksterAssociationProducer_*_*',
    # 'keep *_tracksterSimTracksterAssociationLinking_*_*',
    # 'keep *_tracksterSimTracksterAssociationPR_*_*', 
    # 'keep *_tracksterSimTracksterAssociationLinkingPU_*_*',
    # 'keep *_tracksterSimTracksterAssociationPRPU_*_*', 
    # 'keep *_tracksterSimTracksterAssociationLinkingbyCLUE3D_*_*',
    # 'keep *_tracksterSimTracksterAssociationPRbyCLUE3D_*_*', 
    
    # TICLv5 event content
    # 'keep *_ticlSimTracksters_*_*',
    # 'keep *_ticlSimTICLCandidates_*_*',
    # 'keep *_ticlSimTrackstersFromCP_*_*',
    # 'keep *_tracksterSimTracksterAssociationLinkingbyCLUE3DEM_*_*',
    # 'keep *_tracksterSimTracksterAssociationPRbyCLUE3DEM_*_*',
    # 'keep *_tracksterSimTracksterAssociationLinkingSuperclustering_*_*', # for dnn only
    # 'keep *_tracksterSimTracksterAssociationPRSuperclustering_*_*', 

    # additional content
    'keep *_ticlCandidate_*_*',

    # content for electron superclustering studies
    # "keep *_particleFlowRecHitHGC_*_*", # pretty large
    "keep *_particleFlowClusterHGCal_*_*",
    "keep *_particleFlowSuperClusterHGCal_*_*",

    # DNN output
    "keep *_ticlEGammaSuperClusterProducer_*_*",
    "keep *_ticlTracksterLinksSuperclustering_*_*",

    "keep *_ecalDrivenElectronSeeds_*_*",
    "keep *_mergedSuperClustersHGC_*_*",

    "keep *_electronMergedSeeds_*_*",
    "keep *_photonCoreHGC_*_*",
    "keep *_photonsHGC_*_*",
    "keep *_electronCkfTrackCandidates_*_*",
    "keep *_electronGsfTracks_*_*",
    # "keep *_pfTrack_*_*", # large
    "keep *_pfTrackElec_*_*",
    # "keep *_particleFlowBlock_*_*",

    "keep *_ecalDrivenGsfElectronsHGC_*_*",
    "keep *_ecalDrivenGsfElectronCoresHGC_*_*",
    "keep *_cleanedEcalDrivenGsfElectronsHGC_*_*",
    "keep *_patElectronsHGC_*_*",
)

# TICLDumper needs full hgcal validation so we can't really use it 
# process.ticlDumper.tracksterCollections = [
#     cms.PSet(
#         treeName=cms.string("trackstersCLUE3DEM"),
#         inputTag=cms.InputTag("ticlTrackstersCLUE3DEM")
#     ),
#     cms.PSet(
#         treeName=cms.string("trackstersSuperclustering"),
#         inputTag=cms.InputTag("ticlTracksterLinksSuperclustering")
#     ),
#     cms.PSet(
#         treeName=cms.string("simtrackstersCP"),
#         inputTag=cms.InputTag("ticlSimTracksters", "fromCPs"),
#         tracksterType=cms.string("SimTracksterCP")
#     ),
# ]
# process.ticlDumper.associators = []
# process.ticlDumper.saveLCs = False
# process.ticlDumper.saveTICLCandidate = False
# process.ticlDumper.saveSimTICLCandidate = False
# process.ticlDumper.saveTracks = False
# process.ticlDumper.saveSuperclustering = True
# process.ticlDumper.saveRecoSuperclusters = True

del process.ticlDumper
