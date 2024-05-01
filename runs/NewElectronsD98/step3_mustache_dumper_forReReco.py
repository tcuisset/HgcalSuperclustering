# step3 with ticlV5, mustache, superclsSampleDumper, ticlDumper, electronValidationDQM
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
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
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
process.prevalidation_step = cms.Path(process.baseCommonPreValidation)
process.prevalidation_step1 = cms.Path(process.globalPrevalidationTracking)
process.prevalidation_step2 = cms.Path(process.globalPrevalidationMuons)
process.prevalidation_step3 = cms.Path(process.globalPrevalidationJetMETOnly)
process.prevalidation_step4 = cms.Path(process.prebTagSequenceMC)
process.prevalidation_step5 = cms.Path(process.produceDenoms)
process.prevalidation_step6 = cms.Path(process.globalPrevalidationHCAL)
process.prevalidation_step7 = cms.Path(process.globalPrevalidationHGCal)
process.prevalidation_step8 = cms.Path(process.prevalidation)
process.validation_step = cms.EndPath(process.baseCommonValidation)
process.validation_step1 = cms.EndPath(process.globalValidationTrackingOnly)
process.validation_step2 = cms.EndPath(process.globalValidationMuons)
process.validation_step3 = cms.EndPath(process.globalValidationJetMETonly)
process.validation_step4 = cms.EndPath(process.electronValidationSequence)
process.validation_step5 = cms.EndPath(process.photonValidationSequence)
process.validation_step6 = cms.EndPath(process.bTagPlotsMCbcl)
process.validation_step7 = cms.EndPath()
process.validation_step8 = cms.EndPath(process.globalValidationHCAL)
process.validation_step9 = cms.EndPath(process.globalValidationHGCal)
process.validation_step10 = cms.EndPath(process.globalValidationMTD)
process.validation_step11 = cms.EndPath(process.validationECALPhase2)
process.validation_step12 = cms.EndPath(process.trackerphase2ValidationSource)
process.validation_step13 = cms.EndPath(process.validation)
process.dqmoffline_step = cms.EndPath(process.DQMOfflineBeam)
process.dqmoffline_1_step = cms.EndPath(process.DQMOfflineTracking)
process.dqmoffline_2_step = cms.EndPath(process.DQMOuterTracker)
process.dqmoffline_3_step = cms.EndPath(process.DQMOfflineTrackerPhase2)
process.dqmoffline_4_step = cms.EndPath(process.DQMOfflineMuon)
process.dqmoffline_5_step = cms.EndPath(process.DQMOfflineHcal)
process.dqmoffline_6_step = cms.EndPath(process.DQMOfflineHcal2)
process.dqmoffline_7_step = cms.EndPath(process.DQMOfflineEGamma)
process.dqmoffline_8_step = cms.EndPath(process.DQMOfflineL1TPhase2)
process.dqmoffline_9_step = cms.EndPath(process.HLTMonitoring)
process.dqmofflineOnPAT_step = cms.EndPath(process.PostDQMOffline)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# TICL dumper + superclsSampleDumper
process.TFileService = cms.Service("TFileService",
                                    fileName=cms.string("dumper.root")
                                    )
process.load("RecoHGCal.TICL.ticlDumper_cff")
# remove PU associator as that takes forever and we don't use it
del process.tracksterSimTracksterAssociationLinkingPU
process.ticlDumper.associators.remove(next(x for x in process.ticlDumper.associators if x.suffix == cms.string("PU")))
process.ticlDumper_step = cms.EndPath(process.ticlDumper)

process.load("RecoHGCal.TICL.superclusteringSampleDumper_cfi")
process.superclusteringSampleDumper_step = cms.EndPath(process.superclusteringSampleDumper)

# needs tracksterSimTracksterAssociationLinkingPU and is slow
del process.hgcalValidator

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.recosim_step,
    process.prevalidation_step,
    process.prevalidation_step1,process.prevalidation_step2,process.prevalidation_step3,process.prevalidation_step4,process.prevalidation_step5,process.prevalidation_step6,process.prevalidation_step7,process.prevalidation_step8,
    process.validation_step,
    #process.validation_step1,process.validation_step2,process.validation_step3,process.validation_step4,process.validation_step5,process.validation_step6,process.validation_step7,process.validation_step8,
    process.validation_step9,
    #process.validation_step10,process.validation_step11,process.validation_step12,process.validation_step13,
    #process.dqmoffline_step,
    #process.dqmoffline_1_step,process.dqmoffline_2_step,process.dqmoffline_3_step,process.dqmoffline_4_step,process.dqmoffline_5_step,process.dqmoffline_6_step,
    process.dqmoffline_7_step,
    #process.dqmoffline_8_step,process.dqmoffline_9_step,process.dqmofflineOnPAT_step,
    process.FEVTDEBUGHLToutput_step,process.DQMoutput_step,process.ticlDumper_step, process.superclusteringSampleDumper_step)
process.schedule.associate(process.patTask)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

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
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC 

#call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllMC(process)

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
    'keep *_tracksterSimTracksterAssociationLinkingSuperclustering_*_*', # for dnn only
    'keep *_tracksterSimTracksterAssociationPRSuperclustering_*_*', 

    'keep *_ticlCandidate_*_*',
    'keep *_pfTICL_*_*',

    # content for electron superclustering studies
    # "keep *_particleFlowRecHitHGC_*_*", # needs to be rereco, only there for studies
    # "keep *_particleFlowClusterHGCal_*_*", #rereco
    # "keep *_particleFlowSuperClusterHGCal_*_*", # also needed by ElectronMcSignalValidator to determine if electronCore are endcap or barrel

    "keep *_ecalDrivenElectronSeeds_*_*",
    "keep *_mergedSuperClustersHGC_*_*",

    "keep *_electronMergedSeeds_*_*",
    "keep *_photonCoreHGC_*_*",
    "keep *_photonsHGC_*_*",
    "keep *_electronCkfTrackCandidates_*_*",
    "keep *_electronGsfTracks_*_*",
    # "keep *_pfTrack_*_*",
    "keep *_pfTrackElec_*_*",
    # "keep *_particleFlowBlock_*_*",

    "keep *_ecalDrivenGsfElectronsHGC_*_*",
    "keep *_cleanedEcalDrivenGsfElectronsHGC_*_*",
    "keep *_patElectronsHGC_*_*",

    # Content for reRECO
    # for initialStepSeeds (needed because of dataforamts issue with ecalDrivenElectronSeeds)
    # ecalDrivenElectronSeeds needs initialStepSeeds (vector<TrajectorySeed>), which has a RecHitContainer which is a list of BaseTrackerRecHit (inherits TrackingRecHit)
    # TrackingRecHit has memeber "const GeomDet* m_det" (from Geometry/CommonTopologies/interface/GeomDet.h) which is not a DataFormat so does not get saved
    # Then ElectronSeedProducer -> ElectronSeedGenerator -> PixelHitMatcher::operator() -> BaseTrackerRecHit::globalPosition() -> BaseTrackerRecHit::surface() -> BaseTrackerRecHit.m_det-> crashes
    # "keep *_siPixelClusterShapeCache_*_RECO",
    # "keep *_initialStepHitQuadruplets_*_RECO", # not persistent !
    
    # to solve these tracker issues we rerun tracker each time
    # needed for reco_trackerOnly
    'keep *_simSiPixelDigis_*_HLT',
    'keep *_mix_Tracker_*',
    'keep *_initialStepSelector_*_*',
    'keep *_lowPtTripletStepTracks_*_RECO',
    'keep *_firstStepPrimaryVertices_*_RECO',
    # 'keep *_ak4CaloJetsForTrk_*_RECO',
    # 'keep *_muonSeededTracksInOut_*_RECO',
    #'keep *_hcalDigis_*_*',



    # for ecalDrivenElectronSeeds
    'keep *_HGCalRecHit_*_*',
    "keep *_hbhereco_*_*", # for some reason we need HCAL hits
    # "keep *_initialStepSeeds_*_RECO", # does not work because part of it is not persistent
    "keep *_highPtTripletStepSeeds_*_RECO",
    "keep *_tripletElectronSeeds_*_RECO",
    "keep *_particleFlowSuperClusterECAL_*_RECO", # need both SuperCluster and BasicCluster (otherwise ref error)
    #"keep *_particleFlowSuperClusterECAL_particleFlowBasicClusterECALBarrel_RECO",
    "keep *_offlinePrimaryVerticesWithBS_*_RECO",
    "keep *_offlineBeamSpot_*_*",

    # for ecalDrivenGsfElectronCoresHGC
    # (none)


    # for muonSeededSeedsInOut/OutIn 
    # "keep *_muonSeededSeedsInOut_*_RECO", # for generalTracks rereco
    # "keep *_muonSeededSeedsOutIn_*_RECO",
    "keep *_earlyMuons_*_RECO",
    'keep *_earlyGeneralTracks_*_RECO', # for muons out of earlyMuons read from RECO
    'keep *_standAloneMuons_UpdatedAtVtx_RECO',
    'keep *_siPixelClusters_*_*',
    'keep *_siPhase2Clusters_*_RECO',


    # for earlyMuons
    # 'keep *_siPixelClusters_*_*',
    # 'keep *_rpcRecHits_*_*',
    # 'keep *_standAloneMuons_UpdatedAtVtx_RECO',
    # 'keep *_glbTrackQual_*_RECO',

    # for trackerDrivenElectronSeeds
    # needs generalTracks but rereco as it needs transient stuff
    # "keep *_particleFlowClusterHCAL_*_RECO", #rereco
    # "keep *_particleFlowClusterECAL_*_RECO",
    # "keep *_particleFlowClusterPS_*_RECO",


    # for electronMergedSeeds
    "keep *_trackerDrivenElectronSeeds_SeedsForGsf_RECO",

    # for electronGsfTracks
    # (none)

    # for pfTrack
    #"keep *_generalTracks_*_RECO",
    "keep *_muons1stStep_*_RECO",

    # for particleFlowClusterECAL
    "keep *_addPileupInfo_bunchSpacing_*",
    "keep *_offlinePrimaryVertices4D_*_*",
    "keep *_ecalDetailedTimeRecHit_EcalRecHitsEB_*",
    # "keep *_ecalBarrelClusterFastTimer_PerfectResolutionModelResolution_RECO", # rereco
    # "keep *_ecalBarrelClusterFastTimer_PerfectResolutionModel_RECO",
    # "keep *_particleFlowClusterPS_*_RECO",
    # "keep *_particleFlowTimeAssignerECAL_*_RECO",

    # for pfTrackElec
    "keep *_pfConversions_*_RECO",
    "keep *_pfDisplacedTrackerVertex_*_RECO",
    # "keep *_particleFlowRecHitHGC_*_RECO", # we need to rereco these as we use transient members (caloCell)
    # "keep *_particleFlowRecHitECAL_*_RECO",

    # for ecalDrivenGsfElectronsHGC
    "keep *_ecalRecHit_EcalRecHitsEB_*", 
    "keep *_ecalRecHit_EcalRecHitsEE_*", # not actually used but still consumes it
    "keep *_generalTracks_*_RECO",
    "keep *_offlinePrimaryVertices_*_RECO",
    # "keep *_particleFlowClusterHCAL_*_RECO", # rereco
    # "keep *_particleFlowClusterECAL_*_RECO",
    "keep *_allConversions_*_RECO",

    # for photonCoreHGC
    # (none)

    # for photonsHGC
    "keep *_ecalRecHit_EcalRecHitsEB_*", 
    "keep *_ecalRecHit_EcalRecHitsEE_*", # not actually used but still consumes it
    "keep *_generalTracks_*_RECO",
    # "keep *_particleFlowClusterECAL_*_RECO", # rereco
    # "keep *_particleFlowClusterHCAL_*_RECO",
    "keep *_ecalPreshowerRecHit_EcalRecHitsES_RECO",
    "keep *_ecalPreshowerDigis_*_*",
    "keep *_offlinePrimaryVerticesWithBS_*_RECO",

    # for DQM electron validation
    'keep *_gedGsfElectrons_*_RECO',
    'keep *_gedGsfElectronCores_*_RECO',
    'keep *_ak4GenJets_*_HLT',
    "keep *_offlineBeamSpot_*_*",
    "keep *_particleFlowClusterHGCal_*_RECO", #get rerecoed but still need it as edm::Ref for ElectronMcSignalValidator
    "keep *_particleFlowSuperClusterHGCal_*_RECO", # also needed by ElectronMcSignalValidator to determine if electronCore are endcap or barrel
    "keep *_particleFlowClusterECAL_*_RECO",
    "keep *_particleFlowSuperClusterECAL_*_RECO", # need both SuperCluster and BasicCluster (otherwise ref error)
    "keep *_particleFlowEGamma_*_RECO", # for accessing gedGsfElectrons embedded refs

)

# process.Timing = cms.Service("Timing",
#   summaryOnly = cms.untracked.bool(False),
#   useJobReport = cms.untracked.bool(True)
# )
