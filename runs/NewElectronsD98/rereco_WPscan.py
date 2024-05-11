# ReRECO egamma chain + electron validation from step3 scanning different DNN WP and running chain with each WP
# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 -s RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM --conditions auto:phase2_realistic_T25 --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO -n 10 --eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM --geometry Extended2026D98 --era Phase2C17I13M9 --procModifiers ticl_v5 --filein file:step2.root --fileout file:step3.root --nThreads 10
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

process = cms.Process('reRECO',Phase2C17I13M9,ticl_v5)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D98Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
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
    fileNames = cms.untracked.vstring('file:step3.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),#'ProductNotFound'
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
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3_rereco.root'),
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

############# DNN

# #process.ticlTracksterLinksSuperclustering.linkingPSet.nnWorkingPoint = cms.double({dnn_wp})

# process.superclustering_step = cms.Path(process.ticlTracksterLinksSuperclustering + process.ticlEGammaSuperClusterProducer)
# # trackerDrivenElectronSeeds is needed to make ElectronSeed with detector information (transient)
# process.egamma_step = cms.Path(process.ecalDrivenElectronSeeds + process.trackerDrivenElectronSeeds + process.electronMergedSeeds + process.electronCkfTrackCandidates + process.electronGsfTracks + process.pfTrack + process.pfTrackElec + process.ecalDrivenGsfElectronCoresHGC + process.ecalDrivenGsfElectronsHGC + process.photonCoreHGC + process.photonsHGC)

process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# removing stuff that we don't want to rereco
del process.offlinePrimaryVertices
del process.offlinePrimaryVerticesWithBS
#del process.generalTracks # need rereco because of trackerDrivenElectronSeeds
del process.firstStepPrimaryVertices
del process.ecalPreshowerRecHit
del process.offlineBeamSpot
del process.earlyMuons
# del process.muonSeededSeedsInOut # used for generalTracks rereco, to avoid running whole muon reco we use the RECO for this
# del process.muonSeededSeedsOutIn
del process.rpcRecHits
process.reconstruction_trackingOnly_step = cms.Path(process.reconstruction_trackingOnly)

# Electron validation
process.load("Validation.RecoEgamma.electronValidationSequence_cff")
# process.electronValidationSequence_step = cms.Path(process.mergedSuperClustersHGC + process.electronValidationSequence)

del process.ecalDetailedTimeRecHit
# We need to rereco pfRecHits as we need CaloCell (transient) for pfTrackElec
process.pfRecHit_step = cms.Path(process.particleFlowRecHitHGC + process.particleFlowRecHitECAL +process.particleFlowRecHitPS+process.particleFlowClusterECALUncorrected+ process.particleFlowClusterPS +process.ecalBarrelClusterFastTimer+ process.particleFlowTimeAssignerECAL + process.particleFlowClusterECAL)
process.egamma_noDNN_step = cms.Path(process.trackerDrivenElectronSeeds)
# Schedule definition
process.schedule = cms.Schedule(process.pfRecHit_step,
                                process.reconstruction_trackingOnly_step, 
                                process.egamma_noDNN_step,
                                #process.superclustering_step,
                                #process.egamma_step,
                                #process.electronValidationSequence_step,
                                process.FEVTDEBUGHLToutput_step,
                                process.DQMoutput_step)
#process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.recosim_step,process.Flag_HBHENoiseFilter,process.Flag_HBHENoiseIsoFilter,process.Flag_CSCTightHaloFilter,process.Flag_CSCTightHaloTrkMuUnvetoFilter,process.Flag_CSCTightHalo2015Filter,process.Flag_globalTightHalo2016Filter,process.Flag_globalSuperTightHalo2016Filter,process.Flag_HcalStripHaloFilter,process.Flag_hcalLaserEventFilter,process.Flag_EcalDeadCellTriggerPrimitiveFilter,process.Flag_EcalDeadCellBoundaryEnergyFilter,process.Flag_ecalBadCalibFilter,process.Flag_goodVertices,process.Flag_eeBadScFilter,process.Flag_ecalLaserCorrFilter,process.Flag_trkPOGFilters,process.Flag_chargedHadronTrackResolutionFilter,process.Flag_muonBadTrackFilter,process.Flag_BadChargedCandidateFilter,process.Flag_BadPFMuonFilter,process.Flag_BadPFMuonDzFilter,process.Flag_hfNoisyHitsFilter,process.Flag_BadChargedCandidateSummer16Filter,process.Flag_BadPFMuonSummer16Filter,process.Flag_trkPOG_manystripclus53X,process.Flag_trkPOG_toomanystripclus53X,process.Flag_trkPOG_logErrorTooManyClusters,process.Flag_METFilters,process.prevalidation_step,process.prevalidation_step1,process.prevalidation_step2,process.prevalidation_step3,process.prevalidation_step4,process.prevalidation_step5,process.prevalidation_step6,process.prevalidation_step7,process.prevalidation_step8,process.prevalidation_step9,process.validation_step,process.validation_step1,process.validation_step2,process.validation_step3,process.validation_step4,process.validation_step5,process.validation_step6,process.validation_step7,process.validation_step8,process.validation_step9,process.validation_step10,process.validation_step11,process.validation_step12,process.validation_step13,process.validation_step14,process.dqmoffline_step,process.dqmoffline_1_step,process.dqmoffline_2_step,process.dqmoffline_3_step,process.dqmoffline_4_step,process.dqmoffline_5_step,process.dqmoffline_6_step,process.dqmoffline_7_step,process.dqmoffline_8_step,process.dqmoffline_9_step,process.dqmoffline_10_step,process.dqmofflineOnPAT_step,process.dqmofflineOnPAT_1_step,process.FEVTDEBUGHLToutput_step,process.MINIAODSIMoutput_step,process.DQMoutput_step)

# Adding the multiple egamma paths for each working point
import numpy as np
WPs = list(np.linspace(0., 0.1, 10, endpoint=False)) + list(np.linspace(0.1, 0.9, 8, endpoint=False)) + list(np.linspace(0.9, 1., 11))
#DNN_MODEL_PATH = ""
for wp in WPs:
    tag = "DNNWP" + str(wp).replace(".", "p")
    setattr(process, f"ticlTracksterLinksSuperclustering{tag}", process.ticlTracksterLinksSuperclustering.clone(linkingPSet=dict(onnxModelPath=cms.FileInPath(__DNN_MODEL_PATH__), nnWorkingPoint=cms.double(wp)))) # ,
    setattr(process, f"ticlEGammaSuperClusterProducer{tag}", process.ticlEGammaSuperClusterProducer.clone(ticlSuperClusters=cms.InputTag(f"ticlTracksterLinksSuperclustering{tag}")))
    setattr(process, f"ecalDrivenElectronSeeds{tag}", process.ecalDrivenElectronSeeds.clone(endcapSuperClusters=cms.InputTag(f"ticlEGammaSuperClusterProducer{tag}")))
    setattr(process, f"electronMergedSeeds{tag}", process.electronMergedSeeds.clone(EcalBasedSeeds=cms.InputTag(f"ecalDrivenElectronSeeds{tag}")))
    setattr(process, f"electronCkfTrackCandidates{tag}", process.electronCkfTrackCandidates.clone(src=cms.InputTag(f"electronMergedSeeds{tag}")))
    setattr(process, f"electronGsfTracks{tag}", process.electronGsfTracks.clone(src=cms.InputTag(f"electronCkfTrackCandidates{tag}")))
    setattr(process, f"pfTrack{tag}", process.pfTrack.clone(GsfTrackModuleLabel=cms.InputTag(f"electronGsfTracks{tag}")))
    setattr(process, f"pfTrackElec{tag}", process.pfTrackElec.clone(GsfTrackModuleLabel=cms.InputTag(f"electronGsfTracks{tag}"), PFRecTrackLabel=cms.InputTag(f"pfTrack{tag}")))
    setattr(process, f"ecalDrivenGsfElectronCoresHGC{tag}", process.ecalDrivenGsfElectronCoresHGC.clone(gsfPfRecTracks=cms.InputTag(f"pfTrackElec{tag}"), gsfTracks=cms.InputTag(f"electronGsfTracks{tag}")))
    setattr(process, f"ecalDrivenGsfElectronsHGC{tag}", process.ecalDrivenGsfElectronsHGC.clone(gsfElectronCoresTag=cms.InputTag(f"ecalDrivenGsfElectronCoresHGC{tag}"), gsfPfRecTracksTag=cms.InputTag(f"pfTrackElec{tag}"),
        seedsTag=cms.InputTag(f"ecalDrivenElectronSeeds{tag}")))
    setattr(process, f"photonCoreHGC{tag}", process.photonCoreHGC.clone(pixelSeedProducer=cms.InputTag(f"electronMergedSeeds{tag}"), scIslandEndcapProducer=cms.InputTag(f"ticlEGammaSuperClusterProducer{tag}")))
    setattr(process, f"photonsHGC{tag}", process.photonsHGC.clone(photonProducer=cms.InputTag(f"photonCoreHGC{tag}")))

    setattr(process, f"mergedSuperClustersHGC{tag}", process.mergedSuperClustersHGC.clone(src=cms.VInputTag(cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"), cms.InputTag(f"ticlEGammaSuperClusterProducer{tag}"))))
    #setattr(process, f"electronValidationSequence{tag}", process.electronValidationSequence.clone(photonProducer=cms.InputTag(f"photonCoreHGC{tag}")))
    setattr(process, f"electronMcSignalValidator{tag}", process.electronMcSignalValidator.clone(InputFolderName=cms.string(f"EgammaV{tag}/ElectronMcSignalValidator"), OutputFolderName=cms.string(f"EgammaV{tag}/ElectronMcSignalValidator"),
        electronCollectionEndcaps=cms.InputTag(f"ecalDrivenGsfElectronsHGC{tag}"), electronCoreCollectionEndcaps=cms.InputTag(f"ecalDrivenGsfElectronCoresHGC{tag}"),
        electronSeedCollection=cms.InputTag(f"electronMergedSeeds{tag}"), electronTrackCollection=cms.InputTag(f"electronGsfTracks{tag}")))
    setattr(process, f"electronMcSignalValidatorPt1000{tag}", process.electronMcSignalValidatorPt1000.clone(InputFolderName=cms.string(f"EgammaV{tag}/ElectronMcSignalValidatorPt1000"), OutputFolderName=cms.string(f"EgammaV{tag}/ElectronMcSignalValidatorPt1000"),
        electronCollectionEndcaps=cms.InputTag(f"ecalDrivenGsfElectronsHGC{tag}"), electronCoreCollectionEndcaps=cms.InputTag(f"ecalDrivenGsfElectronCoresHGC{tag}"),
        electronSeedCollection=cms.InputTag(f"electronMergedSeeds{tag}"), electronTrackCollection=cms.InputTag(f"electronGsfTracks{tag}")))
    setattr(process, f"electronMcFakeValidator{tag}", process.electronMcFakeValidator.clone(InputFolderName=cms.string(f"EgammaV{tag}/ElectronMcFakeValidator"), OutputFolderName=cms.string(f"EgammaV{tag}/ElectronMcFakeValidator"),
        electronCollectionEndcaps=cms.InputTag(f"ecalDrivenGsfElectronsHGC{tag}"), electronCoreCollectionEndcaps=cms.InputTag(f"ecalDrivenGsfElectronCoresHGC{tag}"),
        electronSeedCollection=cms.InputTag(f"electronMergedSeeds{tag}"), electronTrackCollection=cms.InputTag(f"electronGsfTracks{tag}")))

    setattr(process, f"DNN_task{tag}", cms.Task(
        getattr(process, f"ticlTracksterLinksSuperclustering{tag}"),
        getattr(process, f"ticlEGammaSuperClusterProducer{tag}"),
        getattr(process, f"ecalDrivenElectronSeeds{tag}"),
        getattr(process, f"electronMergedSeeds{tag}"),
        getattr(process, f"electronCkfTrackCandidates{tag}"),
        getattr(process, f"electronGsfTracks{tag}"),
        getattr(process, f"pfTrack{tag}"),
        getattr(process, f"pfTrackElec{tag}"),
        getattr(process, f"ecalDrivenGsfElectronCoresHGC{tag}"),
        getattr(process, f"ecalDrivenGsfElectronsHGC{tag}"),
        getattr(process, f"photonCoreHGC{tag}"),
        getattr(process, f"photonsHGC{tag}"),
        getattr(process, f"mergedSuperClustersHGC{tag}"),
        getattr(process, f"electronMcSignalValidator{tag}"),
        getattr(process, f"electronMcSignalValidatorPt1000{tag}"),
        getattr(process, f"electronMcFakeValidator{tag}"),
    ))
    process.schedule.associate(getattr(process, f"DNN_task{tag}"))

# removing default egamma path (makes sure we are not accidentally using the default path somewhere)
del process.ticlTracksterLinksSuperclustering
del process.ticlEGammaSuperClusterProducer
del process.ecalDrivenElectronSeeds
del process.electronMergedSeeds
del process.electronCkfTrackCandidates
del process.electronGsfTracks
del process.pfTrack
del process.pfTrackElec
del process.ecalDrivenGsfElectronCoresHGC
del process.photonCoreHGC
del process.photonsHGC
del process.mergedSuperClustersHGC
del process.electronMcSignalValidator

# remove weird unscheduled stuff that runs even when not in a schedule
del process.ecalRecHit
del process.hbhereco
del process.siPixelClustersPreSplitting
del process.siPixelRecHitsPreSplitting




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

process.FEVTDEBUGHLToutput.outputCommands = cms.untracked.vstring(
    "drop *",

    # RECO stuff to keep
    'keep *_genParticles_*_*',
    'keep CaloParticles_mix_*_*',

    "keep *_particleFlowClusterHGCal_*_RECO",
    "keep *_particleFlowSuperClusterHGCal_*_RECO",

)


for wp in WPs:
    tag = "DNNWP" + str(wp).replace(".", "p")
    process.FEVTDEBUGHLToutput.outputCommands.extend([
        f"keep *_ticlTracksterLinksSuperclustering{tag}_*_reRECO",
        f"keep *_ticlEGammaSuperClusterProducer{tag}_*_reRECO",

        # egamma stuff
        f"keep *_ecalDrivenElectronSeeds{tag}_*_reRECO",
        f"keep *_mergedSuperClustersHGC{tag}_*_reRECO",

        f"keep *_electronMergedSeeds{tag}_*_reRECO",
        f"keep *_photonCoreHGC{tag}_*_reRECO",
        f"keep *_photonsHGC{tag}_*_reRECO",
        f"keep *_electronCkfTrackCandidates{tag}_*_reRECO",
        f"keep *_electronGsfTracks{tag}_*_reRECO",
        # "keep *_pfTrack_*_reRECO", # complains about stuff missing
        f"keep *_pfTrackElec{tag}_*_reRECO",

        f"keep *_ecalDrivenGsfElectronCoresHGC{tag}_*_reRECO",
        f"keep *_ecalDrivenGsfElectronsHGC{tag}_*_reRECO",
        # "keep *_cleanedEcalDrivenGsfElectronsHGC_*_reRECO",
        # "keep *_patElectronsHGC_*_reRECO",
    ])