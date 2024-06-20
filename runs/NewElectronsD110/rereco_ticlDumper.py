# ReRECO egamma chain + electron validation from step3 + ticlDumper
# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 -s RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM --conditions auto:phase2_realistic_T25 --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO -n 10 --eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM --geometry Extended2026D98 --era Phase2C17I13M9 --procModifiers ticl_v5 --filein file:step2.root --fileout file:step3.root --nThreads 10
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

process = cms.Process('reRECO',Phase2C17I13M9,ticl_v5,ticl_v5_superclustering_dnn)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D110Reco_cff')
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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '')

# Path and EndPath definitions

############# DNN
process.superclustering_step = cms.Path(process.ticlTracksterLinksSuperclusteringDNN + process.ticlEGammaSuperClusterProducer)
process.ticlEGammaSuperClusterProducer.ticlSuperClusters=cms.InputTag("ticlTracksterLinksSuperclusteringDNN")
# trackerDrivenElectronSeeds is needed to make ElectronSeed with detector information (transient)
process.egamma_step = cms.Path(process.ecalDrivenElectronSeeds + process.trackerDrivenElectronSeeds + process.electronMergedSeeds + process.electronCkfTrackCandidates + process.electronGsfTracks + process.pfTrack + process.pfTrackElec + process.ecalDrivenGsfElectronCoresHGC + process.ecalDrivenGsfElectronsHGC + process.photonCoreHGC + process.photonsHGC)

# process.reconstruction_step = cms.Path(process.reconstruction
# process.recosim_step = cms.Path(process.recosim)

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
# del process.electronMcSignalValidatorPt1000 # we need gedGsfElectrons rereco for this (could be done)
process.electronValidationSequence_step = cms.Path(process.mergedSuperClustersHGC + process.electronValidationSequence)

del process.ecalDetailedTimeRecHit
# We need to rereco pfRecHits as we need CaloCell (transient) for pfTrackElec
process.pfRecHit_step = cms.Path(process.particleFlowRecHitHGC + process.particleFlowRecHitECAL +process.particleFlowRecHitPS+process.particleFlowClusterECALUncorrected+ process.particleFlowClusterPS +process.ecalBarrelClusterFastTimer+ process.particleFlowTimeAssignerECAL + process.particleFlowClusterECAL)


# TICL dumper + superclsSampleDumper
process.TFileService = cms.Service("TFileService",
                                    fileName=cms.string("dumper.root")
                                    )
process.load("RecoHGCal.TICL.ticlDumper_cff")
# remove PU associator as that takes forever and we don't use it
del process.tracksterSimTracksterAssociationLinkingPU
process.ticlDumper.associators.remove(next(x for x in process.ticlDumper.associators if x.suffix == cms.string("PU")))
process.ticlDumper_step = cms.EndPath(process.ticlDumper)

# we need supercls associator as well
# so run it or disable it
process.ticlDumper.associators = cms.VPSet(
        cms.PSet(
            branchName=cms.string("tsCLUE3D"),
            suffix=cms.string("SC"),
            associatorInputTag=cms.InputTag("tracksterSimTracksterAssociationPRbyCLUE3D"),
            tracksterCollection=cms.InputTag("ticlTrackstersCLUE3DHigh"),
            simTracksterCollection=cms.InputTag("ticlSimTracksters")
        ),
        cms.PSet(
            branchName=cms.string("tsCLUE3D"),
            suffix=cms.string("CP"),
            associatorInputTag=cms.InputTag("tracksterSimTracksterAssociationLinkingbyCLUE3D"),
            tracksterCollection=cms.InputTag("ticlTrackstersCLUE3DHigh"),
            simTracksterCollection=cms.InputTag("ticlSimTracksters", "fromCPs")
        ))
# process.supercls_associator = cms.Task(process.tracksterSimTracksterAssociationLinkingSuperclustering, process.tracksterSimTracksterAssociationPRSuperclustering)
# process.superclustering_step.associate(process.supercls_associator)

# Schedule definition
process.schedule = cms.Schedule(process.pfRecHit_step,
                                process.reconstruction_trackingOnly_step, 
                                process.superclustering_step,
                                # process.egamma_step,
                                # process.electronValidationSequence_step,
                                # process.FEVTDEBUGHLToutput_step,
                                # process.DQMoutput_step,
                                process.ticlDumper_step)
#process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.recosim_step,process.Flag_HBHENoiseFilter,process.Flag_HBHENoiseIsoFilter,process.Flag_CSCTightHaloFilter,process.Flag_CSCTightHaloTrkMuUnvetoFilter,process.Flag_CSCTightHalo2015Filter,process.Flag_globalTightHalo2016Filter,process.Flag_globalSuperTightHalo2016Filter,process.Flag_HcalStripHaloFilter,process.Flag_hcalLaserEventFilter,process.Flag_EcalDeadCellTriggerPrimitiveFilter,process.Flag_EcalDeadCellBoundaryEnergyFilter,process.Flag_ecalBadCalibFilter,process.Flag_goodVertices,process.Flag_eeBadScFilter,process.Flag_ecalLaserCorrFilter,process.Flag_trkPOGFilters,process.Flag_chargedHadronTrackResolutionFilter,process.Flag_muonBadTrackFilter,process.Flag_BadChargedCandidateFilter,process.Flag_BadPFMuonFilter,process.Flag_BadPFMuonDzFilter,process.Flag_hfNoisyHitsFilter,process.Flag_BadChargedCandidateSummer16Filter,process.Flag_BadPFMuonSummer16Filter,process.Flag_trkPOG_manystripclus53X,process.Flag_trkPOG_toomanystripclus53X,process.Flag_trkPOG_logErrorTooManyClusters,process.Flag_METFilters,process.prevalidation_step,process.prevalidation_step1,process.prevalidation_step2,process.prevalidation_step3,process.prevalidation_step4,process.prevalidation_step5,process.prevalidation_step6,process.prevalidation_step7,process.prevalidation_step8,process.prevalidation_step9,process.validation_step,process.validation_step1,process.validation_step2,process.validation_step3,process.validation_step4,process.validation_step5,process.validation_step6,process.validation_step7,process.validation_step8,process.validation_step9,process.validation_step10,process.validation_step11,process.validation_step12,process.validation_step13,process.validation_step14,process.dqmoffline_step,process.dqmoffline_1_step,process.dqmoffline_2_step,process.dqmoffline_3_step,process.dqmoffline_4_step,process.dqmoffline_5_step,process.dqmoffline_6_step,process.dqmoffline_7_step,process.dqmoffline_8_step,process.dqmoffline_9_step,process.dqmoffline_10_step,process.dqmofflineOnPAT_step,process.dqmofflineOnPAT_1_step,process.FEVTDEBUGHLToutput_step,process.MINIAODSIMoutput_step,process.DQMoutput_step)

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


    #### reRECO stuff
    # superclustering stuff
    "keep *_ticlTracksterLinksSuperclustering_*_reRECO",
    "keep *_ticlEGammaSuperClusterProducer_*_reRECO",

    # egamma stuff
    "keep *_ecalDrivenElectronSeeds_*_reRECO",
    "keep *_mergedSuperClustersHGC_*_reRECO",

    "keep *_electronMergedSeeds_*_reRECO",
    "keep *_photonCoreHGC_*_reRECO",
    "keep *_photonsHGC_*_reRECO",
    "keep *_electronCkfTrackCandidates_*_reRECO",
    "keep *_electronGsfTracks_*_reRECO",
    # "keep *_pfTrack_*_reRECO", # complains about stuff missing
    "keep *_pfTrackElec_*_reRECO",

    "keep *_ecalDrivenGsfElectronCoresHGC_*_reRECO",
    "keep *_ecalDrivenGsfElectronsHGC_*_reRECO",
    # "keep *_cleanedEcalDrivenGsfElectronsHGC_*_reRECO",
    # "keep *_patElectronsHGC_*_reRECO",
)
