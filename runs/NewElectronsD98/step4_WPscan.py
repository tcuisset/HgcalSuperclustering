# HARVESTING for rereco_WPscan.py (does DQM harvesting for all the working points)
# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step4 -s HARVESTING:@phase2Validation+@phase2 --conditions auto:phase2_realistic_T25 --mc --geometry Extended2026D98 --scenario pp --filetype DQM --era Phase2C17I13M9 --procModifiers ticl_v5 -n -1 --filein file:step3_inDQM.root --fileout file:step4.root --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

process = cms.Process('HARVESTING',Phase2C17I13M9,ticl_v5)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D98Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring('file:step3_inDQM.root')
)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring('ProductNotFound'),
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
    annotation = cms.untracked.string('step4 nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T25', '')

# Path and EndPath definitions
process.alcaHarvesting = cms.Path()
process.dqmHarvesting = cms.Path(process.DQMOffline_SecondStep+process.DQMOffline_Certification)
process.dqmHarvestingExpress = cms.Path(process.DQMOffline_SecondStep_Express)
process.dqmHarvestingExtraHLT = cms.Path(process.DQMOffline_SecondStep_ExtraHLT+process.DQMOffline_Certification)
process.dqmHarvestingFakeHLT = cms.Path(process.DQMOffline_SecondStep_FakeHLT+process.DQMOffline_Certification)
process.dqmHarvestingPOGMC = cms.Path(process.DQMOffline_SecondStep_PrePOGMC)
process.genHarvesting = cms.Path(process.postValidation_gen)
process.validationHarvestingFS = cms.Path(process.recoMuonPostProcessors+process.postValidationTracking+process.MuIsoValPostProcessor+process.calotowersPostProcessor+process.hcalSimHitsPostProcessor+process.hcaldigisPostProcessor+process.hcalrechitsPostProcessor+process.electronPostValidationSequence+process.photonPostProcessor+process.pfJetClient+process.pfMETClient+process.pfJetResClient+process.pfElectronClient+process.rpcRecHitPostValidation_step+process.makeBetterPlots+process.bTagCollectorSequenceMCbcl+process.METPostProcessor+process.L1GenPostProcessor+process.bdHadronTrackPostProcessor+process.MuonCSCDigisPostProcessors+process.MuonGEMHitsPostProcessors+process.MuonGEMDigisPostProcessors+process.MuonGEMRecHitsPostProcessors+process.hgcalPostProcessor+process.trackerphase2ValidationHarvesting+process.postValidation_gen)
process.validationHarvestingHI = cms.Path(process.postValidationHI)
process.validationHarvestingMiniAOD = cms.Path(process.JetPostProcessor+process.METPostProcessorHarvesting+process.bTagMiniValidationHarvesting+process.postValidationMiniAOD)
process.validationHarvestingNoHLT = cms.Path(process.postValidation+process.postValidation_gen)
process.validationHarvestingPhase2 = cms.Path(process.hltpostvalidation)
process.validationpreprodHarvesting = cms.Path(process.postValidation_preprod+process.hltpostvalidation_preprod+process.postValidation_gen)
process.validationpreprodHarvestingNoHLT = cms.Path(process.postValidation_preprod+process.postValidation_gen)
process.validationprodHarvesting = cms.Path(process.hltpostvalidation_prod+process.postValidation_gen)
process.postValidation_common_step = cms.Path(process.postValidation_common)
process.postValidationTracking_step = cms.Path(process.postValidationTracking)
process.postValidation_muons_step = cms.Path(process.postValidation_muons)
process.postValidation_JetMET_step = cms.Path(process.postValidation_JetMET)
process.electronPostValidationSequence_step = cms.Path(process.electronPostValidationSequence)
process.photonPostProcessor_step = cms.Path(process.photonPostProcessor)
process.bTagCollectorSequenceMCbcl_step = cms.Path(process.bTagCollectorSequenceMCbcl)
process.runTauEff_step = cms.Path(process.runTauEff)
process.postValidation_HCAL_step = cms.Path(process.postValidation_HCAL)
process.hgcalValidatorPostProcessor_step = cms.Path(process.hgcalValidatorPostProcessor)
process.mtdValidationPostProcessor_step = cms.Path(process.mtdValidationPostProcessor)
process.trackerphase2ValidationHarvesting_step = cms.Path(process.trackerphase2ValidationHarvesting)
process.DQMHarvestBeam_step = cms.Path(process.DQMHarvestBeam)
process.DQMHarvestTracking_step = cms.Path(process.DQMHarvestTracking)
process.DQMHarvestOuterTracker_step = cms.Path(process.DQMHarvestOuterTracker)
process.DQMHarvestTrackerPhase2_step = cms.Path(process.DQMHarvestTrackerPhase2)
process.DQMHarvestMuon_step = cms.Path(process.DQMHarvestMuon)
process.DQMCertMuon_step = cms.Path(process.DQMCertMuon)
process.DQMHarvestHcal_step = cms.Path(process.DQMHarvestHcal)
process.DQMHarvestHcal2_step = cms.Path(process.DQMHarvestHcal2)
process.DQMHarvestEGamma_step = cms.Path(process.DQMHarvestEGamma)
process.DQMCertEGamma_step = cms.Path(process.DQMCertEGamma)
process.DQMHarvestL1TPhase2_step = cms.Path(process.DQMHarvestL1TPhase2)
process.HLTMonitoringClient_step = cms.Path(process.HLTMonitoringClient)
process.dqmsave_step = cms.Path(process.DQMSaver)


# "fix" crashes
del process.photonPostprocessing
del process.pfPhotonPostprocessing

# Schedule definition
process.schedule = cms.Schedule(process.postValidation_common_step,
    #process.postValidationTracking_step,process.postValidation_muons_step,process.postValidation_JetMET_step,
    #process.electronPostValidationSequence_step,
    process.photonPostProcessor_step,
    #process.bTagCollectorSequenceMCbcl_step,process.runTauEff_step,process.postValidation_HCAL_step,
    process.hgcalValidatorPostProcessor_step,process.mtdValidationPostProcessor_step,
    #process.trackerphase2ValidationHarvesting_step,process.validationHarvesting,process.DQMHarvestBeam_step,process.DQMHarvestTracking_step,process.DQMHarvestOuterTracker_step,process.DQMHarvestTrackerPhase2_step,process.DQMHarvestMuon_step,process.DQMCertMuon_step,process.DQMHarvestHcal_step,process.DQMHarvestHcal2_step,
    process.DQMHarvestEGamma_step,process.DQMCertEGamma_step,
    #process.DQMHarvestL1TPhase2_step,process.HLTMonitoringClient_step,
    process.dqmsave_step)

import numpy as np
WPs = list(np.linspace(0., 0.1, 20, endpoint=False)) + list(np.linspace(0.1, 0.9, 25, endpoint=False)) + list(np.linspace(0.9, 1., 20))
for wp in WPs:
    tag = "DNNWP" + str(wp).replace(".", "p")
    setattr(process, f"electronMcSignalPostValidator{tag}", process.electronMcSignalPostValidator.clone(InputFolderName=cms.string(f"EgammaV{tag}/ElectronMcFakeValidator"), OutputFolderName=cms.string(f"EgammaV{tag}/ElectronMcFakeValidator")))
    process.electronPostValidationSequence_step.insert(-1, getattr(process, f"electronMcSignalPostValidator{tag}"))
del process.electronMcSignalPostValidator
del process.electronMcFakePostValidator
del process.electronMcSignalPostValidatorPt1000

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)



# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
