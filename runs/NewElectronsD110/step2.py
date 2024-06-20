# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 -s DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:@fake2 --conditions auto:phase2_realistic_T33 --datatier GEN-SIM-DIGI-RAW -n -1 --eventcontent FEVTDEBUGHLT --geometry Extended2026D110 --era Phase2C17I13M9 --pileup AVE_200_BX_25ns --pileup_input das:/RelValMinBias_14TeV/CMSSW_14_1_0_pre3-140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/GEN-SIM --filein file:step1.root --fileout file:step2.root --nThreads 8 --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process('HLT',Phase2C17I13M9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D110Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.SimPhase2L1GlobalTriggerEmulator_cff')
process.load('L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.prototypeSeeds')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_Fake2_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('file:step1.root'),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_genParticles_*_*',
        'drop *_genParticlesForJets_*_*',
        'drop *_kt4GenJets_*_*',
        'drop *_kt6GenJets_*_*',
        'drop *_iterativeCone5GenJets_*_*',
        'drop *_ak4GenJets_*_*',
        'drop *_ak7GenJets_*_*',
        'drop *_ak8GenJets_*_*',
        'drop *_ak4GenJetsNoNu_*_*',
        'drop *_ak8GenJetsNoNu_*_*',
        'drop *_genCandidatesForMET_*_*',
        'drop *_genParticlesForMETAllVisible_*_*',
        'drop *_genMetCalo_*_*',
        'drop *_genMetCaloAndNonPrompt_*_*',
        'drop *_genMetTrue_*_*',
        'drop *_genMetIC5GenJs_*_*'
    ),
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
    annotation = cms.untracked.string('step2 nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step2.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.input.nbPileupEvents.averageNumber = cms.double(200.000000)
process.mix.bunchspace = cms.int32(25)
process.mix.minBunch = cms.int32(-3)
process.mix.maxBunch = cms.int32(3)
process.mix.input.fileNames = cms.untracked.vstring(['/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/01b6ca41-1cce-4793-8684-38aa0ba7a4f2.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/05d8f2c1-d565-486b-95e5-b22b33af73a8.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/0ae95870-d715-46ef-aefa-0b7b0af94e40.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/0b7c5daa-43d3-4014-8630-b1a02e681c46.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/0c8fa252-173c-4116-988b-5434ac481979.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/0d77f95d-1114-43fd-913b-a2412165e2b8.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/15c7a0ce-a98d-4913-9dce-f6fe6835cc0c.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/1f2aca34-763c-454c-aefd-6d23f82b1fbf.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/20bfb91d-513f-4040-8919-6536d8a3ecce.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/22a68313-050e-49f8-9e61-ea88d54b719b.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/26fb5b2d-ad0b-42b5-aa75-50a8b038c710.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/29041382-689f-466c-91bd-71be57bda194.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/2a7f7203-69cf-4a49-99b8-bf1283aec431.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/2ce85bac-bae4-49d9-86bd-17441776cd94.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/3d772d0c-6647-476c-af74-2e74fb17493b.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/3f29a508-7dd7-4f7f-9239-30aba8494273.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/42aa8f5d-3010-47da-9639-e0e36fdf0439.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/4422b61f-bf2a-4c83-9778-47ecdb2e5b8a.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/49700831-a610-48fb-960c-3ed34e6038cc.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/4b6cadf3-44f2-405c-a9ef-37481388bd89.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/4d418c98-aea6-440d-9feb-6fd8ce5725a7.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/4ee87b1d-ff2a-4578-85fb-99f3cf03a408.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/5123657b-cb4d-49dc-9a9a-8e134dcceb0b.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/58bd7828-f862-4169-a5aa-770ee0aea563.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/59b08099-6982-4f7f-b73b-0151b9870927.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/5bd0824d-037a-4f42-81ca-af877925a302.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/62197fea-7e21-41c3-92c2-79e35fcb5ded.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/6220d56d-9647-460c-bdcf-5f300393b323.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/6e2d85ac-2dd8-4e26-b0c4-e9343d30b4bc.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/75313a07-9d78-4435-b0a2-0c45f76e5445.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/78452dfa-5512-4108-8f20-ba7a9599d7b9.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/7f1bd20d-bf2e-45db-81df-14cd4b4eab5f.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/85cebfc7-c5d2-496a-b284-9b4b43a5de87.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/86584629-e2b8-4fb9-a0af-5378e0e8413c.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/898a8f2a-9364-41df-8ad1-4e49d94551ee.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/89ad3764-c1c5-4508-8158-b6438bdff67a.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/926df8da-5fc3-4cd6-bca8-74020943dd68.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/92add9c6-d025-414e-b8f1-982acf97b1d6.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/9a974d1a-7c97-4d21-8842-d5b3c8b4e22b.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/9f2ed38f-cba5-4e16-adc5-36df6790a915.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/9f37c2a8-8a75-44b3-a539-86c2bca307a5.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/a46c3e98-4d58-486f-b3f0-38803af95e70.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/a675ad76-25d1-482e-97f0-f97dc4b47e87.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/b0e29c0a-87e9-4545-a77b-526f67718430.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/b2e19e49-e532-4fd0-8003-03fd99fb4b70.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/c191e607-4c19-4982-817a-07ec0540ffe3.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/c2019a51-3c3e-4d17-90ed-e61921f43e81.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/c9b6bdd8-5fa9-42b9-b967-2b775913bb89.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/d666322c-c86d-43d5-a704-fee30a9f51fb.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/e006ba35-cab5-416c-9979-d566dcf865f9.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/e1374671-1446-4f17-ba9c-9a4551fa71a9.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/e56f447f-fb71-402b-881f-7491dffce152.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/eeb1ca7b-d9ea-45f5-8b82-0c9a8187780b.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/f505362d-dd9b-42c0-8f75-a4638b4305e1.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/f917ff01-fbb3-4fc5-9cb5-9b36dae8d420.root', '/store/relval/CMSSW_14_1_0_pre3/RelValMinBias_14TeV/GEN-SIM/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2590000/fba40e06-5edc-48a2-81d6-90a7454bfb42.root'])
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.Phase2L1GTProducer = cms.Path(process.l1tGTProducerSequence)
process.Phase2L1GTAlgoBlockProducer = cms.Path(process.l1tGTAlgoBlockProducerSequence)
process.pDoubleEGEle37_24 = cms.Path(process.DoubleEGEle3724)
process.pDoubleIsoTkPho22_12 = cms.Path(process.DoubleIsoTkPho2212)
process.pDoublePuppiJet112_112 = cms.Path(process.DoublePuppiJet112112)
process.pDoublePuppiTau52_52 = cms.Path(process.DoublePuppiTau5252)
process.pDoubleTkEle25_12 = cms.Path(process.DoubleTkEle2512)
process.pDoubleTkMuon15_7 = cms.Path(process.DoubleTkMuon157)
process.pIsoTkEleEGEle22_12 = cms.Path(process.IsoTkEleEGEle2212)
process.pPuppiHT400 = cms.Path(process.PuppiHT400)
process.pPuppiHT450 = cms.Path(process.PuppiHT450)
process.pPuppiMET200 = cms.Path(process.PuppiMET200)
process.pQuadJet70_55_40_40 = cms.Path(process.QuadJet70554040)
process.pSingleEGEle51 = cms.Path(process.SingleEGEle51)
process.pSingleIsoTkEle28 = cms.Path(process.SingleIsoTkEle28)
process.pSingleIsoTkPho36 = cms.Path(process.SingleIsoTkPho36)
process.pSinglePuppiJet230 = cms.Path(process.SinglePuppiJet230)
process.pSingleTkEle36 = cms.Path(process.SingleTkEle36)
process.pSingleTkMuon22 = cms.Path(process.SingleTkMuon22)
process.pTripleTkMuon5_3_3 = cms.Path(process.TripleTkMuon533)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
# process.schedule imported from cff in HLTrigger.Configuration
process.schedule.insert(0, process.digitisation_step)
process.schedule.insert(1, process.L1TrackTrigger_step)
process.schedule.insert(2, process.L1simulation_step)
process.schedule.insert(3, process.Phase2L1GTProducer)
process.schedule.insert(4, process.Phase2L1GTAlgoBlockProducer)
process.schedule.insert(5, process.pDoubleEGEle37_24)
process.schedule.insert(6, process.pDoubleIsoTkPho22_12)
process.schedule.insert(7, process.pDoublePuppiJet112_112)
process.schedule.insert(8, process.pDoublePuppiTau52_52)
process.schedule.insert(9, process.pDoubleTkEle25_12)
process.schedule.insert(10, process.pDoubleTkMuon15_7)
process.schedule.insert(11, process.pIsoTkEleEGEle22_12)
process.schedule.insert(12, process.pPuppiHT400)
process.schedule.insert(13, process.pPuppiHT450)
process.schedule.insert(14, process.pPuppiMET200)
process.schedule.insert(15, process.pQuadJet70_55_40_40)
process.schedule.insert(16, process.pSingleEGEle51)
process.schedule.insert(17, process.pSingleIsoTkEle28)
process.schedule.insert(18, process.pSingleIsoTkPho36)
process.schedule.insert(19, process.pSinglePuppiJet230)
process.schedule.insert(20, process.pSingleTkEle36)
process.schedule.insert(21, process.pSingleTkMuon22)
process.schedule.insert(22, process.pTripleTkMuon5_3_3)
process.schedule.insert(23, process.digi2raw_step)
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads = 8
process.options.numberOfStreams = 0

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# End of customisation functions


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
