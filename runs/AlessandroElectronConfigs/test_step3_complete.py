import sys
#sys.path.append("../AlessandroElectronSamples")
import FWCore.ParameterSet.Config as cms

#sys.path.append("..")
from AlessandroElectronSamples.step3_complete import process

process.source.fileNames = cms.untracked.vstring('file:../../data_alessandro/electrons_pre4_PU/step2_1.root')
process.maxEvents.input = cms.untracked.int32(1)
