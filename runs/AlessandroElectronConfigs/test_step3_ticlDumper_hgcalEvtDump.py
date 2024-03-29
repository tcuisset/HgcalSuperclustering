import sys
import os
#sys.path.append("../AlessandroElectronSamples")
import FWCore.ParameterSet.Config as cms

# python import hack to allow importing from local directory
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from step3_ticlDumper_hgcalEvtDump import process

process.source.fileNames = cms.untracked.vstring('file:/data_cms_upgrade/cuisset/supercls/alessandro_electrons/input-oppositeSign/step2_201.root')
process.maxEvents.input = cms.untracked.int32(1)
