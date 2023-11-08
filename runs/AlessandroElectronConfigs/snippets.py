# Collection of fixes to issues in reading electron samples
# file not meant to be run, just copy paste relevant fixes into a config file


# Fix for error reading std::vector<l1t::EMTFHit> simEmtfDigis '' HLT
# and l1tPFJets_l1tSCPFL1PuppiCorrectedEmulator__HLT.
for str in process.FEVTDEBUGHLToutput.outputCommands:
    if "simEmtfDigis" in str or "l1tSCPFL1PuppiCorrectedEmulator" in str or "l1tTkMuonsGmt" in str:
        process.FEVTDEBUGHLToutput.outputCommands.remove(str)


# Fix for using FEVTDEBUG and not FEVTDEBUGHLT
# Fix for https://github.com/cms-sw/cmssw/blob/245daa4d7ad8e1412084c2d66381edfd8835d7ad/Configuration/EventContent/python/EventContent_cff.py#L654
# Which only adds this keep line for FEVTDEBUGHLT and not FEVTDEBUG
process.FEVTDEBUGoutput.outputCommands.append("keep recoMuons_muons1stStep_*_*")


# Debug on (needs file compiled with define EDM_ML_DEBUG as well)
process.MessageLogger.cerr.threshold = "DEBUG"
process.MessageLogger.debugModules = ["ticlTrackstersSuperclustering"]



###
from RecoHGCal.TICL.customiseTICLFromReco import *
process = customiseTICLFromReco(process, pileup=True)
process = customiseTICLForDumper(process)

