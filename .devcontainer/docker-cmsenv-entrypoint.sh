#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
#cd CMSSW_13_2_5_patch2
#cmsenv

export TESTVAR=abc
echo TESTABC
exec "$@"
