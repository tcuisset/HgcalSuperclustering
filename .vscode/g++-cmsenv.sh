#!/bin/bash
cd /workspaces/cmssw/CMSSW_13_2_5_patch2
cmsenv
cd -
g++ "$@"