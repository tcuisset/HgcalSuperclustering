#!/bin/bash
cd /workspaces/repo/CMSSW_14_0_0_pre1
cmsset
cmsenv
cd -
g++ "$@"