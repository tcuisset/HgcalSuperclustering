#!/bin/bash
cd /workspaces/repo/base_cmssw/CMSSW_14_1_DBG_X_2024-04-18-2300
cmsenv
cd - &>/dev/null
exec gdb "$@"