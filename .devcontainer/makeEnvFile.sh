#!/bin/bash

docker build --tag 'makeenvfile' .

# --mount type=bind,source=/home/cuisset/hgcal/supercls/cmssw,target=/workspaces/cmssw --mount source=/cvmfs,target=/cvmfs,type=bind,bind-propagation=shared

RUNTIME_RES=$(docker run --rm --mount type=bind,source=/home/cuisset/hgcal/supercls/cmssw,target=/workspaces/cmssw --mount source=/cvmfs,target=/cvmfs,type=bind,bind-propagation=shared \
 -ia stdout makeenvfile /usr/bin/bash << 'EOF'
cd /workspaces/cmssw/CMSSW_13_2_5_patch2
source /cvmfs/cms.cern.ch/cmsset_default.sh
scramv1 runtime -sh
EOF
)

echo "$RUNTIME_RES" | sed -r 's/export ([a-zA-Z_$0-9]+)="([^"]+)";{0,1}/\1=\2/' - >cmsenv.env

#docker exec makeenvfile "cd /workspaces/cmssw/CMSSW_13_2_5_patch2;source /cvmfs/cms.cern.ch/cmsset_default.sh; scramv1 runtime -sh; "
