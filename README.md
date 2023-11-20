# Electron superclustering studies in HGCAL
Based off Alessadnro Tarabini's work

## Setup
You should run `cmsrel CMSSW_13_2_5_patch2` in this directory (it is in gitignore of this repository).
This makes nested git repositories (not submodules), they are completely independent. 
Then checkout branch dnn-supercls from https://github.com/tcuisset/cmssw inside the cmssw repository, and build

~~~bash
cmsrel CMSSW_13_2_5_patch2
cd CMSSW_13_2_5_patch2/src
cmsenv
# replace tcuisset with the github username from which you want to pull the branch
git-cms-checkout-topic tcuisset:dnn-supercls
scram b -j10
~~~

For the DNN see in `trained-dnn/README`

Top-level directories named `scratch` and `data*` are ignored by git