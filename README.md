# Electron superclustering studies in HGCAL
Based off Alessandro Tarabini's work

There is a conda environment : supercls-analyzer-env.yml 

## Organization
 - analyzer : python code for making performance plots
 - cmssw_jobs : archive of all condor jobs submitted (submit files + logs)
 - dnn_training : retraining of the DNN (fairly self-contained)
 - lawJobs : ignore
 - runs : store of all CMSSW python config files used

## Analyzer
### Organization
 - dumperReader : code to read TICLDumper output and study tracksters, superclusters and sim associations
 - energy_resolution : making energy resolution plots
 - computations & driver : make dataframes at scale and in parallel for resolution plots with lots of statistics. 
 - egamma : looking at GSFElectrons, mostly from CMSSW FEVTDEBUG files
 - 
 