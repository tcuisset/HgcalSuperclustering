import functools
import os
import optuna
from optuna.trial import Trial
from pathlib import Path

from dnn_training.trainer import *
from dnn_training.cmsswInference import runInferenceCMSSW, DQMReader, cmssw_inference_WPs

# device = "cuda:1"
# class args:
#     output = "/grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/hyperparams-v1/tb"
#     input = r"/grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v15-sampleDump/superclsDumper_\*.root"
#     nEpochs = 10

def objective_reportValidation(trial:Trial, args, trainer_kwargs=dict(trialReportMode="SigmaOverMu-BestWP_sumEnergyBins")):
    """ Objective for hyperparameter training, reporting per validation"""
    print(f"Starting trial nb {trial.number}")
    model = makeModelOptuna(trial)
    loss = makeLossOptuna(trial)
    trainer = Trainer(model.to(args.device), loss, trial=trial, device=args.device, log_output=args.output, **trainer_kwargs)
    #print("Loading dataset...")
    datasets = makeDatasetsTrainVal_fromCache(args.input, device_valDataset=args.device, trainingLossType=trial.params["trainingLossType"])
    #print("Dataset loaded into RAM")
    return trainer.full_train(datasets["trainDataset"], datasets["valDataset"], nepochs=args.nEpochs)

def objective_reportInference(trial:Trial, args, metric_name="Loss_val"):
    """ Objective for hyperparameter training, reporting once at end the score of CMSSW inference with the model """
    print(f"Starting trial nb {trial.number}")
    model = makeModelOptuna(trial)
    loss = makeLossOptuna(trial)
    trainer = Trainer(model.to(args.device), loss, trial=trial, device=args.device, log_output=args.output, trialReportMode=None)
    #print("Loading dataset...")
    datasets = makeDatasetsTrainVal_fromCache(args.input, device_valDataset=args.device, trainingLossType=trial.params["trainingLossType"])
    #print("Dataset loaded into RAM")
    trainer.full_train(datasets["trainDataset"], datasets["valDataset"], nepochs=args.nEpochs)

    inference_dir = Path(trainer.log_output) / "inference"
    inference_dir.mkdir()
    possible_epochs = set(epoch for epoch, format_dict in trainer.saved_models_paths.items() if "onnx" in format_dict).intersection(trainer.all_metrics["best_wp"].keys())
    print(trainer.all_metrics)
    try:
        bestEpoch = trainer.getBestEpochForMetric(metric_name, possible_epochs)
    except ValueError:
        raise ValueError("Not epoch was found having DNN working point computed & model saved, probably because there were not enough epochs.\n"
            f'Epochs with model : {[epoch for epoch, format_dict in trainer.saved_models_paths.items() if "onnx" in format_dict]}\n'
            f'Epochs with WP : {trainer.all_metrics["best_wp"].keys()}\n'
            f'Epochs with metric : {trainer.all_metrics[metric_name].keys()}' )
    best_wp = trainer.all_metrics["best_wp"][bestEpoch]
    trial.set_user_attr(f"bestEpoch_{metric_name}", bestEpoch)
    trial.set_user_attr(f"bestWP_forBest_{metric_name}", best_wp)
    model_path = str(trainer.saved_models_paths[bestEpoch]["onnx"])
    trial.set_user_attr("bestModel_path", model_path)
    print(args)
    del trainer
    del datasets
    del model
    del loss
    import torch
    torch.cuda.empty_cache()
    if args.cmsswScanWP:
        runInferenceCMSSW(
            cmssw_release_path=args.cmssw_release_path,
            cmssw_py_config_path=args.cmssw_py_config_path,
            cmssw_py_config_path_harvesting=args.cmssw_py_config_path_harvesting,
            output_path=inference_dir,
            model_path=model_path,
            #dnn_wp=best_wp,
            dnn_wp=None, # scanning for WP
            input_files=args.cmmsw_input_files,
            input_DQM_forComparison=args.input_DQM_forComparison
        )
    else:
        runInferenceCMSSW(
            cmssw_release_path=args.cmssw_release_path,
            cmssw_py_config_path=args.cmssw_py_config_path,
            cmssw_py_config_path_harvesting=args.cmssw_py_config_path_harvesting,
            output_path=inference_dir,
            model_path=model_path,
            dnn_wp=best_wp,
            input_files=args.cmmsw_input_files,
            input_DQM_forComparison=args.input_DQM_forComparison
        )

    dqm = DQMReader(inference_dir / "DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root")
    if args.cmsswScanWP:
        trial.set_user_attr(f"ptEffs_WPscan", {wp : dqm.egammaVForWP(wp).ptEff.values().tolist() for wp in cmssw_inference_WPs})
        inclusiveEffs_WPscan = {wp : dqm.egammaVForWP(wp).efficiency() for wp in cmssw_inference_WPs}
        trial.set_user_attr(f"inclusiveEffs_WPscan", inclusiveEffs_WPscan)

        bestWP_inclEffs = max(inclusiveEffs_WPscan, key=inclusiveEffs_WPscan.get)
        inclusiveEffForBestWP = inclusiveEffs_WPscan[bestWP_inclEffs]
        trial.set_user_attr("bestWP_inclEffs", bestWP_inclEffs)
        trial.set_user_attr("inclusiveEffForBestWP", inclusiveEffForBestWP)
        return inclusiveEffForBestWP
    else:
        trial.set_user_attr(f"ptEff", dqm.egammaV.ptEff.values())
        return dqm.efficiency()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='superclusteringDNNHyperparameterScan',
                    description='Hyperparameter scan for training of superclustering DNN for HGCAL')
    parser.add_argument("-i", "--input", help="Path to folder or file holding the superclustering sample dumper output. Passed to uproot.concatenate", required=True)
    parser.add_argument("-o", "--output", help="Directory to put output (trained network, tensorboard logs, etc)")
    parser.add_argument("-D", "--device", help="Device for pytorch, ex : cpu or cuda:0", default="cpu")
    parser.add_argument("-e", "--nEpochs", help="Number of epochs to train for", default=500, type=int)
    parser.add_argument("-n", "--studyName", help="Name of the study")
    parser.add_argument("-t", "--timeout", help="Stop starting new trials after timeout seconds", default=None)
    parser.add_argument("--createStudy", help="Should create the study (otherwise just use existing study)", action="store_true")

    # CMSSW inference
    parser.add_argument("--cmssw", help="CMSSW release path", dest="cmssw_release_path", required=False)
    parser.add_argument("--cmsswConfig", help="Step3 config file for inference", dest="cmssw_py_config_path", required=False)
    parser.add_argument("--cmsswConfigHarvesting", help="Step4 config file for harvesting inference results", dest="cmssw_py_config_path_harvesting", required=False)
    parser.add_argument("--cmsswInputFiles", help="Input files for CMSSW inference", dest="cmmsw_input_files", required=False, nargs="+")
    parser.add_argument("--DQMForComparison", help="DQM file for comparison", dest="input_DQM_forComparison", required=False)
    parser.add_argument("--cmsswScanWP", help="Config in cmsswConfig scans WPs, so don't set a WP", action="store_true", default=False)
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    study_kwargs = dict(
            study_name=args.studyName,
            pruner=optuna.pruners.PatientPruner(optuna.pruners.HyperbandPruner(min_resource=10, max_resource=args.nEpochs), patience=10),
            storage=optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(args.output + "/journal_optuna.log"),)
    )
    if args.createStudy:
        study = optuna.create_study(**study_kwargs)
    else:
        study = optuna.load_study(**study_kwargs)
    
    if "BATCH_SYSTEM" in os.environ: # disable multithreading on HTCondor (until we can set request_cpus to something intermediate between 1 and 8 on LLR T3)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    study.optimize(functools.partial(objective_reportValidation, args=args), timeout=args.timeout)