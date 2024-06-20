""" Optuna on validation loss, fixing trainingLossType 
Using NewELectronsD98 v2-48f198a2-fixAngle
Commands : 
python3 -m dnn_training.hyperparams_config.v7 --device cuda:1

"""
import argparse
import optuna
from optuna.trial import Trial

from dnn_training.hyperparameter_scan import *

VERSION = "v7"

parser = argparse.ArgumentParser(
                    prog='superclusteringDNNHyperparameterScan',
                    description='Hyperparameter scan for training of superclustering DNN for HGCAL')

parser.add_argument("--createStudy", help="Should create the study (otherwise just use existing study)", action="store_true")
# parser.add_argument("--trainingLossType", help="Training loss type")
parser.add_argument("-D", "--device", help="Device for pytorch, ex : cpu or cuda:0", default="cuda:2")

if __name__ == "__main__":
    args = parser.parse_args()
else:
    args = argparse.Namespace()

args.input = "/grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/v2-48f198a2-fixAngle/dumper_*.root"
args.output = f'/grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/hyperparams-{VERSION}/'
# args.device = "cuda:2"
args.studyName = f"hp-{VERSION}-cmssw"
args.nEpochs = 400
args.timeout = None
args.cmssw_release_path="/grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_1_X_2024-05-15-1100"
args.cmssw_py_config_path="/grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/NewElectronsD98/rereco_WPscan.py"
args.cmssw_py_config_path_harvesting="/grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/NewElectronsD98/step4_WPscan.py"
args.cmmsw_input_files=[f"/grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/v2-65772-mustache/step3_{i}.root" for i in range(1, 30)]
args.input_DQM_forComparison="/grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/v2-65772-mustache/DQM/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root"
args.cmsswScanWP = True

os.makedirs(args.output, exist_ok=True)

study_kwargs = dict(
        study_name=args.studyName,
        #pruner=optuna.pruners.PatientPruner(optuna.pruners.HyperbandPruner(min_resource=10, max_resource=args.nEpochs), patience=10),
        storage=optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(args.output + f"/journal_optuna_{args.studyName}.log"),)
)
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    if args.createStudy:
        study = optuna.create_study(**study_kwargs)
    else:
        study = optuna.load_study(**study_kwargs)

    if "BATCH_SYSTEM" in os.environ: # disable multithreading on HTCondor (until we can set request_cpus to something intermediate between 1 and 8 on LLR T3)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # study.sampler = optuna.samplers.PartialFixedSampler({"trainingLossType" : args.trainingLossType}, study.sampler)
    study.optimize(functools.partial(objective_reportInference, args=args), gc_after_trial=True)