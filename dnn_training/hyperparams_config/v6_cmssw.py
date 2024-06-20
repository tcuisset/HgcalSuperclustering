""" Optuna on validation loss, fixing trainingLossType 
Using NewELectronsD98 v2-65772-mustache (NB : this has bugged angle)
Commands : 
python3 -m dnn_training.hyperparams_config.v6_cmssw --device cuda:1

"""
import argparse
import optuna
from optuna.trial import Trial

from dnn_training.hyperparameter_scan import *

VERSION = "v6"

parser = argparse.ArgumentParser(
                    prog='superclusteringDNNHyperparameterScan',
                    description='Hyperparameter scan for training of superclustering DNN for HGCAL')

parser.add_argument("--createStudy", help="Should create the study (otherwise just use existing study)", action="store_true")
# parser.add_argument("--trainingLossType", help="Training loss type")
parser.add_argument("-D", "--device", help="Device for pytorch, ex : cpu or cuda:0", default="cuda:2")

args = parser.parse_args()

args.input = "/grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/v2-65772-mustache/dumper_*.root"
args.output = f'/grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/hyperparams-{VERSION}/cmssw'
# args.device = "cuda:2"
args.studyName = f"hp-{VERSION}-cmssw"
args.nEpochs = 400
args.timeout = None
args.cmssw_release_path="/grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_1_X_2024-04-29-1100"
args.cmssw_py_config_path="/grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/NewElectronsD98/rereco.py"
args.cmssw_py_config_path_harvesting="/grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/NewElectronsD98/step4.py"
args.cmmsw_input_files=[f"/grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/v2-65772-mustache/step3_{i}.root" for i in range(1, 20)]
args.input_DQM_forComparison="/grid_mnt/data_cms_upgrade/cuisset/supercls/NewElectronsD98/v2-65772-mustache/DQM/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root"


os.makedirs(args.output, exist_ok=True)

study_kwargs = dict(
        study_name=args.studyName,
        #pruner=optuna.pruners.PatientPruner(optuna.pruners.HyperbandPruner(min_resource=10, max_resource=args.nEpochs), patience=10),
        storage=optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(args.output + f"/journal_optuna_{args.studyName}.log"),),
        
)
if args.createStudy:
    study = optuna.create_study(**study_kwargs, direction="maximize")
    study.enqueue_trial({"batchSize" : 10000, "dropout" : 0.1, "lr" : 0.0001, "trainingLossType" : "binary", "weightSamples" : False})
    study.enqueue_trial({"batchSize" : 1000, "dropout" : 0.1, "lr" : 0.002, "trainingLossType" : "binary", "weightSamples" : True})
    study.enqueue_trial({"batchSize" : 10000, "dropout" : 0.05, "lr" : 0.001, "trainingLossType" : "continuousAssociationScore", "weightSamples" : False})
    study.enqueue_trial({"batchSize" : 5000, "dropout" : 0.15, "lr" : 0.005, "trainingLossType" : "continuousAssociationScore", "weightSamples" : True})
else:
    study = optuna.load_study(**study_kwargs)

if "BATCH_SYSTEM" in os.environ: # disable multithreading on HTCondor (until we can set request_cpus to something intermediate between 1 and 8 on LLR T3)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

#study.sampler = optuna.samplers.PartialFixedSampler({"trainingLossType" : args.trainingLossType}, study.sampler)
#  trainer_kwargs=dict(trialReportMode="Loss_val")),
study.optimize(functools.partial(objective_reportInference, args=args), gc_after_trial=True)