import functools
import os
import optuna
from optuna.trial import Trial

from dnn_training.trainer import *

# device = "cuda:1"
# class args:
#     output = "/grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/hyperparams-v1/tb"
#     input = r"/grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v15-sampleDump/superclsDumper_\*.root"
#     nEpochs = 10

def objective(trial:Trial, args):
    print(f"Starting trial nb {trial.number}")
    model = makeModelOptuna(trial)
    loss = makeLossOptuna(trial)
    trainer = Trainer(model.to(args.device), loss, trial=trial, device=args.device, log_output=args.output)
    #print("Loading dataset...")
    datasets = makeDatasetsTrainVal_fromCache(args.input, device_valDataset=args.device, trainingLossType=trial.params["trainingLossType"])
    #print("Dataset loaded into RAM")
    return trainer.full_train(datasets["trainDataset"], datasets["valDataset"], nepochs=args.nEpochs)

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

    study.optimize(functools.partial(objective, args=args), timeout=args.timeout)