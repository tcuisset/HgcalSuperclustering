import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.trial import Trial, FixedTrial
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import collections
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6, 5)
import matplotlib.pyplot as plt
import copy

from dnn_training.dataset import *

# case candidateTracksterBestAssociation_simTsIdx != seedTracksterBestAssociation_simTsIdx
# -> avoid at maximum

# case candidateTracksterBestAssociation_simTsIdx == seedTracksterBestAssociation_simTsIdx

# seedTracksterBestAssociationScore good && candidateTracksterAssociationWithSeed_score good -> to target
# seedTracksterBestAssociationScore good && candidateTracksterAssociationWithSeed_score bad -> to avoid
# seedTracksterBestAssociationScore bad : what to do ?

featureNames = ['DeltaEtaBaryc', 'DeltaPhiBaryc', 'multi_en', 'multi_eta', 'multi_pt', 'seedEta','seedPhi','seedEn', 'seedPt', 'theta', 'theta_xz_seedFrame', 'theta_yz_seedFrame', 'theta_xy_cmsFrame', 'theta_yz_cmsFrame', 'theta_xz_cmsFrame', 'explVar', 'explVarRatio']
featureCount = 17

class FeatureScaler(nn.Module):
    """ Scales the input features to [0, 1]. Values are taken from Alessandro's notebook. They are not trainable parameters. """
    def __init__(self) -> None:
        super().__init__()
        # Alessandro v2
        self.register_buffer("scaler_scale", torch.tensor([5.009924761018587,
            1.0018188597242355,
            0.010292174692060528,
            0.16288265085698453,
            0.04559109321673852,
            0.1682473240927026,
            0.15967887456557378,
            0.0015586544595969428,
            0.010722259319073196,
            0.6364301120505811,
            0.6360764093711175,
            0.6353961809445742,
            0.31929392780031585,
            0.6707174031592557,
            0.6534433964656056,
            0.002462418781983633,
            12.514212104101508], requires_grad=False))
        self.register_buffer("scaler_min", torch.tensor([0.49977842782898685,
            0.4994941706919238,
            -0.020606080641338675,
            0.5002919753099604,
            -0.009400341457049127,
            0.49997065712986616,
            0.4996155459607123,
            -0.033067499716928135,
            -0.05139944510782373,
            -0.0005472203187390946,
            -8.2416517989992e-05,
            -3.700403245065246e-05,
            -0.0003170755908927612,
            -1.2027876307280915e-06,
            -7.862871723487552e-05,
            -1.7337162673375185e-06,
            -11.51308429548884], requires_grad=False))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.scaler_scale + self.scaler_min



def makeModel(dropout=0.2):
    model = nn.Sequential(
        FeatureScaler(),
        nn.Linear(featureCount, 100),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(100, 1),
        nn.Sigmoid()
    )
    return model

def makeModelOptuna(trial:Trial):
    return makeModel(trial.suggest_float("dropout", 0., 0.3))

class BaseLoss:
    trainingLossType:str = None
    shouldSwapPrediction:bool = False
    """ If True, network output is closeTo1 = bad, closeTo0 = good (aka weird TICL assoc score). If False, network output is closeTo1 = good"""
    def compute_loss_train(self, pred:torch.Tensor, train_batch:dict[str, tuple[torch.Tensor]]) -> torch.Tensor:
        """ Computes loss for a training batch
        pred is the tensor of predictions from the DNN
        train_batch is dict with keys features, genmatching, etc """
        raise NotImplementedError()

    def compute_loss_eval(self, pred:torch.Tensor, val_dataset:dict[str, torch.Tensor]) -> torch.Tensor:
        """ Computes loss for the validation dataset """
        raise NotImplementedError()
    
    def switchPredIfNeeded(self, pred:torch.Tensor) -> torch.Tensor:
        """ Return network prediction such that better is close to 1 and worse close to 0 """
        return pred # default implementation

class BinaryLoss(BaseLoss):
    """ Loss doing binary classification based on genmatching feature in batches """
    trainingLossType = "binary"

    def __init__(self, loss_module:nn.Module|None=None) -> None:
        """ Default loss is BCELoss (binary cross-entropy) """
        super().__init__()
        self.loss_module = loss_module if loss_module is not None else nn.BCELoss()
    
    def compute_loss_train(self, pred:torch.Tensor, train_batch:dict[str, tuple[torch.Tensor]]) -> torch.Tensor:
        return self.loss_module(pred, train_batch["genmatching"][0].to(pred.device))
    
    def compute_loss_eval(self, pred:torch.Tensor, val_dataset:dict[str, torch.Tensor]) -> torch.Tensor:
        return self.loss_module(pred, val_dataset["genmatching"]) # normally val_dataset is on gpu

class ContinuousAssociationScoreLoss(BaseLoss):
    """ Loss targeting the association score seed-candidate instead of binary genMatched/notGenMatched
    Avoids having to set a working point for genmatching before training
    """
    trainingLossType = "continuousAssociationScore"
    shouldSwapPrediction = True

    def __init__(self, loss_module:nn.Module|None=None) -> None:
        """ Default loss is MSELoss (mean squared error, L2 loss) """
        super().__init__()
        self.loss_module = loss_module if loss_module is not None else nn.MSELoss()
    
    def compute_loss_train(self, pred:torch.Tensor, train_batch:dict[str, tuple[torch.Tensor]]) -> torch.Tensor:
        return self.loss_module(pred, train_batch["assocScore_training"][0].to(pred.device))
    
    def compute_loss_eval(self, pred:torch.Tensor, val_dataset:dict[str, torch.Tensor]) -> torch.Tensor:
        return self.loss_module(pred, val_dataset["assocScore_training"]) # normally val_dataset is on gpu

    def switchPredIfNeeded(self, pred:torch.Tensor) -> torch.Tensor:
        """ Return network prediction such that better is close to 1 and worse close to 0 
        Swaps the weird TICL association score around so we can compare easily the DNN preds with BinaryLoss
        """
        return 1-pred

def makeLossOptuna(trial:Trial):
    lossType = trial.suggest_categorical("trainingLossType", ["binary", "continuousAssociationScore"])
    if lossType == "binary":
        loss = BinaryLoss()
    elif lossType == "continuousAssociationScore":
        loss = ContinuousAssociationScoreLoss()
    return loss


class Trainer:
    def __init__(self, model:nn.Module, loss:BaseLoss, trial:Trial, log_output=None, run_name=None, device="cpu", profiler=None, earlyStopping=True):
        """ profiler : can be used with pytorch profiler """
        self.model = model
        self.loss = loss
        self.trial = trial
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", patience=50, factor=0.5)
        self.device = device
        self.profiler = profiler
        if log_output is None:
            log_output = Path.cwd()
        else:
            log_output = Path(log_output)
        if run_name is None:
            from datetime import datetime
            run_name = f"{trial.number}-" + datetime.now().strftime("%b%d_%H-%M-%S")
        self.log_output = log_output / run_name
        self.writer = SummaryWriter(log_dir=self.log_output, )

        self.train_losses_currentEpoch = []
        self.train_batch_sizes_currentEpoch = [] # keeping the batch sizes so we have a proper average loss 
        self.train_losses_perEpoch = []
        #self.val_losses_currentEpoch = []
        #self.val_batch_sizes_currentEpoch = []
        self.val_losses_perEpoch = []

        self.current_epoch = 0
        # values for reporting values for optuna hyperparameter optimization
        self.trial_report_step = 0
        self.trial_last_report = np.inf
        self.trial_best_report = np.inf
        self.best_val_metrics = {} # keeps values of best metrics seen so far

        # Validation settings
        self.val_dnn_wps = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.992, 0.994, 0.996, 0.998, 1.]
        self.val_energy_bins = [0., 2.21427965, 3.11312909, 4.44952669, 7.04064255, 10., np.inf]
        """ energy of candidate """
        self.val_seed_energy_bins = [0., 40., 120,  200, 300, np.inf]
        self.val_eta_bins = [1.5, 2.28675771, 2.5121839 , 2.68080544, 3.15]#[1.65, 2.15, 2.75]
    
    def train_step(self, batch):
        """ Train for one batch """
        pred = torch.squeeze(self.model(batch["features"][0].to(self.device)), dim=1)
        loss = self.loss.compute_loss_train(pred, batch)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.train_losses_currentEpoch.append(loss.item())
        self.train_batch_sizes_currentEpoch.append(batch["features"][0].shape[0])

    def train_loop(self, train_dataloader):
        """ Train for one epoch """
        self.model.train()
        for batch_id, batch in enumerate(train_dataloader):
            self.train_step(batch)
        # assumes train losses are the mean of individual paris
        self.train_losses_perEpoch.append(np.average(self.train_losses_currentEpoch, weights=self.train_batch_sizes_currentEpoch))
        self.train_losses_currentEpoch = []
        self.train_batch_sizes_currentEpoch = []
    
    def val_evaluateModel(self, val_dataset):
        self.model.eval()
        with torch.no_grad():
            feats = val_dataset["features"]
            pred = torch.squeeze(self.model(feats), dim=1)

            loss = self.loss.compute_loss_eval(pred, val_dataset)

            return pred, loss.item()
            #return {"pred" : pred.cpu(), "genmatching" : genmatching.cpu(), "pred_genMatched" : pred[genmatching==1].cpu(), "pred_nonGenMatched" : pred[genmatching==0].cpu()}
    
    def val_loop(self, val_dataset_gpu:dict[str, torch.Tensor], val_dataset_cpu:dict[str, torch.Tensor], val_dataset_gpu_weighted:dict[str, torch.Tensor]|None=None):
        """ Run validation for the current epoch
        Parameters : 
            - val_dataset_gpu_weighted : Validation dataset that reproduces the same weighted sampling as in training, to be able to compare validation loss to training loss
                                set to None in case there is no weighting to be done
        """
        frequency_val = 1
        val_dataset = val_dataset_cpu
        with torch.no_grad():
            pred, loss = self.val_evaluateModel(val_dataset_gpu)
            pred = self.loss.switchPredIfNeeded(pred).cpu()
            self.val_losses_perEpoch.append(loss)
            self.writer.add_scalar("Loss/val", self.val_losses_perEpoch[-1], self.current_epoch)

            if val_dataset_gpu_weighted is not None:
                loss_weighted = self.loss.switchPredIfNeeded(self.val_evaluateModel(val_dataset_gpu_weighted)[1])
                self.writer.add_scalar("Loss/val_weighted", loss_weighted, self.current_epoch)


            if self.current_epoch % frequency_val != 0:
                return
            genmatching = val_dataset_cpu["genmatching"] # we can use genmatching var even for non-binary loss
            pred_genMatched = pred[genmatching==1]
            pred_nonGenMatched = pred[genmatching==0]

            val_etas = val_dataset["features"][:, featureNames.index("multi_eta")]
            val_energies = val_dataset["features"][:, featureNames.index("multi_en")]
            
            self.writer.add_histogram("Validation/pred_genMatched", pred_genMatched, self.current_epoch)
            self.writer.add_histogram("Validation/pred_nonGenMatched", pred_nonGenMatched, self.current_epoch)
            for b_ens in range(len(self.val_energy_bins)-1):
                sel = (val_energies>self.val_energy_bins[b_ens]) & (val_energies<self.val_energy_bins[b_ens+1])
                self.writer.add_histogram(f"Pred_genMatched_perEnergy/{self.val_energy_bins[b_ens]:.1f},{self.val_energy_bins[b_ens+1]:.1f}", pred[(genmatching==1)&(sel)], self.current_epoch)
                self.writer.add_histogram(f"Pred_nonGenMatched_perEnergy/{self.val_energy_bins[b_ens]:.1f},{self.val_energy_bins[b_ens+1]:.1f}", pred[(genmatching==0)&(sel)], self.current_epoch)

            # computing sum of superclustered energy per event (this part of the code assumes there is only one CaloParticle per event)
            def superclusteredEnergyForWP(wp:float):
                """ Compute for each event the sum of energy superclustered by the DNN at the given WP """
                return val_dataset["seedEnergy_perEvent"] + torch.scatter_add(torch.zeros(torch.max(val_dataset["eventIndex"])+1, device=val_dataset["eventIndex"].device), 0, val_dataset["eventIndex"], val_dataset["features"][:, featureNames.index("multi_en")]*(pred >= wp))

            def gaussianSigmaApprox(ratio_energyPerEventOverCP:torch.Tensor):
                """ Approximate sigma of energySupercls/CP_energy using quantiles to avoid tails """
                p, sigma_factor = 0.95, 2 # 2 sigma on either side of peak. Can also be set to 0.68, 1 for 1sigma on each side
                quantiles = torch.quantile(ratio_energyPerEventOverCP, torch.tensor([0.5-p/2, 0.5+p/2], device=ratio_energyPerEventOverCP.device)) # quantiles corresponding to mu-sigma_factor*sigma, mu+sigma_factor*sigma of a gaussian
                return torch.mean(ratio_energyPerEventOverCP).cpu(), ((quantiles[1] - quantiles[0])/(2*sigma_factor)).cpu()

            # results dict, WP -> energyBin -> value
            # energyBin can be "inclusive"
            gaussianSigmaApprox_res = collections.defaultdict(dict)
            mu_res = collections.defaultdict(dict)
            sigmaOverMu_approx_res = collections.defaultdict(dict)

            sigmaOverMu_sum = dict() # sum of sigma/mu for all energy bins, index is WP
            
            for dnn_wp in self.val_dnn_wps:
                superclsEnergy = superclusteredEnergyForWP(dnn_wp)
                superclsEnergy_ratio = superclsEnergy / val_dataset["caloParticleEnergy_perEvent"]

                mu, sigma = gaussianSigmaApprox(superclsEnergy_ratio)
                WP_str = f"WP={dnn_wp}"
                gaussianSigmaApprox_res[dnn_wp]["inclusive"] = sigma
                mu_res[dnn_wp]["inclusive"] = mu
                sigmaOverMu_approx_res[dnn_wp]["inclusive"] = sigma/mu
                self.writer.add_scalar(f"Val_Sigma_approx/{WP_str}", sigma, self.current_epoch)
                self.writer.add_scalar(f"Val_Mu/{WP_str}", mu, self.current_epoch)
                self.writer.add_scalar(f"Val_SigmaOverMu_approx/{WP_str}", sigma/mu, self.current_epoch)

                if self.current_epoch % (frequency_val*2) == 0:
                    self.writer.add_histogram(f"Val_superclsEnergy/WP={dnn_wp}", superclsEnergy, self.current_epoch)
                    self.writer.add_histogram(f"Val_superclsEnergy_ratioOverCP/WP={dnn_wp}", superclsEnergy_ratio, self.current_epoch)
                    plt.figure()
                    plt.scatter(val_dataset["caloParticleEnergy_perEvent"], superclsEnergy_ratio, s=0.5)
                    plt.xlabel("CaloParticle energy (GeV)")
                    plt.ylabel("Superclustered energy / CP energy")
                    self.writer.add_figure(f"Val_superclsEnergy_ratioOverCP/Scatter_WP={dnn_wp}", plt.gcf(), self.current_epoch)
                    plt.close()
                
                sigmaOverMu_sum[dnn_wp] = 0. 
                for b_ens in range(len(self.val_seed_energy_bins)-1):
                    sel = (val_dataset["seedEnergy_perEvent"]>self.val_seed_energy_bins[b_ens]) & (val_dataset["seedEnergy_perEvent"]<self.val_seed_energy_bins[b_ens+1])
                    mu, sigma = gaussianSigmaApprox(superclsEnergy_ratio[sel])
                    gaussianSigmaApprox_res[dnn_wp][b_ens] = sigma
                    mu_res[dnn_wp][b_ens] = mu
                    sigmaOverMu_approx_res[dnn_wp][b_ens] = sigma/mu
                    self.writer.add_scalar(f"Val_Sigma_approx_{self.val_seed_energy_bins[b_ens]}-{self.val_seed_energy_bins[b_ens+1]}/{WP_str}", sigma, self.current_epoch)
                    self.writer.add_scalar(f"Val_Mu_{self.val_seed_energy_bins[b_ens]}-{self.val_seed_energy_bins[b_ens+1]}/{WP_str}", mu, self.current_epoch)
                    self.writer.add_scalar(f"Val_SigmaOverMu_approx_{self.val_seed_energy_bins[b_ens]}-{self.val_seed_energy_bins[b_ens+1]}/{WP_str}", sigma/mu, self.current_epoch)
                    sigmaOverMu_sum[dnn_wp] += sigma/mu
                self.writer.add_scalar(f"Val_SigmaOverMu_approx/{WP_str}_sumEnergyBins", sigmaOverMu_sum[dnn_wp], self.current_epoch)

            # finding best WP
            best_wp = min(sigmaOverMu_sum, key=sigmaOverMu_sum.get)
            self.writer.add_scalar(f"Validation/bestWP", best_wp, self.current_epoch)
            # resolution at the best WP
            self.writer.add_scalar(f"Val_Sigma_approx/BestWP", gaussianSigmaApprox_res[best_wp]["inclusive"], self.current_epoch)
            self.writer.add_scalar(f"Val_Mu/BestWP", mu_res[best_wp]["inclusive"], self.current_epoch)
            self.writer.add_scalar(f"Val_SigmaOverMu_approx/BestWP", sigmaOverMu_approx_res[best_wp]["inclusive"], self.current_epoch)
            self.writer.add_scalar(f"Val_SigmaOverMu_approx/BestWP_sumEnergyBins", sigmaOverMu_sum[best_wp], self.current_epoch)
            self.trial.report(sigmaOverMu_sum[best_wp], step=self.trial_report_step)
            self.trial_last_report = sigmaOverMu_sum[best_wp]
            if self.trial_last_report < self.trial_best_report:
                self.trial_best_report = self.trial_last_report
            self.trial_report_step += 1
            if sigmaOverMu_sum[best_wp] < self.best_val_metrics.get("SigmaOverMu-BestWP_sumEnergyBins_Best", np.inf):
                self.best_val_metrics["SigmaOverMu-BestWP_sumEnergyBins_Best"] = sigmaOverMu_sum[best_wp]
            # in energy bins
            for b_ens in range(len(self.val_seed_energy_bins)-1):
                self.writer.add_scalar(f"Val_Sigma_approx_{self.val_seed_energy_bins[b_ens]}-{self.val_seed_energy_bins[b_ens+1]}/BestWP", gaussianSigmaApprox_res[best_wp][b_ens], self.current_epoch)
                self.writer.add_scalar(f"Val_Mu_{self.val_seed_energy_bins[b_ens]}-{self.val_seed_energy_bins[b_ens+1]}/BestWP", mu_res[best_wp][b_ens], self.current_epoch)
                self.writer.add_scalar(f"Val_SigmaOverMu_approx_{self.val_seed_energy_bins[b_ens]}-{self.val_seed_energy_bins[b_ens+1]}/BestWP", sigmaOverMu_approx_res[best_wp][b_ens], self.current_epoch)

            
            if self.current_epoch % (frequency_val*2) == 0:
                plt.plot(self.val_dnn_wps, [d["inclusive"] for d in sigmaOverMu_approx_res.values()])
                # plt.semilogy()
                plt.ylabel("SigmaOverMu")
                plt.xlabel("DNN WP")
                # plt.ylim(0.8,1)
                plt.grid(True)
                self.writer.add_figure("Validation_sigmaScans/SigmaOverMu_scan_inclusive", plt.gcf(), self.current_epoch)
                    
            ## Accuracy
            def computeAccuracy(cut):
                tp = len(pred_genMatched[pred_genMatched>cut])
                fp = len(pred_nonGenMatched[pred_nonGenMatched>cut])
                fn = len(pred_genMatched[pred_genMatched<cut])
                tn = len(pred_nonGenMatched[pred_nonGenMatched<cut])
                return (tp+tn)/(tp+tn+fp+fn)
            self.writer.add_scalar("Validation/accuracy_near0", computeAccuracy(0.05), self.current_epoch)
            self.writer.add_scalar("Validation/accuracy_near1", computeAccuracy(1.-0.05), self.current_epoch)

            self.writer.add_pr_curve("PR_curve", genmatching, pred, self.current_epoch)

            if self.current_epoch % (frequency_val*3) != 0:
                return
            
            genmatching = genmatching.cpu()
            pred = pred.cpu()
            ### ROC curve
            fpr, tpr, threshold = roc_curve(genmatching, pred)
            auc1 = auc(fpr, tpr)
            plt.plot(fpr,tpr,label='auc = %.1f%%'%(auc1*100.))
            # plt.semilogy()
            plt.ylabel("sig. efficiency")
            plt.xlabel("bkg. mistag rate")
            # plt.ylim(0.8,1)
            plt.grid(True)
            plt.legend(loc='lower right')
            self.writer.add_figure("Validation/ROC", plt.gcf(), self.current_epoch)
            self.writer.add_scalar("Validation/ROC_AUC", auc1, self.current_epoch)

            ## ROC curve in eta and en bins
            aucs = {}
            for b_ens in range(len(self.val_energy_bins)-1):
                for b_etas in range(len(self.val_eta_bins)-1):
                    sel = (abs(val_etas)>self.val_eta_bins[b_etas]) & (abs(val_etas)<self.val_eta_bins[b_etas+1]) & (val_energies>self.val_energy_bins[b_ens]) & (val_energies<self.val_energy_bins[b_ens+1])
                    if torch.any(sel):
                        fpr, tpr, threshold = roc_curve(genmatching[sel],pred[sel])
                        auc1 = auc(fpr, tpr)
                    else:
                        auc1 = 0.
                    aucs[b_ens,b_etas] = auc1
                    self.writer.add_scalar("Validation_ROC_binned_AUC/eta[%.2f,%.2f]_En" %(self.val_eta_bins[b_etas], self.val_eta_bins[b_etas+1]) + '[%.1f,%.1f]' %(self.val_energy_bins[b_ens], self.val_energy_bins[b_ens+1]), auc1, self.current_epoch)

                
            indices = np.array(list(aucs.keys()))
            values = np.array(list(aucs.values()))

            # Determine the matrix dimensions
            n_rows = indices[:, 0].max() + 1
            n_cols = indices[:, 1].max() + 1

            # Create an empty matrix
            matrix = np.zeros((n_rows, n_cols))

            # Fill in the matrix with the values from the dictionary
            matrix[indices[:, 0], indices[:, 1]] = values

            # Flip the matrix vertically
            matrix = np.flip(matrix, axis=0)

            # Plot the matrix
            cmap_reds = copy.copy(plt.colormaps["Reds"])
            cmap_reds.set_under('white')

            plt.imshow(matrix, cmap=cmap_reds)
            plt.xlabel(r"$|\eta_{trackster}|$")
            plt.ylabel(r"$E_{trackster}$")
            plt.colorbar()

            # Set the ticks for x-axis and y-axis
            x_ticks = np.arange(n_cols)  
            x_labels = ['[%.2f,%.2f]' %(self.val_eta_bins[b], self.val_eta_bins[b+1]) for b in range(len(self.val_eta_bins)-1)]  # Custom text labels
            plt.xticks(x_ticks, x_labels)

            y_ticks = np.arange(n_rows)  # Y-axis tick positions
            y_labels = ['[%.1f,%.1f]' %(self.val_energy_bins[b], self.val_energy_bins[b+1]) for b in range(len(self.val_energy_bins)-1)][::-1]  # Custom text labels
            plt.yticks(y_ticks, y_labels)

            # Hide the x-axis and y-axis scales
            plt.tick_params(axis='both', which='both', bottom=False, left=False)
            plt.title("AUC")
            
            self.writer.add_figure("Validation/ROC_perEnergy", plt.gcf(), self.current_epoch)
        
    
    def saveModel(self, modelName:str, format="pytorch", swapPrediction:bool|None=None):
        """ format can be pytorch or onnx 
        swapPrediction : for ONNX export, if True will swap the prediction output (doing 1-pred)
        set to None (default) it will swap it as needed to ensure DNN pred ~ 1 maps to good association, ~0 is bad
        """
        if format == "pytorch":
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.log_output / ('model_'+modelName+'.pth'))
        elif format == "onnx":
            if swapPrediction or (swapPrediction is None and self.loss.shouldSwapPrediction):
                class SwappingModule(nn.Module):
                    def __init__(self, model) -> None:
                        super().__init__()
                        self.model = model

                    def forward(self, input: torch.Tensor) -> torch.Tensor:
                        return (1-self.model(input))
                model_maybeSwapped = SwappingModule(self.model)
            else:
                model_maybeSwapped = self.model

            torch.onnx.export(
                model_maybeSwapped,
                torch.ones((1, featureCount), device=self.device), # placeholder input
                self.log_output / ('model_'+modelName+'.onnx'),
                export_params=True,
                verbose=False,
                input_names=["input"],
                output_names=["output"],
                opset_version=10,
                dynamic_axes={"input" : {0:"batch_size"}, "output": {0:"batch_size"}}
            )
    
    def reloadModel(self, checkpointPath:str):
        checkpoint = torch.load(checkpointPath)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]

    
    def full_train(self, train_dataset, val_dataset, nepochs=10):
        batch_size = self.trial.suggest_int("batchSize", 64, 16384, log=True)
        weightSamples = self.trial.suggest_categorical("weightSamples", [True, False])

        val_dataset_cpu = {key : tensor.cpu() for key, tensor in val_dataset.items()}
        val_dataset_gpu = {key : tensor.to(self.device) for key, tensor in val_dataset.items()}
        if self.device != "cpu":
            dataloader_kwargs = dict(num_workers=2, pin_memory=True, pin_memory_device=self.device)
        else:
            dataloader_kwargs = dict()
        if weightSamples:
            # weight samples so we are equally likely to train on genmatched seed-candidate pairs than not
            if self.loss.trainingLossType == "binary":
                gen = train_dataset[:]["genmatching"][0] == 1
                gen_val = val_dataset_gpu["genmatching"] == 1
            else:
                # Put 0.2 as something vaguely equivalent to what is done in binary case
                gen = train_dataset[:]["assocScore_training"][0] <= 0.2
                gen_val = val_dataset_gpu["assocScore_training"] <= 0.2
            # using replacement=True is important as otherwise the sampler runs out of genmatched samples and just spits out batches with only non-genmatched pairs, which is very bad for training !
            sampler = WeightedRandomSampler(torch.where(gen, torch.sum(~gen)/torch.sum(gen), 1.), gen.shape[0], replacement=True)
            shuffle = False
            # make indices to reproduce weighted sampling in validation dataset. genmatched pairs will be sampled 3 times on average
            weighted_val_indices = torch.tensor(list(WeightedRandomSampler(torch.where(gen_val, torch.sum(~gen_val)/torch.sum(gen_val), 1.), torch.sum(gen_val).item()*3, replacement=True)))
            val_dataset_gpu_weighted = {"features" : val_dataset_gpu["features"][weighted_val_indices],
                                        "genmatching" : val_dataset_gpu["genmatching"][weighted_val_indices], 
                                        "assocScore_training" : val_dataset_gpu["assocScore_training"][weighted_val_indices]}
        else:
            sampler = None
            shuffle = True
            val_dataset_gpu_weighted = None
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, **dataloader_kwargs)
        #val_dataloader = DataLoader(val_dataset, batch_size=100000, **dataloader_kwargs)

                
        self.writer.add_custom_scalars({
            "Loss" : {
                "Losses_comparison_weighted" : ["Multiline", ["Loss/train", "Loss/val_weighted"]],
                "Losses_comparison_unweighted" : ["Multiline", ["Loss/train", "Loss/val"]]
            },
            "Validation_SigmaAndMu" : {
                "Sigma_approx" : ["Multiline", [f"Val_Sigma_approx/WP={dnn_wp}" for dnn_wp in self.val_dnn_wps]], 
                "Mu" : ["Multiline", [f"Val_Mu/WP={dnn_wp}" for dnn_wp in self.val_dnn_wps]],
                "SigmaOverMu_approx" : ["Multiline", [f"Val_SigmaOverMu_approx/WP={dnn_wp}" for dnn_wp in self.val_dnn_wps]],
            },
            "Validation_SigmaAndMu_BestWP" : {
                "Sigma_approx" : ["Multiline", [f"Val_Sigma_approx_{self.val_seed_energy_bins[b_ens]}-{self.val_seed_energy_bins[b_ens+1]}/BestWP" for b_ens in range(len(self.val_seed_energy_bins)-1)]], 
                "Mu" : ["Multiline", [f"Val_Mu_{self.val_seed_energy_bins[b_ens]}-{self.val_seed_energy_bins[b_ens+1]}/BestWP" for b_ens in range(len(self.val_seed_energy_bins)-1)]],
                "SigmaOverMu_approx" : ["Multiline", [f"Val_SigmaOverMu_approx_{self.val_seed_energy_bins[b_ens]}-{self.val_seed_energy_bins[b_ens+1]}/BestWP" for b_ens in range(len(self.val_seed_energy_bins)-1)]],
            },
            "Validation_ROC" : {
                "binned_AUC" : ["Multiline", ["Validation_ROC_binned_AUC/eta[%.2f,%.2f]_En" %(self.val_eta_bins[b_etas], self.val_eta_bins[b_etas+1]) + '[%.1f,%.1f]' %(self.val_energy_bins[b_ens], self.val_energy_bins[b_ens+1]) 
                                            for b_ens in range(len(self.val_energy_bins)-1) for b_etas in range(len(self.val_eta_bins)-1)]]
            }
        })

        if self.profiler: self.profiler.start()
        try:
            for epoch in (pbar := tqdm(range(nepochs))):
                self.current_epoch += 1
                if self.profiler: self.profiler.step()
                self.train_loop(train_dataloader)
                self.writer.add_scalar("Loss/train", self.train_losses_perEpoch[-1], epoch)

                self.val_loop(val_dataset_gpu=val_dataset_gpu, val_dataset_cpu=val_dataset_cpu, val_dataset_gpu_weighted=val_dataset_gpu_weighted)
                pbar.set_description(f"train_loss = {self.train_losses_perEpoch[-1]}, val_loss = {self.val_losses_perEpoch[-1]}")

                self.scheduler.step(self.val_losses_perEpoch[-1])
                self.writer.add_scalar("Training/learning_rate", self.scheduler.get_last_lr()[0], epoch)
                scheduler_ended = True
                for i, param_group in enumerate(self.optimizer.param_groups):
                    old_lr = float(param_group['lr'])
                    new_lr = max(old_lr * self.scheduler.factor, self.scheduler.min_lrs[i])
                    if old_lr - new_lr > self.scheduler.eps:
                        scheduler_ended = False
                if scheduler_ended and self.scheduler.num_bad_epochs > 20: # early stopping when no val loss improvement and already at end of LR scheduler
                    return self.trial_best_report

                if epoch % 5 == 0:
                    self.saveModel(f"epoch{epoch}")
                if epoch % 10 == 0:
                    self.saveModel(f"epoch{epoch}", format="onnx")
                
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
        finally:
            if self.profiler: self.profiler.stop()
            try:
                self.writer.add_hparams(self.trial.params, {"SigmaOverMu-sum-Best" : self.best_val_metrics["SigmaOverMu-BestWP_sumEnergyBins_Best"]})
            except KeyError: pass
            self.writer.close()
            self.saveModel(f"lastEpoch_{epoch}")
            self.saveModel(f"lastEpoch_{epoch}", format="onnx")

        return self.trial_last_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='superclusteringDNNTrainer',
                    description='Training of superclustering DNN for HGCAL')
    parser.add_argument("-i", "--input", help="Path to folder or file holding the superclustering sample dumper output. Passed to uproot.concatenate", required=True)
    parser.add_argument("-o", "--output", help="Directory to put output (trained network, tensorboard logs, etc)")
    parser.add_argument("-D", "--device", help="Device for pytorch, ex : cpu or cuda:0", default="cpu")
    parser.add_argument("-r", "--resume", help="Resume from checkpoint. Path to pytorch saved model checkpoint", required=False)
    parser.add_argument("-e", "--nEpochs", help="Number of epochs to train for", default=10, type=int)
    parser.add_argument("-b", "--batchSize", help="Batch size", default=8000, type=int)
    parser.add_argument("--dropout", help="Dropout probability", default=0.2, type=float)
    parser.add_argument("--weightSamples", help="weight samples so we are equally likely to train on genmatched seed-candidate pairs than not", default=True, action="store_true")
    parser.add_argument("--trainingLossType", help="Type of training loss", choices=["binary", "continuousAssociationScore"], default="binary")
    args = parser.parse_args()

    trial = FixedTrial({
        "dropout" : args.dropout,
        "trainingLossType" : args.trainingLossType,
        "batchSize" : args.batchSize,
        "lr" : 1e-3,
        "weightSamples" : args.weightSamples
    })
    device = args.device
    model = makeModelOptuna(trial)
    loss = makeLossOptuna(trial)
    trainer = Trainer(model.to(device), loss, device=device, log_output=args.output)
    if args.resume:
        trainer.reloadModel(args.resume)
    print("Loading dataset...")
    datasets = makeDatasetsTrainVal_fromCache(args.input, device_valDataset=device, trainingLossType=trial.params["trainingLossType"])
    print("Dataset loaded into RAM")
    trainer.full_train(datasets["trainDataset"], datasets["valDataset"], nepochs=args.nEpochs)