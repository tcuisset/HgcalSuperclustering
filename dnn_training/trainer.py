import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import uproot
import awkward as ak
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import copy

from dnn_training.dataset_ak import *
from dnn_training.dataset_torch import *

# case candidateTracksterBestAssociation_simTsIdx != seedTracksterBestAssociation_simTsIdx
# -> avoid at maximum

# case candidateTracksterBestAssociation_simTsIdx == seedTracksterBestAssociation_simTsIdx

# seedTracksterBestAssociationScore good && candidateTracksterAssociationWithSeed_score good -> to target
# seedTracksterBestAssociationScore good && candidateTracksterAssociationWithSeed_score bad -> to avoid
# seedTracksterBestAssociationScore bad : what to do ?

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



def makeModelLoss(dropout=0.2):
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
    loss = nn.BCELoss()
    return model, loss


def makeDatasetsTrainVal(inputFolder:str="/grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v13/superclsDumper_1.root", device="cpu"):
    """ Set device to gpu only if you want to move the whole dataset to GPU then use num_workers=0 in DataLoader. THis is usually slower than keeping tensors on CPU then using num_workers > 1 """
    dataset_ak = makeTarget(selectSeedOnly(zipDataset(loadDataset_ak(inputFolder))))
    dataset_torch = makeTorchDataset(makeFeatures(dataset_ak,  ['DeltaEtaBaryc', 'DeltaPhiBaryc', 'multi_en', 'multi_eta', 'multi_pt', 'seedEta','seedPhi','seedEn', 'seedPt', 'theta', 'theta_xz_seedFrame', 'theta_yz_seedFrame', 'theta_xy_cmsFrame', 'theta_yz_cmsFrame', 'theta_xz_cmsFrame', 'explVar', 'explVarRatio']),
                    ak.to_numpy(ak.flatten(dataset_ak.genMatching)), device=device)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset_torch, [0.7, 0.3])
    return train_dataset, val_dataset



class Trainer:
    def __init__(self, model:nn.Module, loss_fn:nn.Module, log_output=None, run_name=None, device="cpu"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", patience=30, factor=0.5)
        self.device = device
        if log_output is None:
            log_output = Path.cwd()
        else:
            log_output = Path(log_output)
        if run_name is None:
            from datetime import datetime
            run_name = datetime.now().strftime("%b%d_%H-%M-%S")
        self.log_output = log_output / run_name
        self.writer = SummaryWriter(log_dir=self.log_output, )

        self.train_losses_currentEpoch = []
        self.train_batch_sizes_currentEpoch = [] # keeping the batch sizes so we have a proper average loss 
        self.train_losses_perEpoch = []
        self.val_losses_currentEpoch = []
        self.val_batch_sizes_currentEpoch = []
        self.val_losses_perEpoch = []

        self.current_epoch = 0
    
    def train_step(self, batch):
        """ Train for one batch """
        pred = torch.squeeze(self.model(batch["features"][0].to(self.device)), dim=1)
        loss = self.loss_fn(pred, batch["genmatching"][0].to(self.device))

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
        self.train_losses_perEpoch.append(np.average(self.train_losses_currentEpoch, weights=self.train_batch_sizes_currentEpoch))
        self.train_losses_currentEpoch = []
        self.train_batch_sizes_currentEpoch = []
    
    def val_step(self, batch):
        with torch.no_grad():
            feats = batch["features"][0].to(self.device)
            genmatching = batch["genmatching"][0].to(self.device)
            pred = torch.squeeze(self.model(feats), dim=1)

            loss = self.loss_fn(pred, genmatching)
            self.val_losses_currentEpoch.append(loss.item())
            self.val_batch_sizes_currentEpoch.append(feats.shape[0])

            return {"pred" : pred.cpu(), "genmatching" : genmatching.cpu(), "pred_genMatched" : pred[genmatching==1].cpu(), "pred_nonGenMatched" : pred[genmatching==0].cpu()}
    
    def val_loop(self, val_dataloader, val_dataset:Dataset):
        self.model.eval()
        pred_vals = []
        for batch_id, batch in enumerate(val_dataloader):
            pred_vals.append(self.val_step(batch))
        self.val_losses_perEpoch.append(np.average(self.val_losses_currentEpoch, weights=self.val_batch_sizes_currentEpoch))
        self.val_losses_currentEpoch = []
        self.val_batch_sizes_currentEpoch = []

        pred = torch.cat([d["pred"] for d in pred_vals])
        genmatching = torch.cat([d["genmatching"] for d in pred_vals])
        pred_genMatched = torch.cat([d["pred_genMatched"] for d in pred_vals])
        pred_nonGenMatched = torch.cat([d["pred_nonGenMatched"] for d in pred_vals])
        self.writer.add_histogram("Validation/pred_genMatched", pred_genMatched, self.current_epoch)
        self.writer.add_histogram("Validation/pred_nonGenMatched", pred_nonGenMatched, self.current_epoch)

        def computeAccuracy(cut):
            tp = len(pred_genMatched[pred_genMatched>cut])
            fp = len(pred_nonGenMatched[pred_nonGenMatched>cut])
            fn = len(pred_genMatched[pred_genMatched<cut])
            tn = len(pred_nonGenMatched[pred_nonGenMatched<cut])
            return (tp+tn)/(tp+tn+fp+fn)
        self.writer.add_scalar("Validation/accuracy_near0", computeAccuracy(0.05), self.current_epoch)
        self.writer.add_scalar("Validation/accuracy_near1", computeAccuracy(1.-0.05), self.current_epoch)

        if self.current_epoch % 5 != 0:
            return
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
        ens = [0., 2.21427965, 3.11312909, 4.44952669, 7.04064255, 10., np.inf]
        etas = [1.5, 2.28675771, 2.5121839 , 2.68080544, 3.15]#[1.65, 2.15, 2.75]
        aucs = {}
        val_etas = val_dataset[:]["features"][0][:, 3] # index 3 is multi_eta
        val_energies = val_dataset[:]["features"][0][:, 2] # index 2 is multi_en
        for b_ens in range(len(ens)-1):
            for b_etas in range(len(etas)-1):
                sel = (abs(val_etas)>etas[b_etas]) & (abs(val_etas)<etas[b_etas+1]) & (val_energies>ens[b_ens]) & (val_energies<ens[b_ens+1])
                if torch.any(sel):
                    fpr, tpr, threshold = roc_curve(genmatching[sel],pred[sel])
                    auc1 = auc(fpr, tpr)
                else:
                    auc1 = 0.
                aucs[b_ens,b_etas] = auc1
                self.writer.add_scalar("Validation_ROC_binned_AUC/eta[%.2f,%.2f]_En" %(etas[b_etas], etas[b_etas+1]) + '[%.1f,%.1f]' %(ens[b_ens], ens[b_ens+1]), auc1, self.current_epoch)

             
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
        x_labels = ['[%.2f,%.2f]' %(etas[b], etas[b+1]) for b in range(len(etas)-1)]  # Custom text labels
        plt.xticks(x_ticks, x_labels)

        y_ticks = np.arange(n_rows)  # Y-axis tick positions
        y_labels = ['[%.1f,%.1f]' %(ens[b], ens[b+1]) for b in range(len(ens)-1)][::-1]  # Custom text labels
        plt.yticks(y_ticks, y_labels)

        # Hide the x-axis and y-axis scales
        plt.tick_params(axis='both', which='both', bottom=False, left=False)
        plt.title("AUC")
        
        self.writer.add_figure("Validation/ROC_perEnergy", plt.gcf(), self.current_epoch)
        
    
    def saveModel(self, modelName:str, format="pytorch"):
        """ format can be pytorch or onnx """
        if format == "pytorch":
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.log_output / ('model_'+modelName+'.pth'))
        elif format == "onnx":
            torch.onnx.export(
                self.model,
                torch.ones((1, featureCount)), # placeholder input
                self.log_output / ('model_'+modelName+'.onnx'),
                export_params=True,
                verbose=True,
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

    
    def full_train(self, train_dataset, val_dataset, nepochs=10, batch_size=8000):
        if self.device != "cpu":
            dataloader_kwargs = dict(num_workers=10, pin_memory=True, pin_memory_device=self.device)
        else:
            dataloader_kwargs = dict()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
        val_dataloader = DataLoader(val_dataset, batch_size=100000, **dataloader_kwargs)
        for epoch in tqdm(range(nepochs)):
            self.current_epoch += 1
            self.train_loop(train_dataloader)
            tqdm.write(f"train_loss = {self.train_losses_perEpoch[-1]}")
            self.writer.add_scalar("Loss/train", self.train_losses_perEpoch[-1], epoch)

            self.val_loop(val_dataloader, val_dataset)
            tqdm.write(f"val_loss = {self.val_losses_perEpoch[-1]}")
            self.writer.add_scalar("Loss/val", self.val_losses_perEpoch[-1], epoch)

            self.scheduler.step(self.val_losses_perEpoch[-1])
            self.writer.add_scalar("Training/learning_rate", self.scheduler.get_last_lr()[0], epoch)

            if epoch % 4 == 0:
                self.saveModel(f"epoch{epoch}")
        
        self.writer.close()
        self.saveModel(f"lastEpoch_{epoch}")





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
    args = parser.parse_args()

    device = args.device
    model, loss = makeModelLoss(dropout=args.dropout)
    trainer = Trainer(model.to(device), loss, device=device, log_output=args.output)
    if args.resume:
        trainer.reloadModel(args.resume)
    print("Loading dataset...")
    train_dataset, val_dataset = makeDatasetsTrainVal(args.input)
    print("Dataset loaded into RAM")
    trainer.full_train(train_dataset, val_dataset, nepochs=args.nEpochs, batch_size=args.batchSize)