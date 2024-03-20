import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import uproot
import awkward as ak
from tqdm.auto import tqdm
import argparse
from pathlib import Path

from dnn_training.dataset_ak import *
from dnn_training.dataset_torch import *

# case candidateTracksterBestAssociation_simTsIdx != seedTracksterBestAssociation_simTsIdx
# -> avoid at maximum

# case candidateTracksterBestAssociation_simTsIdx == seedTracksterBestAssociation_simTsIdx

# seedTracksterBestAssociationScore good && candidateTracksterAssociationWithSeed_score good -> to target
# seedTracksterBestAssociationScore good && candidateTracksterAssociationWithSeed_score bad -> to avoid
# seedTracksterBestAssociationScore bad : what to do ?

featureCount = 17

def makeModelLoss():
    model = nn.Sequential(
        nn.Linear(featureCount, 100),
        nn.ReLU(),
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
        nn.Sigmoid()
    )
    loss = nn.BCELoss()
    return model, loss


def makeDatasetsTrainVal(inputFolder:str="/grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v13/superclsDumper_1.root", device="cpu"):
    """ Set device to gpu only if you want to move the whole dataset to GPU then use num_workers=0 in DataLoader. THis is usually slower than keeping tensors on CPU then using num_workers > 1 """
    dataset_ak = makeTarget(zipDataset(loadDataset_ak(inputFolder)))
    dataset_torch = makeTorchDataset(makeFeatures(dataset_ak,  ['DeltaEtaBaryc', 'DeltaPhiBaryc', 'multi_en', 'multi_eta', 'multi_pt', 'seedEta','seedPhi','seedEn', 'seedPt', 'theta', 'theta_xz_seedFrame', 'theta_yz_seedFrame', 'theta_xy_cmsFrame', 'theta_yz_cmsFrame', 'theta_xz_cmsFrame', 'explVar', 'explVarRatio']),
                    ak.to_numpy(ak.flatten(dataset_ak.genMatching)), device=device)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset_torch, [0.7, 0.3])
    return train_dataset, val_dataset



class Trainer:
    def __init__(self, model:nn.Module, loss_fn:nn.Module, log_output=None, run_name=None, device="cpu"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", patience=2)
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
        self.train_losses_perEpoch = []
        self.val_losses_currentEpoch = []
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

    def train_loop(self, train_dataloader):
        """ Train for one epoch """
        self.model.train()
        for batch_id, batch in enumerate(train_dataloader):
            self.train_step(batch)
        self.train_losses_perEpoch.append(sum(self.train_losses_currentEpoch))
        self.train_losses_currentEpoch = []
    
    def val_step(self, batch):
        with torch.no_grad():
            feats = batch["features"][0].to(self.device)
            genmatching = batch["genmatching"][0].to(self.device)
            pred = torch.squeeze(self.model(feats), dim=1)

            loss = self.loss_fn(pred, genmatching)
            self.val_losses_currentEpoch.append(loss.item())

            return {"pred_genMatched" : pred[genmatching==1].cpu(), "pred_nonGenMatched" : pred[genmatching==0].cpu()}
    
    def val_loop(self, val_dataloader):
        self.model.eval()
        pred_vals = []
        for batch_id, batch in enumerate(val_dataloader):
            pred_vals.append(self.val_step(batch))
        self.val_losses_perEpoch.append(sum(self.val_losses_currentEpoch))
        self.val_losses_currentEpoch = []
        self.writer.add_histogram("Validation/pred_genMatched", torch.cat([d["pred_genMatched"] for d in pred_vals]), self.current_epoch)
        self.writer.add_histogram("Validation/pred_nonGenMatched", torch.cat([d["pred_nonGenMatched"] for d in pred_vals]), self.current_epoch)
    
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

            self.val_loop(val_dataloader)
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
    parser.add_argument("-e", "--nEpochs", help="Number of epochs to train for", default=10, type=int)
    parser.add_argument("-r", "--resume", help="Resume from checkpoint. Path to pytorch saved model checkpoint", required=False)
    args = parser.parse_args()

    device = args.device
    model, loss = makeModelLoss()
    trainer = Trainer(model.to(device), loss, device=device, log_output=args.output)
    if "resume" in args:
        trainer.reloadModel(args.resume)
    print("Loading dataset...")
    train_dataset, val_dataset = makeDatasetsTrainVal(args.input)
    print("Dataset loaded into RAM")
    trainer.full_train(train_dataset, val_dataset, nepochs=args.nEpochs)