import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, StackDataset
from pathlib import Path
from typing import Literal

######## Awkward array loading code
def loadDataset_ak(inputFiles):
    return uproot.concatenate(inputFiles)

def zipDataset(data_ak:ak.Array):
    return ak.zip({key: data_ak[key] for key in data_ak.fields if key != "Event"}, with_name="inferencePair")

def selectSeedOnly(d_zipped:ak.Array) -> ak.Array:
    """ Select from all inference pairs only those containing the trackster with highest pt"""
    # fill_none is to remove the optional type
    return d_zipped[ak.fill_none(d_zipped.seedTracksterIdx == d_zipped.seedTracksterIdx[ak.argmax(d_zipped.feature_seedPt, axis=-1, keepdims=True)][:, 0], [], 0)]

def removeBadSeeds(data_ak:ak.Array, seed_assocScore_threshold=0.2) -> ak.Array:
    """ Remove from the sample the events for which the seed has a bad association score to any CaloParticle (ie the seed is pileup contaminated)."""
    data_ak = data_ak[data_ak.seedTracksterBestAssociationScore < seed_assocScore_threshold]
    return data_ak[ak.num(data_ak, axis=1) > 0] # remove events that have no pairs left

# candidate assoc score was picked to maximize resolution of superclusters built assuming "perfect" superclustering
# seed threshold is kind of arbitrary, designed to train the network not to supercluster when fed with random input
# we should probably consider 
def makeTargetBinary(data_ak:ak.Array, seed_assocScore_threshold=0.1, candidate_assocScore_threshold=0.15): 
    genMatching = (
     (data_ak.candidateTracksterBestAssociation_simTsIdx == data_ak.seedTracksterBestAssociation_simTsIdx) &
     (data_ak.seedTracksterBestAssociationScore < seed_assocScore_threshold) & 
     (data_ak.candidateTracksterAssociationWithSeed_score < candidate_assocScore_threshold)
    )
    return ak.with_field(data_ak, genMatching, "genMatching")

def makeTargetContinuous(data_ak:ak.Array):
    """ Make DNN target the association score as continuous function of [0, 1] rather than binary classification
    Note that target close to 1 means bad association, close to 0 means good association to CaloParticle
    """

    assocScore_training = 1 - (
        (1 - data_ak.candidateTracksterAssociationWithSeed_score)
        # set worst target score in case candidate is not sim associated with seed
        * (data_ak.candidateTracksterBestAssociation_simTsIdx == data_ak.seedTracksterBestAssociation_simTsIdx)
        # Idea : reduce target score in case seed trackster has lots of PU contamination (not tested)
        # maybe there should be some factor to increase the effect of the deviation of the score from 1
        # * (1-data_ak.seedTracksterBestAssociationScore)
    )
    return ak.with_field(data_ak, assocScore_training, "assocScore_training")

def makeFeatures(data_ak:ak.Array, featureNames:list[str]) -> list[np.ndarray]:
    return [ak.to_numpy(ak.flatten(data_ak["feature_" + featName])) for featName in featureNames]

def makeEventIndex(data_ak:ak.Array) -> np.ndarray:
    """ Make an array giving for each seed-cand pair, the event index it belongs to. Meant to be used for validation. """
    return ak.to_numpy(ak.flatten(ak.broadcast_arrays(ak.local_index(data_ak.seedTracksterIdx, axis=0), data_ak.seedTracksterIdx)[0]))


####### Pytorch datasets 

def makeDatasetsTrainVal(data_ak:ak.Array, featureNames:list[str]=['DeltaEtaBaryc', 'DeltaPhiBaryc', 'multi_en', 'multi_eta', 'multi_pt', 'seedEta','seedPhi','seedEn', 'seedPt', 'theta', 'theta_xz_seedFrame', 'theta_yz_seedFrame', 'theta_xy_cmsFrame', 'theta_yz_cmsFrame', 'theta_xz_cmsFrame', 'explVar', 'explVarRatio'],
        val_fraction=0.2, device_valDataset="cpu", trainingLossType:Literal["binary", "continuousAssociationScore"]="binary"):
    """ Makes a pair of trainDataset, valDataset splitting events randomly """
    # We split the dataset per event to avoid having events split across train and val set (for validation we need the full events present)
    event_count = ak.num(data_ak, axis=0) # nb of events
    rand_indices = torch.randperm(event_count).tolist()
    border_train_val_idx = int((1-val_fraction)*event_count)
    indices_train, indices_val = rand_indices[:border_train_val_idx], rand_indices[border_train_val_idx:]
    data_ak_train, data_ak_val = data_ak[indices_train], data_ak[indices_val]

    device_trainDataset = "cpu" # for training, since we need to shuffle the dataset, we keep it on CPU then move the batches to GPU one at a time
    if trainingLossType == "binary":
        trainDataset = StackDataset(
            features=TensorDataset(torch.tensor(np.stack(makeFeatures(data_ak_train, featureNames=featureNames), axis=1), device=device_trainDataset)),
            genmatching=TensorDataset(torch.tensor(ak.to_numpy(ak.flatten(data_ak_train.genMatching)), device=device_trainDataset, dtype=torch.float32)),
        )
    elif trainingLossType == "continuousAssociationScore":
        trainDataset = StackDataset(
            features=TensorDataset(torch.tensor(np.stack(makeFeatures(data_ak_train, featureNames=featureNames), axis=1), device=device_trainDataset)),
            assocScore_training=TensorDataset(torch.tensor(ak.to_numpy(ak.flatten(data_ak_train.assocScore_training)), device=device_trainDataset, dtype=torch.float32)),
        )
    
    # for validation, we move everything to GPU now. Also no batching for validation (way easier to keep track of indices)
    valDataset = {"features" : torch.tensor(np.stack(makeFeatures(data_ak_val, featureNames=featureNames), axis=1), device=device_valDataset),
            "genmatching" : torch.tensor(ak.to_numpy(ak.flatten(data_ak_val.genMatching)), device=device_valDataset, dtype=torch.float32),
            "assocScore_training" : torch.tensor(ak.to_numpy(ak.flatten(data_ak_val.assocScore_training)), device=device_valDataset, dtype=torch.float32),
            "seedTracksterBestAssociationScore" : torch.tensor(ak.to_numpy(ak.flatten(data_ak_val.seedTracksterBestAssociationScore)), device=device_valDataset),
            "candidateTracksterAssociationWithSeed_score" : torch.tensor(ak.to_numpy(ak.flatten(data_ak_val.candidateTracksterAssociationWithSeed_score)), device=device_valDataset),
            "caloParticleEnergy_perEvent" : torch.tensor(ak.to_numpy(data_ak_val.seedTracksterBestAssociation_caloParticleEnergy[:, 0]), device=device_valDataset),
            "seedEnergy_perEvent" : torch.tensor(ak.to_numpy(data_ak_val.feature_seedEn[:, 0]), device=device_valDataset),
            "eventIndex" : torch.tensor(makeEventIndex(data_ak_val), device=device_valDataset)}

    return {"trainDataset" : trainDataset, "valDataset" : valDataset}

def makeDatasetsTrainVal_fromCache(inputFolder:str, trainingLossType:Literal["binary", "continuousAssociationScore"]="binary", **kwargs):
    cached_dataset_path = Path("/".join(inputFolder.split("/")[:-1])) / f"cached_torch_dataset_{trainingLossType}.pt"
    if cached_dataset_path.is_file():
        return torch.load(cached_dataset_path, map_location="cpu")
    else:
        dataset_ak = makeTargetContinuous(makeTargetBinary(removeBadSeeds(selectSeedOnly(zipDataset(loadDataset_ak({inputFolder:"superclusteringSampleDumper/superclusteringTraining"}))))))
        dataset_dict = makeDatasetsTrainVal(dataset_ak, trainingLossType=trainingLossType, **kwargs)
        torch.save(dataset_dict, cached_dataset_path)
        return dataset_dict

# def makeTorchDataset(features:list[np.ndarray], genmatching:np.ndarray, device="cpu"):
#     return StackDataset(features=TensorDataset(torch.tensor(np.stack(features, axis=1), device=device)),
#         genmatching=TensorDataset(torch.tensor(genmatching, device=device, dtype=torch.float32)))

# def makeTorchValidationDataset(features:list[np.ndarray], data_ak:ak.Array, device="cpu"):
#     return 