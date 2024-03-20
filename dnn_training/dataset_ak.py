import uproot
import awkward as ak
import numpy as np

def loadDataset_ak(inputFiles):
    return uproot.concatenate(inputFiles)

def zipDataset(data_ak:ak.Array):
    return ak.zip({key: data_ak[key] for key in data_ak.fields if key != "Event"}, with_name="inferencePair")

def makeTarget(data_ak:ak.Array, seed_assocScore_threshold=0.4, candidate_assocScore_threshold=0.6):
    genMatching = (
     (data_ak.candidateTracksterBestAssociation_simTsIdx == data_ak.seedTracksterBestAssociation_simTsIdx) &
     (data_ak.seedTracksterBestAssociationScore < seed_assocScore_threshold) & 
     (data_ak.candidateTracksterAssociationWithSeed_score < candidate_assocScore_threshold)
    )
    return ak.with_field(data_ak, genMatching, "genMatching")

def makeFeatures(data_ak:ak.Array, featureNames:list[str]) -> list[np.ndarray]:
    return [ak.to_numpy(ak.flatten(data_ak["feature_" + featName])) for featName in featureNames]
