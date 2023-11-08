from functools import cached_property

import awkward as ak
import pandas as pd
import uproot

def supercluster_properties(supercluster_df:pd.DataFrame, tracksters:ak.Array) -> pd.DataFrame:
    tracksters_df = ak.to_dataframe(tracksters[["raw_energy", "regressed_energy"]], 
            levelname=lambda x : {0:"eventInternal", 1:"ts_id"}[x])

    return supercluster_df.join(tracksters_df, on=["eventInternal", "ts_id"])


def assocs_zip_recoToSim(assocs_unzipped:ak.Array) -> ak.Array:
    """ Zip associations array into records
    From type: nevts * {
        tsCLUE3D_recoToSim_SC: var * var * uint32,
        tsCLUE3D_recoToSim_SC_score: var * var * float32,
        tsCLUE3D_recoToSim_SC_sharedE: var * var * float32
    }
    To type: nevts * var (nbOfTsForCurEvent) * var (nbOfAssocsForCurTs) * {
        ts_id: int64,
        simts_id: uint32,
        score: float32,
        sharedE: float32
    }
    """
    return ak.zip({
        "ts_id":ak.local_index(assocs_unzipped.tsCLUE3D_recoToSim_SC, axis=1),
        "simts_id":assocs_unzipped.tsCLUE3D_recoToSim_SC,
        "score":assocs_unzipped.tsCLUE3D_recoToSim_SC_score,
        "sharedE":assocs_unzipped.tsCLUE3D_recoToSim_SC_sharedE})

def assocs_zip_simToReco(assocs_unzipped:ak.Array) -> ak.Array:
    """ Zip associations array into records
    From type: nevts * {
        tsCLUE3D_simToReco_CP: var * var * uint32,
        tsCLUE3D_simToReco_CP_score: var * var * float32,
        tsCLUE3D_simToReco_CP_sharedE: var * var * float32
    }
    To type: nevts * var (nbOfCPPerEvent = 2 normally) * var (nbOfAssocsForCurCP) * {
        simts_id: int64,
        ts_id: uint32,
        score: float32,
        sharedE: float32
    }
    """
    return ak.zip({
        "simts_id":ak.local_index(assocs_unzipped.tsCLUE3D_simToReco_CP, axis=1),
        "ts_id":assocs_unzipped.tsCLUE3D_simToReco_CP,
        "score":assocs_unzipped.tsCLUE3D_simToReco_CP_score,
        "sharedE":assocs_unzipped.tsCLUE3D_simToReco_CP_sharedE})

def assocs_largestScore(assocs_zipped:ak.Array) -> ak.Array:
    """ Selects the association with the best (lowest) score for each trackster 
    Trackster with no associations are dropped (therefore ts_id can be different from the index in the list)
    From type: nevts * var (nbOfTsForCurEvent) * var (nbOfAssocsForCurTs) * {
        ts_id: int64,
        simts_id: uint32,
        score: float32,
        sharedE: float32
    }
    to type: nevts * var (nbOfTsForCurEvent) * {
        ts_id: int64,
        simts_id: uint32,
        score: float32,
        sharedE: float32
    }
    """
    idx_sort = ak.argsort(assocs_zipped.score, ascending=True) # sort by score
    # Take the first assoc, putting None in case a trackster has no assocation
    # Then drop the None
    return ak.drop_none(ak.firsts(assocs_zipped[idx_sort], axis=-1), axis=-1)

def assocs_toDf(assocs_zipped:ak.Array, score_threshold=0.5) -> pd.DataFrame:
    """ Makes a pandas dataframe of associations
    Parameters : 
     - score_threshold : can be a float in [0, 1], in which case it will keep only associations with score lower than this threshold (lower score = better).
                        If None, will not place any cut on score
    Index : eventInternal, ts_id
    Columns: simts_id score sharedE
    """
    assocs = assocs_largestScore(assocs_zipped)
    if score_threshold is not None:
        assocs = assocs[assocs_largestScore(assocs_zipped).score < score_threshold]
    return (ak.to_dataframe(
        assocs,
        levelname=lambda x : {0:"eventInternal", 1:"mapping"}[x])
        .reset_index("mapping", drop=True).set_index("ts_id", append=True)
    )

def tracksters_toDf(tracksters:ak.Array) -> pd.DataFrame:
    """ Makes a dataframe with all tracksters
    Index : eventInternal, ts_id
    """
    return (ak.to_dataframe(
            tracksters[["raw_energy", "barycenter_x", "barycenter_y", "barycenter_z"]],
            levelname=lambda x : {0:"eventInternal", 1:"ts_id"}[x])
    )

def tracksters_joinWithSimTracksters(tracksters:ak.Array, simTracksters:ak.Array, assoc:ak.Array, score_threshold=assocs_toDf.__defaults__[0]) -> pd.DataFrame:
    """ Make a merged dataframe holding trackster information joined with the sim tracksters information.
    Only tracksters with an association are kept.
    Index : eventInternal ts_id
    """
    df_merged_1 = pd.merge(
        tracksters_toDf(tracksters),
        assocs_toDf(assoc, score_threshold=score_threshold),
        left_index=True, right_index=True
    )
    return pd.merge(
        df_merged_1,
        tracksters_toDf(simTracksters),
        left_on=["eventInternal", "simts_id"], right_index=True,
        how="left",
        suffixes=(None, "_sim")
    )

def superclusterAssociatedToSimTracksterCP(supercls:ak.Array, associations:ak.Array) -> ak.Array:
    """ Returns the supercluster ids corresponding to the CaloParticle for each event (from the simTracksterCP collection) 
    Warning : Removes all CPs that are associated to a trackster not in a supercluster !
    Parameters : 
     - supercls : type: nevts * var * var * uint64 
     - associations : type nevts * {
        tsCLUE3D_simToReco_CP: var * var * uint32,
        tsCLUE3D_simToReco_CP_score: var * var * float32,
        tsCLUE3D_simToReco_CP_sharedE: var * var * float32
        ***
    }
    Returns type: nevts * 2 (nb of simTracksterCP) * int64 (supercls id)

    """
    assocs_simToReco_largestScore = assocs_largestScore(assocs_zip_simToReco(associations))

    df_mainAssoc = ak.to_dataframe(assocs_simToReco_largestScore.ts_id, levelname=lambda x : {0:"eventInternal", 1:"genparticle_id"}[x]).reset_index(level=1)
    df_supercls = ak.to_dataframe(supercls, levelname=lambda x : {0:"eventInternal", 1:"supercls_id", 2:"ts_in_supercls_id"}[x]).reset_index(level=[1, 2])

    merged_df = pd.merge(df_mainAssoc, df_supercls,
        left_on=["eventInternal", "values"],
        right_on=["eventInternal", "values"],
        how="left" # keep the CP row event if there is no matching supercluster
    )

    # transform back to awkward array
    return ak.unflatten(ak.from_numpy(merged_df.supercls_id.to_numpy()), ak.run_lengths(ak.from_numpy(merged_df.index.to_numpy())))


def superclustersToEnergy(supercls_ts_idxs:ak.Array, tracksters:ak.Array) -> ak.Array:
    """ Computes the total energy of a supercluster from an array of supercluster ids 
    Preserves the inner index (usually CP id)
    Parameters :
     - supercls_ts_idxs : type nevts * var * var * uint64
    Returns : type nevts * var * float (energy sum)
    """
    # FIrst flatten the inner dimension (CP id) before taking tracksters
    energies_flat = tracksters.raw_energy[ak.flatten(supercls_ts_idxs, axis=-1)]
    # Rebuild the inner index
    energies = ak.unflatten(energies_flat,  ak.flatten(ak.num(supercls_ts_idxs, axis=-1)), axis=-1)

    return ak.sum(energies, axis=-1)



class DumperReader:
    def __init__(self, filePath:str, directoryName:str="ticlDumper") -> None:
        self.file = uproot.open(filePath + ":" + directoryName)
    
    @property
    def nEvents(self):
        return self.file["tracksters"].num_entries

    @cached_property
    def tracksters(self) -> ak.Array:
        return self.file["tracksters"].arrays()
    
    @cached_property
    def tracksters_zipped(self) -> ak.Array:
        
        return ak.zip({key : self.tracksters[key] for key in self.tracksters.fields 
                        if key not in ["event", "NClusters", "NTracksters"]},
            depth_limit=2 # don't try to zip vertices
        )

    @cached_property
    def simTrackstersCP(self) -> ak.Array:
        return self.file["simtrackstersCP"].arrays(filter_name="regressed_energy")
    @cached_property
    def simTrackstersCP_df(self) -> ak.Array:
        return ak.to_dataframe(self.simTrackstersCP, levelname=lambda x : {0:"eventInternal", 1:"genparticle_id"}[x])
    
    @cached_property
    def superclusters(self) -> ak.Array:
        """ Gets the supercluster trackster ids
        type: nevts * var (superclsCount) * var (trackstersInSupercls) * uint64 (trackster id)
        """
        return self.file["superclustering/superclusteredTracksters"].array()
    
    @cached_property
    def superclusters_all(self) -> ak.Array:
        """ Same as superclusters but tracksters not in a supercluster are included in a one-trackster supercluster each
        """
        return self.file["superclustering/superclusteredTrackstersAll"].array()

    @cached_property
    def associations(self) -> ak.Array:
        return self.file["associations"].arrays(filter_name="tsCLUE3D_*")
    
    
    @cached_property
    def supercluster_df(self) -> pd.DataFrame:
        return ak.to_dataframe(self.superclusters, anonymous="ts_id",
            levelname=lambda x : {0:"eventInternal", 1:"supercls_id", 2:"ts_in_supercls_id"}[x])

    @cached_property
    def supercluster_all_df(self) -> pd.DataFrame:
        return ak.to_dataframe(self.superclusters_all, anonymous="ts_id",
            levelname=lambda x : {0:"eventInternal", 1:"supercls_id", 2:"ts_in_supercls_id"}[x])
    
    @cached_property
    def supercluster_properties_df(self) -> pd.DataFrame:
        """ Dataframe left joining superclusters with trackster properties """
        return supercluster_properties(self.supercluster_df, self.tracksters)
    @cached_property
    def supercluster_properties_all_df(self) -> pd.DataFrame:
        """ Dataframe left joining superclusters with trackster properties """
        return supercluster_properties(self.supercluster_all_df, self.tracksters)
    
    @cached_property
    def supercluster_merged_properties_all(self) -> pd.DataFrame:
        """ Dataframe holding supercluster properties (one row per supercluster) """
        return (self.supercluster_properties_all_df
                .groupby(level=["eventInternal", "supercls_id"])
                .agg(
                    raw_energy_supercls=pd.NamedAgg("raw_energy", "sum"),
                    regressed_energy_supercls=pd.NamedAgg("regressed_energy", "sum"),
        ))
    
    def getCPToSuperclusterProperties(self) -> pd.DataFrame:
        """ Returns the properties of superclusters associated to each CaloParticle of the event 
        Returns pd.DataFrame with
        Index : eventInternal, genparticle_id (0 or 1),
        Columns :  supercls_id seed_ts_id (seed trackster id), raw_energy_supercls	regressed_energy_supercls"""
        # Get largest association score
        assocs_simToReco_largestScore = assocs_largestScore(assocs_zip_simToReco(self.associations))
        # Make a df out of it : index eventInternal	ts_id, column : genparticle_id
        df_mainAssoc:pd.DataFrame = (ak.to_dataframe(assocs_simToReco_largestScore[["ts_id"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"genparticle_id"}[x])
            .reset_index("genparticle_id")
            .set_index("ts_id", append=True)
        )

        # merge with supercluster df to get supercls_id
        # INDEX eventInternal	supercls_id, COLS genparticle_id	ts_id
        merged_tsId = (pd.merge(df_mainAssoc, self.supercluster_all_df,
                how="left", 
                left_index=True, right_on=["eventInternal", "ts_id"]
            )
            .reset_index("ts_in_supercls_id", drop=True)
        )
        # merge supercls_id to merged properties of superclusters
        merged_supercls = pd.merge(
            merged_tsId, self.supercluster_merged_properties_all, 
            left_index=True, right_index=True
        ).rename(columns={"ts_id" : "seed_ts_id"}).reset_index("supercls_id").set_index("genparticle_id", append=True)
    
        # merge with CaloParticle information
        merged_CP = pd.merge(merged_supercls, self.simTrackstersCP_df,
                        how="left", right_index=True, left_index=True)
        return merged_CP.rename(columns={str(col) : str(col)+"_CP" for col in self.simTrackstersCP_df.columns})
