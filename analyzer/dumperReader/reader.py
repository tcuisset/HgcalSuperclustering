from functools import cached_property
from typing import Literal

import awkward as ak
import pandas as pd
import uproot

from .assocs import *
from .tracksters import *
from .recoToSim import *
from .simToReco import *



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
    class MultiFileReader:
        def __init__(self, files:list[uproot.ReadOnlyDirectory]):
            self.files = files
        
        def __getitem__(self, key):
            excpts = []
            for file in self.files:
                try:
                    return file[key]
                except uproot.KeyInFileError as e:
                    excpts.append(e)
            raise KeyError(*excpts)
            
    def __init__(self, file:str|uproot.ReadOnlyDirectory|list[uproot.ReadOnlyDirectory], directoryName:str="ticlDumper") -> None:
        try:
            self.fileDir = file[directoryName]
        except TypeError:
            try:
                self.fileDir = self.MultiFileReader([f[directoryName] for f in file])
            except TypeError:
                self.fileDir = uproot.open(file + ":" + directoryName)
    
    @property
    def nEvents(self):
        return self.fileDir["tracksters"].num_entries

    @cached_property
    def tracksters(self) -> ak.Array:
        return self.fileDir["tracksters"].arrays()
    
    @cached_property
    def tracksters_zipped(self) -> ak.Array:
        return ak.zip({"ts_id" : ak.local_index(self.tracksters.raw_energy, axis=1)} | 
                      {key : self.tracksters[key] for key in self.tracksters.fields 
                        if key not in ["event", "NClusters", "NTracksters"]},
            depth_limit=2, # don't try to zip vertices
            with_name="tracksters"
        )

    @cached_property
    def simTrackstersCP(self) -> ak.Array:
        return self.fileDir["simtrackstersCP"].arrays(filter_name=["raw_energy", "raw_energy_em", "regressed_energy", "barycenter_*"])
    @cached_property
    def simTrackstersCP_df(self) -> pd.DataFrame:
        return ak.to_dataframe(self.simTrackstersCP, levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id"}[x])
    
    @cached_property
    def superclusters(self) -> ak.Array:
        """ Gets the supercluster trackster ids
        type: nevts * var (superclsCount) * var (trackstersInSupercls) * uint64 (trackster id)
        """
        return self.fileDir["superclustering/superclusteredTracksters"].array()
    
    @cached_property
    def superclusters_all(self) -> ak.Array:
        """ Same as superclusters but tracksters not in a supercluster are included in a one-trackster supercluster each
        """
        return self.fileDir["superclustering/superclusteredTrackstersAll"].array()

    @cached_property
    def superclusteringDnnScore(self) -> ak.Array:
        return self.fileDir["superclustering/superclusteringDNNScore"].array()

    @cached_property
    def associations(self) -> ak.Array:
        return self.fileDir["associations"].arrays(filter_name="tsCLUE3D_*")
    
    
    @cached_property
    def supercluster_df(self) -> pd.DataFrame:
        return ak.to_dataframe(self.superclusters, anonymous="ts_id",
            levelname=lambda x : {0:"eventInternal", 1:"supercls_id", 2:"ts_in_supercls_id"}[x])

    @cached_property
    def supercluster_all_df(self) -> pd.DataFrame:
        """ Same as superclusters_df but tracksters not in a supercluster are included in a one-trackster supercluster each
        """
        return ak.to_dataframe(self.superclusters_all, anonymous="ts_id",
            levelname=lambda x : {0:"eventInternal", 1:"supercls_id", 2:"ts_in_supercls_id"}[x])
    
    @cached_property
    def supercluster_merged_properties_all(self) -> pd.DataFrame:
        """ Dataframe holding supercluster properties (one row per supercluster) 
        
        Tracksters not in a supercluster are included as one-trackster superclusters
        """
        return (supercluster_joinTracksters(self.supercluster_all_df, self.tracksters_zipped[["raw_energy", "regressed_energy"]])
                .groupby(level=["eventInternal", "supercls_id"])
                .agg(
                    raw_energy_supercls=pd.NamedAgg("raw_energy", "sum"),
                    regressed_energy_supercls=pd.NamedAgg("regressed_energy", "sum"),
        ))
    
    @cached_property
    def assocs_bestScore_simToReco_df(self) -> pd.DataFrame:
        """ Make a Df of largest score associations of each SimTrackster 
        
        Index eventInternal	ts_id, column : caloparticle_id
        """
        # Get largest association score
        assocs_simToReco_largestScore = assocs_bestScore(assocs_zip_simToReco(self.associations))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return  (ak.to_dataframe(assocs_simToReco_largestScore[["ts_id", "caloparticle_id", "score", "sharedE"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id_wrong"}[x])
                .reset_index("caloparticle_id_wrong", drop=True)
                .set_index("caloparticle_id", append=True)
        )
    
    def assocs_bestScore_recoToSim_df(self, dropOnes=True) -> pd.DataFrame:
        """ Make a Df of largest score associations of each Trackster 
        
        Parameters : 
         - dropOnes : if True, do not include associations of score 1 (worst score)
        Index eventInternal	caloparticle_id, column : ts_id, score, sharedE
        """
        # Get largest association score
        assocs = assocs_bestScore((assocs_dropOnes if dropOnes else lambda x:x)(assocs_zip_recoToSim(self.associations)))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return (ak.to_dataframe(assocs[["ts_id", "caloparticle_id", "score", "sharedE"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"ts_id_wrong"}[x])
                .reset_index("ts_id_wrong", drop=True)
                .set_index("ts_id", append=True)
        )
        
    



