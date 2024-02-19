import ROOT
import cppyy
from DataFormats.FWLite import Events, Handle
import edmReader.pythonizations

class EdmReader(Events):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._handles:dict[tuple, Handle] = {}
        self._handles_valid:dict[tuple, bool] = {}
    
    def get(self, *args):
        """ Calls getByLabel, caching the handle, returns the product directly 
        get (typeString, moduleLabel)
        get (typeString, moduleLabel, productInstanceLabel),
        get (typeString, moduleLabel, productInstanceLabel, processLabel),"""
        length = len (args)
        if length < 2 or length > 4:
            # not called correctly
            raise RuntimeError("Incorrect number of arguments")
        if tuple(args) not in self._handles:
            self._handles[tuple(args)] = Handle(args[0])
        handle = self._handles[tuple(args)]
        self.getByLabel(*args[1:], handle)
        return handle.product()

    @property
    def genParticles(self):
        return self.get("vector<reco::GenParticle>", "genParticles")
    @property
    def electronSeeds(self):
        return self.get("reco::ElectronSeedCollection", "ecalDrivenElectronSeeds")
    @property
    def ticlEgammaSuperClusters(self):
        return self.get("reco::SuperClusterCollection", "ticlEGammaSuperClusterProducer")
    @property
    def ticlEgammaSuperClusters_caloClusters(self):
        return self.get("reco::CaloClusterCollection", "ticlEGammaSuperClusterProducer")
    @property
    def ticlSuperclusters(self):
        return self.get("ticl::TracksterCollection", "ticlTracksterLinksSuperclustering")
    @property
    def ticlSuperclusterLinks(self):
        return self.get("vector<vector<unsigned int> >", "ticlTracksterLinksSuperclustering")
    @property
    def trackstersCLUE3DEM(self):
        return self.get("ticl::TracksterCollection", "ticlTrackstersCLUE3DEM")
    
    @property
    def hgcalPfRechits(self):
        return self.get("reco::PFRecHitCollection", "particleFlowRecHitHGC")
    @property
    def hgcalPfClusters(self):
        return self.get("reco::PFClusterCollection", "particleFlowClusterHGCal")
    @property
    def oldHgcalSuperclusters(self):
        return self.get("reco::SuperClusterCollection", "particleFlowSuperClusterHGCal")
    @property
    def oldHgcalSuperclusters_caloClusters(self):
        return self.get("reco::CaloClusterCollection", "particleFlowSuperClusterHGCal")

    @property
    def pfBlocks(self):
        return self.get("vector<reco::PFBlock>", "particleFlowBlock")
    def printPfBlocks(self, filterOutNonEndcap=False):
        for pfBlock in self.pfBlocks:
            if filterOutNonEndcap: # filter out barrel and forward calorimeter
                if all((pfElement.clusterRef().layer() in [cppyy.gbl.PFLayer.ECAL_BARREL, cppyy.gbl.PFLayer.HCAL_BARREL1, cppyy.gbl.PFLayer.HCAL_BARREL2, cppyy.gbl.PFLayer.HF_EM, cppyy.gbl.PFLayer.HF_HAD] for pfElement in pfBlock.elements())):
                    continue
            print(pfBlock)
            