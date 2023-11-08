from typing import Iterable, Callable

import pandas as pd

from analyzer.tools import DumperReader


class Computation:
    def workOnSample(self, reader:DumperReader):
        """ Do work on one sample, returning a result (must be pickleable) """
        pass

    def reduce(self, results:Iterable, store:pd.HDFStore, nbOfEvents:Iterable[int]):
        """ Reduce the results given (objects returned by workOnSample) and then store them
        Parameters : 
         - nbOfEvents : list of same length as results, holding the number of event in each batch
        """
        pass


class DataframeComputation(Computation):
    def __init__(self, workFct:Callable[[DumperReader], pd.DataFrame], key:str, hdfSettings:dict=dict()) -> None:
        self.workFct = workFct
        self.key = key
        self.hdfSettings = hdfSettings
    
    def workOnSample(self, reader: DumperReader) -> pd.DataFrame:
        return self.workFct(reader)
    
    def reduce(self, results: Iterable[pd.DataFrame], store:pd.HDFStore, nbOfEvents:Iterable[int]):
        # Update the event index so it is unique across batches
        runningEventCount = 0
        for result, cuurentNbOfEvts in zip(results, nbOfEvents):
            if runningEventCount > 0:
                result.index = result.index.set_levels(result.index.levels[0]+runningEventCount, level=0)
            runningEventCount += cuurentNbOfEvts
        
        pd.concat(results).to_hdf(store, self.key, **self.hdfSettings)