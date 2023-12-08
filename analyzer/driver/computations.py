from typing import Iterable, Callable
import functools

import pandas as pd
import hist

from analyzer.driver.fileTools import Computation, DumperInputManager




class DataframeComputation(Computation):
    def __init__(self, workFct:Callable[[DumperInputManager], pd.DataFrame], key:str|None=None, hdfSettings:dict=dict(), inputMode="ticlDumper") -> None:
        self.workFct = workFct
        self.key = key
        self.hdfSettings = hdfSettings
        self.inputMode = inputMode
    
    def workOnSample(self, *args, **kwargs) -> pd.DataFrame:
        return self.workFct(*args, **kwargs)
    
    def reduce(self, results: Iterable[pd.DataFrame], store:pd.HDFStore, nbOfEvents:Iterable[int]):
        # Update the event index so it is unique across batches
        runningEventCount = 0
        for result, cuurentNbOfEvts in zip(results, nbOfEvents):
            if runningEventCount > 0:
                result.index = result.index.set_levels(result.index.levels[0]+runningEventCount, level=0)
            runningEventCount += cuurentNbOfEvts
        
        df = pd.concat(results)
        if store is not None:
            assert (self.key is not None), "To store in HDF store, key should be set"
            df.to_hdf(store, self.key, **self.hdfSettings)
        return df


class HistogramComputation(Computation):
    def __init__(self, h:hist.Hist, fillFct:Callable[[hist.Hist, DumperInputManager], None], paramsForFillFct:dict=dict()) -> None:
        """ 
        Parameters : 
            - h : empty histogram for filling (must be pickleable) 
            - paramsForFillFct : dict of keyword arguments that will be passed to fillFct after dumperReader
        """
        self.h = h
        self.fillFct = fillFct
        self.paramsForFillFct = paramsForFillFct
    
    def workOnSample(self, reader: DumperInputManager) -> pd.DataFrame:
        self.fillFct(self.h, reader, **self.paramsForFillFct)
        return self.h
    
    def reduce(self, results: Iterable[hist.Hist], store:pd.HDFStore=None, nbOfEvents:Iterable[int]=None) -> hist.Hist:
        return sum(results)
