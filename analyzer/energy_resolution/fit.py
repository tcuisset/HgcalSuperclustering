from functools import partial
import dataclasses
from dataclasses import dataclass

from scipy.optimize import curve_fit
import numpy as np
import hist
import pandas as pd

@dataclass
class CruijffParam:
    A:float
    """ Amplitude"""
    m:float
    """ Central value """
    sigmaL:float
    """ Left tail sigma """
    sigmaR:float
    """ Right tail sigma """
    alphaL:float
    """ Left tail alpha """
    alphaR:float
    """ Right tail alpha """

    @property
    def sigmaAverage(self) -> float:
        return (self.sigmaL + self.sigmaR) / 2
    
    def makeTuple(self) -> tuple[float]:
        return dataclasses.astuple(self)

@dataclass
class CruijffFitResult:
    params:CruijffParam
    covMatrix:np.ndarray

def cruijff(x, A, m, sigmaL,sigmaR, alphaL, alphaR):
    dx = (x-m)
    SL = np.full(x.shape, sigmaL)
    SR = np.full(x.shape, sigmaR)
    AL = np.full(x.shape, alphaL)
    AR = np.full(x.shape, alphaR)
    sigma = np.where(dx<0, SL,SR)
    alpha = np.where(dx<0, AL,AR)
    f = 2*sigma*sigma + alpha*dx*dx
    return A* np.exp(-dx*dx/f)

def histogram_quantiles(h:hist.Hist, quantiles):
    """ Compute quantiles from histogram. Quantiles should be a float (or array of floats) representing the requested quantiles (in [0, 1])
    Returns array of quantile values
    """
    # adapated from https://stackoverflow.com/a/61343915
    assert len(h.axes) == 1 and h.storage_type == hist.storage.Double, "Histogram quantiles needs a 1D double (non-weighted) histogram"
    cdf = (np.cumsum(h.values()) - 0.5 * h.values()) / np.sum(h.values()) 
    return np.interp(quantiles, cdf, h.axes[0].centers)

def fitCruijff(h_forFit:hist.Hist) -> CruijffFitResult:
    mean = np.average(h_forFit.axes[0].centers, weights=h_forFit.values())
    q_min2, q_min1, median, q_plus1, q_plus2 = histogram_quantiles(h_forFit, [0.5-0.95/2, 0.5-0.68/2, 0.5, 0.5+0.68/2, 0.5+0.95/2])

    # Approximate sigmaL and sigmaR using quantiles. Using quantiles that are equivalent to 1 sigma left and right if the distribution is Gaussian
    # Compared to using standard deviation, it is asymmetric and less sensitive to tails

    p0 = [
        np.max(h_forFit)*0.8, # normalization. The 0.8 is because it seems that the max value is usually a bit higher
        mean, # central value
        median-q_min1, #sigmaL : this quantile difference is 1sigma for a Gaussian
        q_plus1-median, # sigmaR
        (q_min1-q_min2) / (median-q_min1)/3.81 * 0.28067382, #alphaL : in the ratio, numerator and denominator should be sigma for a gaussian. Otherwise, the heavier the tails, the higher from one. Then some norm coefficient (could be improved)
        (q_plus2-q_plus1) / (q_plus1-median)/3.81 * 0.28067382 #alphaR 
    ] 
    try:
        param_optimised,param_covariance_matrix = curve_fit(cruijff, h_forFit.axes[0].centers, h_forFit.values(), 
            p0=p0, sigma=np.maximum(np.sqrt(h_forFit.values()), 1.8), absolute_sigma=True, maxfev=500000,
            bounds=np.transpose([(0., np.inf), (-np.inf, np.inf), (0., np.inf), (0., np.inf), (-np.inf, np.inf), (-np.inf, np.inf)])
            )
    except ValueError: # sometimes it fails with ValueError: array must not contain infs or NaNs and removing bounds helps
        param_optimised,param_covariance_matrix = curve_fit(cruijff, h_forFit.axes[0].centers, h_forFit.values(), 
            p0=p0, sigma=np.maximum(np.sqrt(h_forFit.values()), 1.8), absolute_sigma=True, maxfev=500000,
            #bounds=np.transpose([(0., np.inf), (-np.inf, np.inf), (0., np.inf), (0., np.inf), (-np.inf, np.inf), (-np.inf, np.inf)])
            )
    return CruijffFitResult(CruijffParam(*param_optimised), param_covariance_matrix)


eratio_axis = partial(hist.axis.Regular, 500, 0, 2, name="e_ratio")
eta_axis = hist.axis.Variable([1.65, 2.15, 2.75], name="absSeedEta", label="|eta|seed")
seedPt_axis = hist.axis.Variable([ 0.44310403, 11.58994007, 23.00519753, 34.58568954, 46.85866928,
       58.3225441 , 68.96975708, 80.80027771, 97.74741364], name="seedPt", label="Seed Et (GeV)") # edges are computed so that there are the same number of events in each bin
def make_scOrTsOverCP_energy_histogram(name, label=None):
    h = hist.Hist(eratio_axis(label=label),
                  eta_axis, seedPt_axis, name=name, label=label)
    return h

def fill_scOverCP_energy_histogram(h:hist.Hist, df:pd.DataFrame):
    """ df should be CPtoSC_df ie CaloParticle to Supercluster """
    h.fill(e_ratio=df.raw_energy_supercls_sum/df.regressed_energy_CP,
        absSeedEta=np.abs(df.barycenter_eta_seed),
        seedPt=df.raw_pt_seed)

def fill_seedTsOverCP_energy_histogram(h:hist.Hist, df:pd.DataFrame):
    """ df should be CPtoTs_df ie CaloParticle to seed trackster (highest pt trackster for each endcap) """
    h.fill(e_ratio=df.raw_energy/df.regressed_energy_CP,
        absSeedEta=np.abs(df.barycenter_eta),
        seedPt=df.raw_pt)



def fitMultiHistogram(h:hist.Hist) -> list[list[CruijffFitResult]]:
    """ Cruijff fit of multi-dimensional histogram of Supercluster/CaloParticle energy """
    res = []
    for eta_bin in range(len(h.axes["absSeedEta"])):
        res.append([])
        for seedPt_bin in range(len(h.axes["seedPt"])):
            h_1d = h[{"absSeedEta":eta_bin, "seedPt":seedPt_bin}]
            res[-1].append(fitCruijff(h_1d))
    return res

def etaBinToText(etaBin:int) -> str:
    low, high = eta_axis[etaBin]
    return r"$|\eta_{\text{seed}}| \in \left[" + f"{low}; {high}" + r"\right]$"

def ptBinToText(ptBin:int) -> str:
    low, high = seedPt_axis[ptBin]
    return r"$E_{\text{T, seed}} \in \left[" + f"{low:.3g}; {high:.3g}" + r"\right]$"