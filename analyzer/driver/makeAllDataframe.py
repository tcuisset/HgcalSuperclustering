import os
import argparse
import glob
import concurrent.futures
import collections.abc
import importlib

import uproot
import tqdm
import pandas as pd
# Enable Pandas Copy on Write. Leads to about 30% gain in run time and less memory usage
pd.options.mode.copy_on_write = True


from analyzer.tools import DumperReader
from analyzer.driver.computations import Computation


parser = argparse.ArgumentParser(description="Reads TICLDumper output and makes histograms/reduced dataframes.")
parser.add_argument("--input-dir", "-i", dest='input_dir', action="append",
    help="The path to the directory where all the TICLDumper output data is. Can be a directory (takes all .root files inside), or a file. Can be mentionned multiple times.",
)
# parser.add_argument("--output-tag", "-t", dest="tag",
#     help="The tag (version) to use for the output") 
parser.add_argument("--output-dir", "-o", dest="output_dir",
    help="The output folder"
)
# parser.add_argument("--force-output-directory", dest='force_output_directory',
#     default=None,
#     help="Complete path to folder where to store histograms  (for testing)")
parser.add_argument("--max-workers", dest="max_workers", default=10, type=int,
    help="Number of workers to use")
#parser.add_argument("--save-metadata", dest="save_metadata", action=argparse.BooleanOptionalAction,
#    default=False, help="Save metadata of histograms (name, label, axes) to a pickle file")
parser.add_argument("--computation-module", dest="computation_module", 
    default="analyzer.computations.all_computations",
    help="module to load to find computations")
parser.add_argument("--computation-variable", dest="computation_vars", action="append",
    #default=["all_computations"]
)
args = parser.parse_args()

# list default values need to handled after parsing (otherwise command line arguments add to default value)
if args.computation_vars is None:
    # in case of custom --computation-variable remove default value
    args.computation_vars = ["all_computations"]

#if args.input_dir is None:
#    args.input_dir=['/data_cms_upgrade/cuisset/supercls/alessandro_electrons/input-oppositeSign', '/data_cms_upgrade/cuisset/supercls/alessandro_electrons/input-sameSign']

# Listing input
pathToInputs = []
for inputParam in args.input_dir:
    if os.path.isfile(inputParam):
        pathToInputs.append(inputParam)
    elif os.path.isdir(inputParam):
        pathToInputs.extend(glob.glob(os.path.join(inputParam, "*.root")))
    else:
        raise ValueError("File path " + inputParam + " is neither a file nor a directory")

# filling computations from the variables of the module given in parameters
computations:list[Computation] = []

compModule = importlib.import_module(args.computation_module)
for comp_variable_str in args.computation_vars:
    comp_variable = getattr(compModule, comp_variable_str)
    if isinstance(comp_variable, Computation):
        computations.append(comp_variable)
    else:
        computations.extend(comp_variable)


with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.max_workers, len(pathToInputs))) as executor:
    def map_fcn(pathToInput:str) -> list:
        reader = DumperReader(pathToInput)
        return [comp.workOnSample(reader) for comp in computations], reader.nEvents

    map_res = executor.map(map_fcn, pathToInputs)

# map_res is [([comp1Res, comp2Res, ...], nEvts), ...]
resultsPerFile, nbOfEvents = zip(*map_res)
# listOfResults is ([comp1Res, comp2Res, ...], ...)
# Reduce
with pd.HDFStore(os.path.join(args.output_dir, "store.hdf")) as store:
    for comp, resultsPerComputation in zip(computations, zip(*resultsPerFile)):
        # resultsPerComputation is a tuple of ResultToReduce that are to be reduced together
        comp.reduce(resultsPerComputation, store, nbOfEvents=nbOfEvents)

