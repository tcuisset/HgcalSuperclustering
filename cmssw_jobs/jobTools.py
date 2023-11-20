import glob
import subprocess
import os
from pathlib import Path
import collections
import re

import htcondor

pathToBashStarter = (Path(__file__).parent / "cmsenvStarter.sh").resolve()
pathToCmsRunStarter = (Path(__file__).parent / "cmsRunConfigStarter.py").resolve()
pathToCmssw = (Path(__file__).resolve().parent.parent / "CMSSW_13_2_5_patch2").resolve()
pathTest = Path(__file__)


def makeSubmitObject(cmsRunArguments:str, logDir:str, cmssw_path:str=str(pathToCmssw), t3queue:str="long"):
    return htcondor.Submit({
        "executable" : str(pathToBashStarter),
        "arguments" : f"cmsRun {cmsRunArguments}", # OMP_NUM_THREADS is set by HTCondor to the nb of CPU cores allocated on the node
        "environment" : f"\"CMSSW_BASE='{Path(cmssw_path).resolve()}'\"",
        "universe" : "vanilla",
        "output" : f"{logDir}/$(SampleId).out",
        "error" : f"{logDir}/$(SampleId).err",
        "log" : f"{logDir}/$(SampleId).log",

        "request_cpus" : "2",
        #"request_memory" : 
        
        "T3Queue" : t3queue,
        "WNTag" : "el7",
        #"+SingularityCmd" : "",

        "UNIX_GROUP" : subprocess.check_output(['id', '-g', '--name']).strip().decode("utf8"), # strip removes the ending \n
        "accounting_group" : "$(UNIX_GROUP)",
        "concurrency_limits_expr" : 'strcat(T3Queue,":",RequestCpus," ",AcctGroupUser,":",RequestCpus)',
        "+T3Queue" : htcondor.classad.quote("$(T3Queue)"),
        "+T3Group" : htcondor.classad.quote("$(UNIX_GROUP)"),
        "+WNTag" : htcondor.classad.quote("$(WNTag)"),
        "+T3Submit" : "true",
    })

def makeSubmitStringTemplate(cmsRunArguments:str, logDir:str, cmssw_path:str=str(pathToCmssw), t3queue:str="long") -> str:
    res = "executable = " + str(pathToBashStarter) + "\n"
    res += "arguments = \"cmsRun " + cmsRunArguments + '"\n'
    res += f'environment = "CMSSW_BASE=\'{Path(cmssw_path).resolve()}\'"\n'
    res += "universe = vanilla\n"
    res += "output = $(LogDir)/$(SampleId).out\nerror = $(LogDir)/$(SampleId).err\nlog = $(LogDir)/$(SampleId).log\n"
    res += f"T3Queue = {t3queue}\n"
    res += "include : /opt/exp_soft/cms/t3/t3queue |"
    return res

# def writeCmsConfigFile(pathToBaseConfig:str, inputFile:str, outputFileService:str|None=None, outputEventDump:str|None=None, outputDQM:str|None=None):
#     pathToBaseConfig = Path(pathToBaseConfig)
#     script_identifier = pathToBaseConfig.with_suffix("").name
#     assert script_identifier.isidentifier()
#     config = [
#         "import sys",
#         "import os",
#         "import FWCore.ParameterSet.Config as cms",
#         f"sys.path.append('{pathToBaseConfig.resolve().parent}')",
#         f"from {script_identifier} import process",
#         "",
#         f"process.source.fileNames = cms.untracked.vstring('file:{inputFile}')",
#     ]
#     if outputFileService is not None:
#         config.append(f"process.TFileService.fileName = cms.untracked.string('{outputFileService}')")
#     if outputEventDump is not None:
#         config.append(f"process.FEVTDEBUGHLToutput.fileName = cms.untracked.string('{outputEventDump}')")
#     if outputDQM is not None:
#         config.append(f"process.DQMoutput.fileName = cms.untracked.string('{outputDQM}')")
    
#     return "\n".join(config)

def makeArgumentsForCmsRunConfigStarter(pathToConfig:os.PathLike, inputFile:os.PathLike, outputDir:os.PathLike|None=None, *, pathToStarter=pathToCmsRunStarter, extraArgs:str|list[str]=[], maxEvents:int=None):
    """ Makes an argument list for cmsenvStarter.sh launching cmsRunConfigStarter.py
    Parameters : 
     - pathToStarter : path to cmsRunConfigStarter.py
     - pathToConfig : path to cmsRun config file
     - inputFile : path to ROOT input file for cmsRun job
     - outputDir : path to put output files in
     - extraArgs : any argument to append to the command. They are not quoted by this function.
    """
    args = f"'{str(Path(pathToStarter).resolve())}' -- -s {Path(pathToConfig).resolve()} -i {Path(inputFile).resolve()}"
    if outputDir is not None:
        fevtOutput = Path(outputDir) / "step3_$(SampleId).root"
        dqmOutput = Path(outputDir) / "step3_inDQM_$(SampleId).root"
        fileServiceOutput = Path(outputDir) / "dumper_$(SampleId).root"
        args += f" '--output-fevt={fevtOutput.resolve()}' '--output-dqm={dqmOutput.resolve()}' '--output-fileService={fileServiceOutput.resolve()}'"
    
    if maxEvents is not None:
        args += f" 'process.maxEvents.input=cms.untracked.int32({maxEvents})'"
    if isinstance(extraArgs, str):
        extraArgs = [extraArgs]
    for extrArg in extraArgs:
        args += f" {extrArg}"
    return args


InputFile = collections.namedtuple("InputFile", ["path", "ntupleId", "sameSign"])

pattern_ntupleId = re.compile(r".{,}_([0-9]{1,})\.root")

def makeInputsInPath(inputsDir:Path, **kwargs_inputFile) -> list[InputFile]:
    inputsDir = Path(inputsDir)
    inputs = []
    for file in inputsDir.iterdir():
        if file.is_file():
            match_res = pattern_ntupleId.match(file.name)
            inputs.append(InputFile(file, match_res[1], **kwargs_inputFile))
    return inputs

inputFiles = makeInputsInPath("/data_cms_upgrade/cuisset/supercls/alessandro_electrons/input-oppositeSign/", sameSign=False)
inputFiles.extend(makeInputsInPath("/data_cms_upgrade/cuisset/supercls/alessandro_electrons/input-sameSign/", sameSign=True))
#inputFiles = [InputFile(path, ntupleId= sameSign=False) for path in glob.glob("/data_cms_upgrade/cuisset/supercls/alessandro_electrons/input-oppositeSign/step2_*.root")]
#inputFiles.extend(glob.glob("/data_cms_upgrade/cuisset/supercls/alessandro_electrons/input-sameSign/step2_*.root"))


