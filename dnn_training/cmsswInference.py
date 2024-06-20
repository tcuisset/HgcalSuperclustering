""" Runs inference in CMSSW using trained DNN """
import subprocess
from pathlib import Path
import shutil
import uuid
import uproot
import argparse
import numpy as np
from analyzer.dumperReader.dqmReader import DQMReader

cmssw_inference_WPs = [0.05] + list(np.linspace(0.1, 0.9, 8, endpoint=False)) + list(np.linspace(0.9, 1., 11))
""" WPs used in cmssw config for rereco scanning WPs """

def runInferenceCMSSW(cmssw_release_path:str, cmssw_py_config_path:str, cmssw_py_config_path_harvesting:str, input_files:list[str], input_DQM_forComparison:str, output_path:str, model_path:str, dnn_wp:float|None, nThreads=20):
    """ Takes a DNN model and WP, and run inference using it with CMSSW, then DQM, then compare DQM with reference (EGammaV)
    Parameters : 
     - cmssw_release_path : path to CMSSW release to use
     - cmssw_py_config_path : path to python CMSSW config file to use for step3 (will be copied and input adjusted)
     - cmssw_py_config_path_harvesting : path to harvesting python config file (step4)
     - input_files : inputs for step3 (usually list of FEVT step3 for rereco, but could be step2 FEVT in case of full reco)
     - input_DQM_forComparison : DQM file (harvesting output) for comparison, usually Mustache
     - output_path : working directory for output and logs, etc
     - model_path : path to DNN model in ONNX format
     - dnn_wp : working point to use for DNN. If float, will use that WP. If None, will assume that the config runs multiple WPs in parallel, and it will replace "__DNN_MODEL_PATH__" with the path to the model
     - nThreads : number of threads for cmsRun
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_path)
    # using CMSSW_SEARCH_PATH leads to strange issues
    cmssw_release_path = Path(cmssw_release_path)
    model_unique_id = uuid.uuid4()
    # we need to copy the model because cmssw does not want symlinked files in FileInPath (directories symlinked are fine however)
    (cmssw_release_path / f"external/slc7_amd64_gcc12/data/RecoHGCal/TICL/data/models_hyperparam_search/").mkdir(parents=True, exist_ok=True)
    shutil.copy(model_path, cmssw_release_path / f"external/slc7_amd64_gcc12/data/RecoHGCal/TICL/data/models_hyperparam_search/{model_unique_id}.onnx")
    (output_path / f"{model_unique_id}.onnx").symlink_to(cmssw_release_path / f"external/slc7_amd64_gcc12/data/RecoHGCal/TICL/data/models_hyperparam_search/{model_unique_id}.onnx")

    with open(cmssw_py_config_path, "r") as f:
        cmssw_py_config = f.read()
    cmssw_py_config += "\n"
    cmssw_py_config += "process.source.fileNames = cms.untracked.vstring(" + ", ".join([f"'file:{p}'" for p in input_files]) + ")\n"
    if dnn_wp is None:
        # case of multiple WPs already in config file
        cmssw_py_config = cmssw_py_config.replace("__DNN_MODEL_PATH__", f'"RecoHGCal/TICL/data/models_hyperparam_search/{model_unique_id}.onnx"')
    else:
        cmssw_py_config += f"process.ticlTracksterLinksSuperclustering.linkingPSet.onnxModelPath = cms.FileInPath('RecoHGCal/TICL/data/models_hyperparam_search/{model_unique_id}.onnx')\n"
        cmssw_py_config += f"process.ticlTracksterLinksSuperclustering.linkingPSet.nnWorkingPoint = cms.double({dnn_wp})\n"
    with open(output_path / "step3_modelInference.py", "w") as f:
        f.write(cmssw_py_config)
    


    cmsenv_cmds = f'cd "{cmssw_release_path}" && cmsenv && cd "{output_path}" && '
    shell_cmds = cmsenv_cmds + \
        f'CUDA_VISIBLE_DEVICES="" cmsRun -n {nThreads} step3_modelInference.py > cmsRun.out 2>cmsRun.err'
    
    harvesting_shell_cmds = cmsenv_cmds + f'CUDA_VISIBLE_DEVICES="" cmsRun "{cmssw_py_config_path_harvesting}"'

    if dnn_wp is not None:
        # case of multiple WPs already in config file
        compare_dqm_cmds = cmsenv_cmds + \
            f'compare_using_files.py "{input_DQM_forComparison}" DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompareEgammaV -CR --standalone --use_black_file -p -d EgammaV'
    else:
        compare_dqm_cmds = "echo 'No DQM comparison in multi-WPs mode'"
    with open(output_path / "command.sh", "w") as f:
        f.write(shell_cmds + "\n" + harvesting_shell_cmds + "\n" + compare_dqm_cmds)
    subprocess.run(shell_cmds, capture_output=False, shell=True, text=True, check=True)

    # harvesting DQM
    subprocess.run(harvesting_shell_cmds, capture_output=False, shell=True, text=True, check=True)

    # compare DQM
    subprocess.run(compare_dqm_cmds, capture_output=False, shell=True, text=True, check=True)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='superclusteringDNNInference',
                    description='Superclustering DNN for HGCAL : inference with CMSSW + DQM')
    parser.add_argument("--cmssw", help="CMSSW release path", dest="cmssw_release_path")
    parser.add_argument("--cmsswConfig", help="Step3 config file for inference", dest="cmssw_py_config_path")
    parser.add_argument("--cmsswConfigHarvesting", help="Step4 config file for harvesting inference results", dest="cmssw_py_config_path_harvesting")
    parser.add_argument("--cmsswInputFiles", help="Input files for CMSSW inference", dest="cmmsw_input_files", nargs="+")
    parser.add_argument("--DQMForComparison", help="DQM file for comparison", dest="input_DQM_forComparison")
    parser.add_argument("--output", help="Ouput directiory")
    parser.add_argument("--model", help="Path to model")
    parser.add_argument("--wp", help="DNN Working point", default=None)

    args = parser.parse_args()

    runInferenceCMSSW(
        cmssw_release_path=args.cmssw_release_path,
        cmssw_py_config_path=args.cmssw_py_config_path,
        cmssw_py_config_path_harvesting=args.cmssw_py_config_path_harvesting,
        output_path=args.output,
        model_path=args.model,
        dnn_wp=args.wp,
        input_files=args.cmmsw_input_files,
        input_DQM_forComparison=args.input_DQM_forComparison
    )