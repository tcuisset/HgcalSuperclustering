from analyzer.driver.computations import DataframeComputation
from analyzer.driver.fileTools import SingleInputReader
from analyzer.dumperReader.reader import DumperReader, getCPToSuperclusterProperties

# cannot use a lambda as multirprocessing does not work due to pickle issues
def _CPToSuperclusterProperties_fct(reader:SingleInputReader):
    reader = reader.ticlDumperReader
    return getCPToSuperclusterProperties(reader.supercluster_all_df, reader.assocs_bestScore_simToReco_df, 
        reader.tracksters_zipped[["ts_id", "raw_energy", "raw_em_energy", "regressed_energy", "raw_pt", "barycenter_eta"]], reader.simTrackstersCP_df)
CPToSuperclusterProperties = DataframeComputation(_CPToSuperclusterProperties_fct, "CPToSupercluster")
