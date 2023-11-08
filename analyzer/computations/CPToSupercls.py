from analyzer.driver.computations import DataframeComputation



CPToSuperclusterProperties = DataframeComputation(lambda reader : reader.getCPToSuperclusterProperties(), "CPToSupercluster")
