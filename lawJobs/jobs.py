import os

import law
law.contrib.load("cms", "git", "htcondor")

class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):
    pass

class BaseTask(law.Task):
    store = "/grid_mnt/data_cms_upgrade/cuisset/supercls/law"

    def local_path(self, path):
        return os.path.join(self.store, self.task_name, path)

class CmsRunTask():
    pass

class Gen(BaseTask, law.LocalWorkflow, HTCondorWorkflow):
    task_name = "gen"

    def output(self):
        return [law.LocalFileTarget(self.local_path(f"step2_{self.branch}.root"))]

    def run(self):
        pass
    
