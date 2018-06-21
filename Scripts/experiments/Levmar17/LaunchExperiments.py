import os
import PythonModel3dTracker.Paths as Paths

dry_run = 0
experiments_exec = 'LevmarExperiments.py'
experiments_path = os.path.join(Paths.package_path, 'Scripts/experiments/Levmar17/', experiments_exec)


for i in range(100):
    command_ = "python {0} {1} {2}".format(experiments_path, i, dry_run)
    #print "Calling:", command_
    os.system(command_)