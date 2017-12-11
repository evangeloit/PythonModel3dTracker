import os.path

import PythonModel3dTracker.PythonModelTracker.ModelTrackingResults as mtr
from PythonModel3dTracker.PythonModelTracker.ResultLandmarksGenerator import GenerateLandmarks
import PythonModel3dTracker.Paths as Paths



dry_run = True
input_dir = os.path.join(Paths.results, "Human_tracking/Levmar/")


for i,f in enumerate(os.listdir(input_dir)):
    results_in = os.path.join(input_dir, f)
    f_base, f_ext = os.path.splitext(f)
    if (f_ext == '.json') and os.path.isfile(results_in):
        results_out = results_in #os.path.join(input_dir, f_base + '_ldm.json')
        print i,results_in, results_out
        if dry_run == False:
            results = mtr.ModelTrackingResults()
            results.load(results_in)
            results_ldm = GenerateLandmarks(results)
            results_ldm.save(results_out)





