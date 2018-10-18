import os.path

import PythonModel3dTracker.PythonModelTracker.TrackingResults.ModelTrackingResults as mtr
from PythonModel3dTracker.PythonModelTracker.TrackingResults.ResultLandmarksGenerator import GenerateLandmarks
import PythonModel3dTracker.Paths as Paths
import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.PythonModelTracker.PFHelpers.TrackingTools as tt
import PythonModel3dTracker.PythonModelTracker.Landmarks.Model3dLandmarks as M3DL

dry_run = False
input_dir = os.path.join(Paths.results, "Human_tracking")
# input_dir = "/home/evangeloit/Desktop/Gest2/LocalTests/results/human_tracking"


for i,f in enumerate(os.listdir(input_dir)):
    results_in = os.path.join(input_dir, f)
    f_base, f_ext = os.path.splitext(f)
    results = mtr.ModelTrackingResults()
    if results.load(results_in) and (len(results.landmark_names) == 0):
        results_out = os.path.join(input_dir, f_base + '_ldm.json')
        print i,results_in, results_out
        if dry_run == False:
            # results.load(results_in)
            results.landmark_names = {}
            results.landmarks = {}
            # model_name = results.models[0]
            model_name = "mh_body_male_customquat_950"
            model3d, _ = tt.ModelTools.GenModel(model_name)
            landmark_names_mbv, landmarks = \
                M3DL.GenerateModelLandmarksfromObservationLandmarks(model3d,'bvh', ldm_obs_names=None)
            results_ldm = GenerateLandmarks(results,landmark_names_mbv, landmarks)
            results_ldm.save(results_out)
    else:
        print i, '--- skipping ---', f_base





