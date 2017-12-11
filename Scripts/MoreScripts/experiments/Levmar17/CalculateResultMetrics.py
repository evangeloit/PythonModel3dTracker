import numpy as np
import os

import PythonModel3dTracker.Paths as Paths
from PythonModel3dTracker.PythonModelTracker.ModelTrackingResults import ModelTrackingResults
import PythonModel3dTracker.PythonModelTracker.LandmarksGrabber as LG
import PythonModel3dTracker.PythonModelTracker.DatasetInfo as DI


input_dir = os.path.join(Paths.results, "Human_tracking/Levmar/")


for i,f in enumerate(os.listdir(input_dir)):
    results_in = os.path.join(input_dir, f)
    f_base, f_ext = os.path.splitext(f)
    if (f_ext == '.json') and os.path.isfile(results_in):

        #print i, results_in
        res = ModelTrackingResults()
        res.load(results_in)
        if res.did == 'mhad_s10_a04':
            model_name = str(res.models[0])
            di = DI.DatasetInfo()
            di.load(Paths.datasets_dict[str(res.did)])

            lg = LG.LandmarksGrabber(di.landmarks['gt']['format'],
                                     di.landmarks['gt']['filename'],
                                     di.landmarks['gt']['calib_filename'])

            lnames, landmarks = res.get_model_landmarks(model_name)

            seq_dists = []
            for frame in landmarks:
                lg.seek(frame)
                gt_names, gt_landmarks, gt_clb = lg.acquire()
                l_cor, g_cor =  LG.GetCorrespondingLandmarks(model_name, lnames, landmarks[frame],
                                                             di.landmarks['gt']['format'], gt_names,gt_landmarks)
                lnp = np.array(l_cor)
                gnp = np.array(g_cor)
                dists = np.linalg.norm(lnp-gnp,axis=1)
                avg_dist = np.average(dists)
                seq_dists.append(avg_dist)
            seq_dist = np.average(np.array(seq_dists))
            print results_in, seq_dist
