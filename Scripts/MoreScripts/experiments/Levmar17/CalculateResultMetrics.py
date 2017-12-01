import numpy as np
import os

import PythonModel3dTracker.Paths as Paths
from PythonModelTracker.ModelTrackingResults import ModelTrackingResults
import PythonModel3dTracker.PythonModelTracker.LandmarksGrabber as LG
import PythonModel3dTracker.PythonModelTracker.DatasetInfo as DI


input_dir = os.path.join(Paths.results, "Human_tracking/Levmar/")


for i,f in enumerate(os.listdir(input_dir)):
    results_in = os.path.join(input_dir, f)
    f_base, f_ext = os.path.splitext(f)
    if (f_ext == '.json') and os.path.isfile(results_in):

        if i<3:
            print i, results_in
            res = ModelTrackingResults()
            res.load(results_in)
            model_name = res.models[0]
            di = DI.DatasetInfo()
            di.load(Paths.datasets_dict[str(res.did)])

            lg = LG.LandmarksGrabber(di.landmarks['gt']['format'],
                                     di.landmarks['gt']['filename'],
                                     di.landmarks['gt']['calib_filename'])

            lnames, landmarks = res.get_model_landmarks(model_name)


            for frame in landmarks:
                lg.seek(frame)
                gt_names, gt_landmarks, gt_clb = lg.acquire()
                cor_lnames = LG.LandmarksGrabber.getPrimitiveNamesfromLandmarkNames(gt_names, 'bvh', model_name)
                gt_indices = [i for i,g in enumerate(cor_lnames) if g != 'None']
                cor_indices = [lnames.index(g) for g in cor_lnames if g != 'None']
                # for g, c in zip(gt_indices, cor_indices):
                #     print g, gt_names[g],c, lnames[c]

                lnp_all = np.array(landmarks[frame])
                lnp = np.array([landmarks[frame][l] for l in cor_indices ])
                gnp = np.array([gt_landmarks[l].data[:,0] for l in gt_indices ])

                dists = np.linalg.norm(lnp-gnp,axis=1)
                avg_dist = np.average(dists)
                print di.did, frame, avg_dist
