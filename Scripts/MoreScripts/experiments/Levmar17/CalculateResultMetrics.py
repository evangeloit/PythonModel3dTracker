import numpy as np
import os

import PythonModel3dTracker.Paths as Paths
from PythonModel3dTracker.PythonModelTracker.ModelTrackingResults import ModelTrackingResults
import PythonModel3dTracker.PythonModelTracker.LandmarksGrabber as LG
import PythonModel3dTracker.PythonModelTracker.DatasetInfo as DI


input_dir = os.path.join(Paths.results, "Human_tracking/Levmar/")


def CalculateMetricsDir(input_dir):
    for i,f in enumerate(sorted(os.listdir(input_dir))):
        results_in = os.path.join(input_dir, f)
        f_base, f_ext = os.path.splitext(f)
        if (f_ext == '.json') and os.path.isfile(results_in):
            CalculateMetricsJson(fname=results_in)



def CalculateMetricsJson(fname):
    #print i, results_in
    res = ModelTrackingResults()
    res.load(fname)

    model_name = str(res.models[0])
    di = DI.DatasetInfo()
    di.load(Paths.datasets_dict[str(res.did)])

    lg = LG.LandmarksGrabber(di.landmarks['gt']['format'],
                             di.landmarks['gt']['filename'],
                             di.landmarks['gt']['calib_filename'])

    lnames, landmarks = res.get_model_landmarks(model_name)


    ### Calculating the sequence average error in mm.
    seq_dists = []
    joint_trans = []
    for frame in landmarks:
        lg.seek(frame)
        gt_names, gt_landmarks, gt_clb = lg.acquire()
        _, l_cor, _, g_cor =  LG.GetCorrespondingLandmarks(model_name, lnames, landmarks[frame],
                                                     di.landmarks['gt']['format'], gt_names,gt_landmarks)
        lnp = np.array(l_cor)
        gnp = np.array(g_cor)
        dists = np.linalg.norm(lnp-gnp,axis=1)
        joint_trans.append(lnp-gnp)
        avg_dist = np.average(dists)
        seq_dists.append(avg_dist)
    seq_dist = np.average(np.array(seq_dists))
    seq_joint_trans = np.average(np.array(joint_trans), axis=0)
    print fname, seq_dist,

    ### Calculating the sequence average error after removing the mean offset per joint over the sequence.
    seq_dists = []
    for frame in landmarks:
        lg.seek(frame)
        gt_names, gt_landmarks, gt_clb = lg.acquire()
        _, l_cor, _, g_cor = LG.GetCorrespondingLandmarks(model_name, lnames, landmarks[frame],
                                                          di.landmarks['gt']['format'], gt_names, gt_landmarks)
        lnp = np.array(l_cor)
        gnp = np.array(g_cor)
        dists = np.linalg.norm(lnp - gnp - seq_joint_trans, axis=1)

        avg_dist = np.average(dists)
        seq_dists.append(avg_dist)
    seq_dist = np.average(np.array(seq_dists))
    print 'corrected dist:', seq_dist

if __name__ == "__main__":
    CalculateMetricsDir(input_dir=input_dir)

