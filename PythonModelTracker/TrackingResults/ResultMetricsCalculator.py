import numpy as np
import os

import PythonModel3dTracker.Paths as Paths
from PythonModelTracker.TrackingResults.ModelTrackingResults import ModelTrackingResults
import PythonModel3dTracker.PythonModelTracker.Landmarks.LandmarksGrabber as LG
import PythonModel3dTracker.PythonModelTracker.Landmarks.Model3dLandmarks as M3DL
import PythonModel3dTracker.PythonModelTracker.Grabbers.DatasetInfo as DI


def IsResultsFile(filename):
    valid_file = False
    f_base, f_ext = os.path.splitext(filename)
    if (f_ext == '.json') and os.path.isfile(filename):
        res = ModelTrackingResults()
        if res.check_file(filename): valid_file = True
    return valid_file


def CalculateMetricsDir(input_dir):
    results_metrics = {}
    results_counter = 0
    for i,f in enumerate(sorted(os.listdir(input_dir))):
        results_in = os.path.join(input_dir, f)
        res = ModelTrackingResults()
        if res.check_file(results_in, ModelTrackingResults.all_required_fields + ["parameters"]):
            print results_counter,
            results_counter += 1
            did, model_name, parameters, seq_dist, seq_dist_corr, seq_success_ratio = CalculateMetricsJson(fname=results_in)
            results_metrics[f] = {"did":did, "model_name":model_name, "parameters":parameters,
                                  "seq_dist":seq_dist, "seq_dist_corr":seq_dist_corr,
                                  "seq_success_ratio":seq_success_ratio}
    return results_metrics



def CalculateMetricsJson(fname, dist_cutoffs=[75, 100, 150, 200, 250]):

    res = ModelTrackingResults()
    res.load(fname)

    model_name = str(res.models[0])
    did = str(res.did)
    parameters = res.parameters
    di = DI.DatasetInfo()
    di.load(Paths.datasets_dict[did])

    lg = LG.LandmarksGrabber(di.landmarks['gt']['format'],
                             di.landmarks['gt']['filename'],
                             di.landmarks['gt']['calib_filename'])

    lnames, landmarks = res.get_model_landmarks(model_name)


    ### Calculating the sequence average error in mm.
    seq_dists = []
    seq_success_frames = [0] * len(dist_cutoffs)
    joint_trans = []
    for frame in landmarks:
        lg.seek(frame)
        gt_names, gt_landmarks3d, gt_landmarks2d, gt_clb, gt_src = lg.acquire()


        l_cor = landmarks[frame]
        gt_names = [n for n in gt_names]
        gt3d0 = gt_landmarks3d[0]
        g_idx = []
        for n in lnames: g_idx.append(gt_names.index(n))
        g_cor = [[float(gt3d0[l].x), float(gt3d0[l].y), float(gt3d0[l].z)] for l in g_idx]
        lnp = np.array(l_cor)
        gnp = np.array(g_cor)
        dists = np.linalg.norm(lnp-gnp,axis=1)
        dists = np.clip(dists, 0, 2 * max(dist_cutoffs))
        #print frame, 'dists:', np.average(dists)
        for i, c in enumerate(dist_cutoffs):
            if np.average(dists) < c: seq_success_frames[i] +=1
        joint_trans.append(lnp-gnp)
        avg_dist = np.average(dists)
        seq_dists.append(avg_dist)
    seq_success_ratio = [(dc, float(sf)/float(len(landmarks))) for dc, sf in zip(dist_cutoffs, seq_success_frames)]
    seq_dist = np.average(np.array(seq_dists))
    seq_joint_trans = np.average(np.array(joint_trans), axis=0)
    print fname, "dist:", seq_dist, "C:", seq_success_ratio

    ### Calculating the sequence average error after removing the mean offset per joint over the sequence.
    seq_dists = []
    for frame in landmarks:
        lg.seek(frame)
        gt_names, gt_landmarks3d, gt_landmarks2d, gt_clb, gt_src = lg.acquire()
        l_cor = landmarks[frame]
        gt_names = [n for n in gt_names]
        gt3d0 = gt_landmarks3d[0]
        g_idx = []
        for n in lnames: g_idx.append(gt_names.index(n))
        g_cor = [[float(gt3d0[l].x), float(gt3d0[l].y), float(gt3d0[l].z)] for l in g_idx]

        lnp = np.array(l_cor)
        gnp = np.array(g_cor)
        dists = np.linalg.norm(lnp - gnp - seq_joint_trans, axis=1)

        avg_dist = np.average(dists)
        seq_dists.append(avg_dist)
    seq_dist_corr = np.average(np.array(seq_dists))
    #print 'corrected dist:', seq_dist_corr
    return did, model_name, parameters, seq_dist, seq_dist_corr, seq_success_ratio


