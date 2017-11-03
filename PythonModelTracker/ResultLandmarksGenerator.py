import PyMBVCore as core
import PyMBVOpenMesh as mbvom
import PyMBVParticleFilter as mpf

import PythonModelTracker.LandmarksGrabber as ldm
import PythonModelTracker.PFHelpers.TrackingTools as tt


def GenerateLandmarks(results,landmark_names=None, landmarks=None):
    model_name = results.models[0]
    model3d, _ = tt.ModelTools.GenModel(model_name)

    if (landmarks is None) or (landmark_names is None):
        landmark_names = model3d.parts.parts_map['all']
        landmarks = ldm.GetDefaultModelLandmarks(model3d, landmark_names)

    did = results.did
    params_ds = tt.DatasetTools.Load(did)

    mmanager = core.MeshManager()
    openmesh_loader = mbvom.OpenMeshLoader()
    mmanager.registerLoader(openmesh_loader)
    model3d.setupMeshManager(mmanager)

    dof = tt.ObjectiveTools.GenModel3dObjectiveFrameworkDecoding(mmanager, model3d)

    landmarks_decoder = mpf.LandmarksDecoder()
    landmarks_decoder.decoder = dof.decoder
    results.add_landmark_names(model_name,[l for l in landmark_names])

    # Main loop
    continue_loop = True
    f = params_ds.limits[0]
    state = core.DoubleVector(tt.DatasetTools.GenInitState(params_ds,model3d))
    while continue_loop:
        if (f > params_ds.limits[1]) or (f < 0): break

        cur_results_flag = results.has_state(f, model3d.model_name)
        if cur_results_flag:
            state = core.DoubleVector(results.states[f][model3d.model_name])

        points3d_ldm = landmarks_decoder.decode(state, landmarks)
        results.add_landmarks(f,model_name,points3d_ldm)

        f +=1
    return results





