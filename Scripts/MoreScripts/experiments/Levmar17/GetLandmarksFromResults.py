import os.path

import PythonModel3dTracker.PythonModelTracker.ModelTrackingResults as mtr
from PythonModel3dTracker.PythonModelTracker.ResultLandmarksGenerator import GenerateLandmarks
import PythonModel3dTracker.Paths as Paths
import PythonModel3dTracker.PythonModelTracker.PyMBVAll as mbv
import PythonModel3dTracker.PythonModelTracker.PFHelpers.TrackingTools as tt
import PythonModel3dTracker.PythonModelTracker.LandmarksGrabber as LG



dry_run = False
input_dir = os.path.join(Paths.results, "Human_tracking/Levmar/")

# AMMAR Synthetic MHAD Correspondences.
landmark_names = [
 "L.UArm", "L.LArm", "L.Wrist",
 "R.UArm", "R.LArm", "R.Wrist",
 "L.ULeg", "L.LLeg", "L.Foot",
 "R.ULeg", "R.LLeg", "R.Foot",
 "neck.001", "neck", "root"
]

landmark_positions = [ [0,0,0], [0,0,0], [0,0,0],
                       [0,0,0], [0,0,0], [0,0,0],
                       [0,150,0], [0,0,0], [0,0,0],
                       [0,150,0], [0,0,0], [0,0,0],
                       [0,120,0], [0,0,0], [0,0,0]]


# MHAD Correspondences
# landmark_names = ['L.torso', 'L.ULeg', 'L.LLeg',
#                   'R.torso', 'R.ULeg', 'R.LLeg',
#                   'L.shoulder','L.UArm', 'L.Wrist',
#                   'R.shoulder','R.UArm', 'R.Wrist',
#                   'neck.001',  'root']
# landmark_positions = [ [0,440,0], [0,350,0], [0,350,0],
#                        [0,440,0], [0,350,0], [0,350,0],
#                        [0,150,0], [0,200,0], [0,70,0],
#                        [0,150,0], [0,200,0], [0,70,0],
#                        [0,-20,0], [0,0,0]]
landmark_names_mbv = mbv.Core.StringVector(landmark_names)
landmark_positions_mbv = mbv.Core.Vector3fStorage([mbv.Core.Vector3(lp) for lp in landmark_positions])



for i,f in enumerate(os.listdir(input_dir)):
    results_in = os.path.join(input_dir, f)
    f_base, f_ext = os.path.splitext(f)
    if (f_ext == '.json') and os.path.isfile(results_in):
        results_out = results_in#os.path.join(input_dir, f_base + '_ldm.json')
        print i,results_in, results_out
        if dry_run == False:
            results = mtr.ModelTrackingResults()
            results.load(results_in)
            results.landmark_names = {}
            results.landmarks = {}
            model_name = results.models[0]
            model3d, _ = tt.ModelTools.GenModel(model_name)
            #landmark_names, landmarks = LG.GetDefaultModelLandmarks(model3d)
            landmarks = mbv.PF.Landmark3dInfoSkinned.create_multiple(landmark_names_mbv,
                                                                     landmark_names_mbv,
                                                                     mbv.PF.ReferenceFrame.RFGeomLocal,
                                                                     landmark_positions_mbv,
                                                                     model3d.parts.bones_map)
            transform_node = mbv.Dec.TransformNode()
            mbv.PF.LoadTransformNode(model3d.transform_node_filename, transform_node)
            landmarks_decoder = mbv.PF.LandmarksDecoder()
            landmarks_decoder.convertReferenceFrame(mbv.PF.ReferenceFrame.RFModel, transform_node, landmarks)
            results_ldm = GenerateLandmarks(results,landmark_names, landmarks)
            results_ldm.save(results_out)





