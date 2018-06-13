import cv2
import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.Paths as Paths
import PythonModel3dTracker.PythonModelTracker.Grabbers.DatasetInfo as DSI
import PythonModel3dTracker.PythonModelTracker.Grabbers.AutoGrabber as AutoGrabber
from PythonModel3dTracker.PythonModelTracker.PFHelpers.VisualizationTools import Visualizer
from PythonModelTracker.TrackingResults.ModelTrackingResults import ModelTrackingResults
import PythonModel3dTracker.PythonModelTracker.Model3dUtils.Model3dUtils as M3DU


model_name = 'mh_body_male_custom_1050'
results_json = '/home/mad/Development/Results/Human_tracking/Levmar/mhad_26dofs/mhad_s05_a04_mh_body_male_custom_1050_p20_lp20_ransac[0.05, 0.8].json'
dataset = 'mhad_s05_a04'
depth_cutoff = 500

#Creating Model
mmanager = mbv.Core.MeshManager()
model3d_xml = Paths.model3d_dict[model_name]['path']
model3d = mbv.PF.Model3dMeta.create(str(model3d_xml))
model3d.parts.genBonesMap()
model3d.setupMeshManager(mmanager)



#Loading Dataset
params_ds = DSI.DatasetInfo()
params_ds.generate(dataset)
grabber = AutoGrabber.create_di(params_ds)

# Creating Parts Objective
modelparts_confidence = M3DU.ModelPartsConfidence(model3d=model3d, mesh_manager=mmanager)
renderer = modelparts_confidence.model3dobj.renderer
decoder = modelparts_confidence.model3dobj.decoder


#Loading Tracking Results
results = ModelTrackingResults()
results.load(results_json)

#Initializing Visualizer
visualizer = Visualizer(model3d, mmanager, decoder, renderer )

#Loop
for f in range(5):
    images, calibs = grabber.grab()
    camera = calibs[0]
    rgb = images[1]
    state = model3d.default_state
    if results.has_state(f, model3d.model_name):
        state = mbv.Core.DoubleVector(results.states[f][model3d.model_name])
    #state[0] += 600

    states = mbv.Core.ParamVectors()
    for p in model3d.parts.parts_map:
        states.append(state)

    obj_vals = modelparts_confidence.process(images, calibs, state)
    viz = modelparts_confidence.visualize(rgb, camera, state, obj_vals, excluded_parts=['all', 'torso'])

    cv2.imshow("rgb", viz)
    cv2.waitKey(0)

mbv.Core.CachedAllocatorStorage.clear()