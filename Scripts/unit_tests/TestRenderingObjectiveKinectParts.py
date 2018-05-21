import cv2
import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.Paths as Paths
import PythonModel3dTracker.PythonModelTracker.DatasetInfo as DSI
import PythonModel3dTracker.PythonModelTracker.AutoGrabber as AutoGrabber
from PythonModel3dTracker.PythonModelTracker.PFHelpers.VisualizationTools import Visualizer
from PythonModel3dTracker.PythonModelTracker.ModelTrackingResults import ModelTrackingResults
import PyModel3dTracker as M3DT


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
model3dobj = M3DT.Model3dObjectiveFrameworkRendering(mmanager)
model3dobj.decoder = model3d.createDecoder()  # m3dt.Model3dObjectiveFrameworkDecoding.generateDefaultDecoder(model3d.model_collada)
model3dobj.renderer = \
    M3DT.Model3dObjectiveFrameworkRendering. \
        generateDefaultRenderer(2048, 2048, "opengl",
                                model3d.n_bones,
                                mbv.Ren.RendererOGLBase.Culling.CullFront)
#model3dobj.tile_size = (128, 128)
model3dobj.bgfg = M3DT.Model3dObjectiveFrameworkRendering.generate3DBoxBGFG(depth_cutoff)
rois = M3DT.RenderingObjectives()
roi = M3DT.RenderingObjectiveKinectParts()
roi.model_parts = model3d.parts
roi.architecture = M3DT.Architecture.cuda
roi.depth_cutoff = depth_cutoff
rois.append(roi)
model3dobj.appendRenderingObjectivesGroup(rois)


#Loading Tracking Results
results = ModelTrackingResults()
results.load(results_json)

meshes = mbv.Core.MeshTicketList()
mmanager.enumerateMeshes(meshes)
model3d.parts.mesh = mmanager.getMesh(meshes[0])

#Initializing Visualizer
visualizer = Visualizer(model3d, mmanager, model3dobj.decoder, model3dobj.renderer )

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

    # model3dobj.observations = images
    # model3dobj.virtual_camera = calibs[0]
    # bb = model3dobj.computeBoundingBox(state, .2)
    # model3dobj.focus_rect = bb
    # model3dobj.preprocessObservations()
    model3dobj.evaluateSetup(images, calibs[0], state, .2)
    obj_vals = model3dobj.evaluate(states, 0)


    for p, o in zip(model3d.parts.parts_map, obj_vals):
        print p.key(), o
    viz = visualizer.visualize_parts(state, camera, rgb, model3d.parts)

    #viz = visualizer.visualize_overlay(state, camera, rgb)
    cv2.imshow("rgb", viz)
    cv2.waitKey(0)

mbv.Core.CachedAllocatorStorage.clear()