import PyMBVCore as core
import PyMBVOptimization as opt
import PyMBVDecoding as dec
import PyMBVRendering as ren
import PyMBVAcquisition as acq
import PyMBVLibraries as lib
import PyMBVParticleFilter as pf
import PyHandTrackerPF as htpf
import PyMBVOpenMesh as mbvom
import cv2
import RenderingUtils as ru
import PFInitialization as pfi
import CustomRenderingObjectiveImplementation as croi
import PFSettings as pfs
import time
import copy
import numpy as np
import os

print 'pid: ', os.getpid()

params_ds = htpf.DatasetParams()
supported_objective_types = ["kinect", "weighted", "custom"]

###
#Testing Model3dObjectiveFramework Rendering. Creates multiple (len(objective_type)) frameworks
# and calculates the individual and the combined objective.
###

#Parameters
n_frames = 5
n_hypotheses = 1
start_frame = 0
objective_type = ["kinect"]
real_data_flag = True
arch = [htpf.Architecture.cuda]


#Model Selection
model3d_name = "hand_skinned"
assert model3d_name in pfs.model3d_dict
model_class = pfs.model3d_dict[model3d_name][0]
model3d_xml = pfs.model3d_dict[model3d_name][1]
model3d = pf.Model3dMeta.create(model3d_xml)

#Dataset Selection
if model_class == "Hand":
    datasets_xml = "ds_info/ht_datasets.xml"
    did = ["dev_00","seq0","seq1","sensor"][1]
    depth_cutoff = 150
elif model_class == "Human":
    datasets_xml = "ds_info/bt_datasets.xml"
    did = ["pdt", "s09_a02"][0]
    depth_cutoff = 450


#Objective Framework Init
N = len(objective_type)
for o in objective_type: assert o in supported_objective_types
mmanager = core.MeshManager()
openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
model3d.setupMeshManager(mmanager)
model3dobj = htpf.Model3dObjectiveFrameworkRendering(mmanager)


model3dobj.decoder = model3d.createDecoder()
model3dobj.decoder.loadMeshTickets(mmanager)
renderer = htpf.Model3dObjectiveFrameworkRendering.generateDefaultRenderer(2048,2048,"opengl")
renderer.delegate.culling = ren.RendererOGLBase.Culling.CullFront
renderer.delegate.bonesPerHypothesis = model3d.n_bones
model3dobj.renderer = renderer
model3dobj.tile_size = (64,64)

if real_data_flag:
    #Real Data Acq
    params_ds.load(datasets_xml, did, model3d.model_name)
    start_frame = params_ds.start_frame
    if not len(params_ds.init_state) == model3d.n_dims: params_ds.init_state = model3d.default_state
    grabber = htpf.AutoGrabber.create(params_ds.format, params_ds.input_stream, params_ds.calib_filename)
    solution = params_ds.init_state
    model3dobj.bgfg = htpf.Model3dObjectiveFrameworkRendering.generateDefaultBGFG( "media/hands_faceP.dat", 40, 50)

else:
    #Simulated Data Acq
    grabber = lib.RGBDAcquisitionSimulation(model3dobj.decoder,model3dobj.renderer,model3dobj.mesh_manager)
    solution = core.DoubleVector(model3d.default_state)
    sle = lib.SyntheticLabelExtractor()
    sle.setPolicy_AlltoWhite_WhiteBgrk()
    model3dobj.bgfg = sle
#model3d.parts.genPrimitivesMap(model3dobj.decoder)

# objective combination
#obj_combination = htpf.ObjectiveCombination()
rois = htpf.RenderingObjectives()
#RenderingObjective Init
for w in objective_type:
    if w == "weighted":
        roi = htpf.RenderingObjectiveKinectWeighted()
        roi.model_parts = model3d.parts
        part_weights = pf.PartWeightsMap()
        for p in model3d.parts.parts_map:
            part_weights[p.key()] = 1
        part_weights['little'] = 3
        roi.part_weights = part_weights
    elif w == "kinect":
        roi = htpf.RenderingObjectiveKinect()
    elif w == "custom":
        roi = croi.MyCustomRenderingObjective()
    else:
        roi = htpf.RenderingObjectiveKinect()
    roi.depth_cutoff = depth_cutoff
    rois.append(roi)
model3dobj.rendering_objectives = rois
    #obj_combination.addObjective(roi.getPFObjective(),1.0/float(N))


#Main loop
for i in range(start_frame+n_frames):
    if not real_data_flag:
        grabber.pushHypothesis(model3d.default_state)
    images, calibs = grabber.grab()
    if i>=start_frame:

        depth = images[0]
        rgb = images[1]

        model3dobj.observations = images
        model3dobj.virtual_camera = calibs[0]
        bb = model3dobj.computeBoundingBox(solution)
        model3dobj.focus_rect = bb
        model3dobj.preprocessObservations()

        hypotheses = core.ParamVectors()
        if i>start_frame:
            #state[0] -= 10
            solution[8] += 0.25
            #state[8] = 1.2

        for h in range(n_hypotheses):
            hypotheses.append(solution)


        for a in arch:
            for i,o in enumerate(model3dobj.rendering_objectives):
                o.architecture = a
                t1 = time.clock()
                results = model3dobj.evaluate(hypotheses,i)
                t2 = time.clock()
                print objective_type[i], a, #'obj:', results.data[:,0],
                print 'time:', t2-t1
            # results = core.DoubleVector()
            #obj_combination.evaluate(hypotheses,results,0)
            #print 'obj_comb:', results.data[:, 0]


        viz = model3dobj.visualize(rgb,solution)

        #displaying the depth and rgb images
        #nncv2.imshow("depth",depth)
        cv2.imshow("rgb",viz)
        key = cv2.waitKey(0)

core.CachedAllocatorStorage.clear()