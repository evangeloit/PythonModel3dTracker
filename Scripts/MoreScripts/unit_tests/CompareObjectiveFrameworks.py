import PyMBVCore as core
import PyMBVOpenMesh as mbvom
import PyMBVOptimization as opt
import PyMBVDecoding as dec
import PyMBVRendering as ren
import PyMBVAcquisition as acq
import PyMBVLibraries as lib
import PyMBVParticleFilter as pf
import PyHandTrackerPF as htpf
import PyHandTracker as ht
import cv2
import PFSettings as pfs
import time

params_ds = htpf.DatasetParams()
supported_objective_types = ["kinect", "weighted", "custom"]

###
#Compares Model3dObjectiveFramework Rendering with HandTrackerLib.
###

#Parameters
n_frames = 12
start_frame = 0
real_data_flag = False
n_hypotheses = 12


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


# Creating Decoder/Renderer.
decoder = model3d.createDecoder()
renderer = htpf.Model3dObjectiveFrameworkRendering.generateDefaultRenderer(2048,2048,"opengl")
renderer.delegate.culling = ren.RendererOGLBase.Culling.CullFront
renderer.delegate.bonesPerHypothesis = model3d.n_bones
tile_size = (64,64)


#Initializing HandTrackerLib
htlib = ht.HandTrackerLib(2048,2048,64,64,renderer,decoder)
mmanager = htlib.meshManager
openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
model3d.setupMeshManager(mmanager)
decoder.loadMeshTickets(mmanager)

#Objective Framework Init
model3dobj = htpf.Model3dObjectiveFrameworkRendering(mmanager)
model3dobj.decoder = decoder
model3dobj.renderer = renderer


#htlib.renderer = model3dobj.renderer
#htlib.decoder = model3dobj.decoder
htlib.objectiveHelper.mode = ht.KinectObjectiveHTMode.mode_cvprw15
htlib.objectiveHelper.T = depth_cutoff


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

rois = htpf.RenderingObjectives()

roi = htpf.RenderingObjectiveKinect()
roi.depth_cutoff = depth_cutoff
roi.architecture = htpf.Architecture.cuda
rois.append(roi)
model3dobj.rendering_objectives = rois


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
            solution[2] += 15
            solution[7] += 0.15
            #state[8] = 1.2
        for h in range(n_hypotheses):
            hypotheses.append(solution)
        view, proj = htlib.step1_setupVirtualCamera(calibs[0])
        htlib.step3_zoomVirtualCamera(proj,bb)
        htlib.step5_setObservations(model3dobj.observations[1],model3dobj.observations[0])


        t1 = time.clock()
        results = model3dobj.evaluate(hypotheses,0)
        t2 = time.clock()
        ht_results = htlib.step6_evaluate(hypotheses)
        t3 = time.clock()
        #print '{', results.data[:,0], ht_results.data[:,0], '}',
        print 'calc time --- model3dobj, htlib:', t2-t1, t3-t2
        results = core.DoubleVector()

        viz = model3dobj.visualize(rgb,solution)

        #displaying the depth and rgb images
        #cv2.imshow("depth",depth)
        cv2.imshow("rgb",viz)
        key = cv2.waitKey(300)
core.CachedAllocatorStorage.clear()