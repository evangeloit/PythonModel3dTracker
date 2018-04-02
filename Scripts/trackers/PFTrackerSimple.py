import PyMBVCore as core
import PyMBVOptimization as opt
import PyMBVDecoding as dec
import PyMBVRendering as ren
import PyMBVAcquisition as acq
import PyMBVLibraries as lib
import PyMBVParticleFilter as mpf
import PyMBVOpenMesh as mbvom
import PyHandTrackerPF as htpf
import cv2
import PFInitialization as pfi
import PFSettings as pfs
import numpy as np
import PFTrackingHelper as pfh
import LandmarksGrabber as ldm

core.ScopeReportTimer.setReportDepth(-1)


#General Parameters
wait_time = 0   # cv::waitkey
frames_num = [1, 100]
depth_cutoff = 550
bgfg_type = ['skin', 'depth'][1]
input_sequence = 'ds/hand_tracking/generic/seq1.oni'
input_format = htpf.StreamFormat.SFOni
flip_input = True

# 3D Model Selection
model3d_name = "hand_skinned"
model_class = ["Hand", "Human"][0]
model3d_xml = pfs.model3d_dict[model3d_name][1]
model3d = mpf.Model3dMeta.create(model3d_xml)
init_state = model3d.default_state

pf_params,meta_params = pfs.GetSettings(model3d,model_class)


# Grabber Initialization
grabber_auto = htpf.AutoGrabber.create(input_format, input_sequence, '')
grabber = htpf.FlipInputGrabber(grabber_auto,flip_input)

# PF Initialization
rng = mpf.RandomNumberGeneratorOpencv()
pf = pfi.CreatePF(rng, model3d, pf_params)
pf.state = init_state
solution = pf.state

# MeshManagaer Initialization.
mmanager = core.MeshManager()
openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
model3d.setupMeshManager(mmanager)

# RenderingObjective Initialization
rendering_objective = htpf.RenderingObjectiveKinect()
rendering_objective.architecture = htpf.Architecture.cuda
rendering_objective.depth_cutoff = depth_cutoff

# Model3dObjective Framework Initialization.
model3dobj = htpf.Model3dObjectiveFrameworkRendering(mmanager)
model3dobj.decoder = model3d.createDecoder()
model3dobj.renderer = \
    htpf.Model3dObjectiveFrameworkRendering.generateDefaultRenderer(2048, 2048, "opengl",
                                                                    model3d.n_bones,
                                                                    ren.RendererOGLBase.Culling.CullFront)
if bgfg_type == 'skin':
    model3dobj.bgfg = htpf.Model3dObjectiveFrameworkRendering.generateDefaultBGFG("media/hands_faceP.dat", 40, 50)
else:
    model3dobj.bgfg = htpf.Model3dObjectiveFrameworkRendering.generate3DBoxBGFG(depth_cutoff)
model3dobj.setRenderingObjective(rendering_objective)
objective = model3dobj.getPFObjective()


# Main loop
for i in range(frames_num[1]):
    images, calibs = grabber.grab()

    if i > frames_num[0]:
        depth = images[0]
        rgb = images[1]

        model3dobj.observations = images
        model3dobj.virtual_camera = calibs[0]
        bb = model3dobj.computeBoundingBox(solution,.2)
        model3dobj.focus_rect = bb
        model3dobj.preprocessObservations()

        pf.track(solution,objective)

        viz = model3dobj.visualize(rgb,solution)

        cv2.imshow("rgb",viz)
        key = cv2.waitKey(wait_time)
        if chr(key & 255) == 'q': break
        if chr(key & 255) == 'p': wait_time = 0
        if chr(key & 255) == 'c': wait_time = 10

core.CachedAllocatorStorage.clear()

