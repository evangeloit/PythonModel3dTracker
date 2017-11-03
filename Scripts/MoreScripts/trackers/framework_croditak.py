import PyMBVCore as core
import PyMBVOptimization as opt
import PyMBVDecoding as dec
import PyMBVRendering as ren
import PyMBVAcquisition as acq
import PyMBVLibraries as lib
import PyMBVParticleFilter as mpf
import PyHandTrackerPF as htpf
import cv2
import PFInitialization as pfi
import PFSettings as pfs
import numpy as np
import PFTrackingHelper as pfh

core.ScopeReportTimer.setReportDepth(-1)
params_ds = htpf.DatasetParams()

#Method Parameters
n_frames = 360                      # Total number of frames
wait_time = 0                       # cv::waitkey
meta_mult = 1.2                     # usually scale
weighted_part_mult = 1            # Set to 1 for kinect objective, any other value for weighted objective HMF.
enable_metaopt = False
pf_listener_flag = False
bgfg_type = ['skin', 'depth'][0]
model3d_name = "human_ext"

assert model3d_name in pfs.model3d_dict
model_class = pfs.model3d_dict[model3d_name][0]
model3d_xml = pfs.model3d_dict[model3d_name][1]
model3d = mpf.Model3dMeta.create(model3d_xml)

if model_class == "Hand":
    datasets_xml = "ds_info/ht_datasets.xml"
    did = ["dev_00","seq0","seq1","sensor"][1]
    depth_cutoff = 150
elif model_class == "Human":
    datasets_xml = "ds_info/bt_datasets.xml"
    did = ["pdt", "s09_a02"][0]
    depth_cutoff = 550
    bgfg_type = 'depth'

pf_params,meta_params = pfs.GetSettings(model3d,model_class)

params_ds.load(datasets_xml, did, model3d.model_name)
if not len(params_ds.init_state) == model3d.n_dims:
    print 'Invalid dataset init state: ', params_ds.init_state
    params_ds.init_state = model3d.default_state
    print 'Getting init state from model default: ', params_ds.init_state

grabber_auto = htpf.AutoGrabber.create(params_ds.format, params_ds.input_stream, params_ds.calib_filename)
grabber = htpf.FlipInputGrabber(grabber_auto,params_ds.flip_images)

rng = mpf.RandomNumberGeneratorOpencv()
pf = pfi.CreatePF(rng, model3d, pf_params)


pf.state = pfh.mult_meta(model3d.dim_types,params_ds.init_state,meta_mult)
solution = pf.state
if pf_listener_flag: pf.listener = htpf.ParticleFilterVisualizer()

mmanager = core.MeshManager()
model3d.setupMeshManager(mmanager)


if weighted_part_mult != 1:
    model3dobj = pfh.generate_model3dobjectiveframework_weighted(mmanager,model3d,depth_cutoff,
                                                                 bgfg_type,pf,weighted_part_mult)
else:
    model3dobj = pfh.generate_model3dobjectiveframework(mmanager,model3d,depth_cutoff,bgfg_type)
    model3dobj.setRenderingObjective(pfh.generate_rendering_objective_kinect(depth_cutoff))

objective = model3dobj.getPFObjective()
parallel_objective = mpf.PFObjectiveCast.toParallel(objective)

if enable_metaopt:
    mo = pfi.CreateMetaOptimizer(model3d,"meta",meta_params.metaopt_params)
    mf = pfi.CreateMetaFitter(model3d,"meta",meta_params.metafit_params)
    mf.setFrameSetupFunctions(mpf.ObservationsSet(model3dobj.setObservations),
                              mpf.FocusCamera(model3dobj.setFocus))

# Main loop
for i in range(params_ds.start_frame+n_frames):
    images, calibs = grabber.grab()
    if i > params_ds.start_frame:
        depth = images[0]
        rgb = images[1]

        model3dobj.observations = images
        model3dobj.virtual_camera = calibs[0]
        bb = model3dobj.computeBoundingBox(solution,.2)
        model3dobj.focus_rect = bb
        model3dobj.preprocessObservations()

        pf.track(solution,objective)
        if enable_metaopt:
            mo.optimize(pf,parallel_objective)
            mf.push(model3dobj.observations, solution, bb)
            mf.update(parallel_objective)
            pf.aux_models.pf.setSubState(mf.state,model3d.partitions.partitions["meta"])

        viz = model3dobj.visualize(rgb,solution)

        if pf_listener_flag:
            pf.listener.viz_single_model_overlay = mpf.VizSingleModelOverlay(model3dobj.visualize)
            pf.listener.visualize(rgb)

        #displaying the depth and rgb images
        # cl = model3dobj.observations[1]
        # cl[cl > 0] = np.iinfo(cl.dtype).max
        # cv2.imshow("labels", cl)
        cv2.imshow("depth",depth)
        cv2.imshow("rgb",viz)
        #cv2.imshow("vsm", viz1)
        key = cv2.waitKey(wait_time)
        if chr(key & 255) == 'q': break
        if chr(key & 255) == 'p': wait_time = 0
        if chr(key & 255) == 'c': wait_time = 10

