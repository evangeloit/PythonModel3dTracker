import PyMBVRendering as ren
import os

import BlenderMBV.BlenderMBVLib.BlenderMBVConversions as blconv
import BlenderMBV.BlenderMBVLib.RenderingUtils as ru
import PyMBVCore as core
import PyMBVOpenMesh as mbvom
import PyMBVParticleFilter as mpf
import PythonModel3dTracker.PythonModelTracker.PFSettings as pfs
import cv2

import PythonModel3dTracker.PythonModelTracker.LandmarksGrabber as ldm
import PythonModel3dTracker.PythonModelTracker.ModelTrackingGui as mtg
import PythonModel3dTracker.PythonModelTracker.ModelTrackingResults as mtr

os.chdir(os.environ['bmbv']+"/Scripts/")

wait_time = 1
results_txt = Paths.results + "/Hand_tracking/experiment_all_fing1-hand_skinned_rds_80.json"
results_txt_out = ""#"rs/Human_tracking/kostas_good_01_out.json"
visualize = {'enable':True,
             'client': 'opencv',
             'labels':True, 'depth':True, 'rgb':True, 'wait_time':0}
assert visualize['client'] in ['opencv','blender']

#Loading state vectors from txt file.
results = mtr.ModelTrackingResults()
results.load(results_txt)
#state_vectors = svu.load_states(results_txt, model3d.n_dims)
#print('Loaded states for {0} frames'.format(len(state_vectors)))
#state_vectors = svu.smooth_states(state_vectors)
model_name = results.models[0]
assert model_name in pfs.model3d_dict
model_class = pfs.model3d_dict[model_name][0]
model3d_xml = pfs.model3d_dict[model_name][1]
model3d = mpf.Model3dMeta.create(model3d_xml)
datasets_xml = results.datasets_xml
did = results.did

sel_landmarks = 0   #see datasets_xml for available landmarks.




output_video = ["", Paths.results + "/{0}_tracking/{1}.avi".format(model_class, did)][0]
output_frames = ["", Paths.results + "/{0}_tracking/{1}/{2}.png".format(model_class, did, "{}")][0]

video_writer = None
if len(output_video) > 5:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video, fourcc, 20.0, (640, 480))


ds = htpf.HTDataset(datasets_xml)
params_ds = ds.getDatasetInfo(did)

grabber_auto = htpf.AutoGrabber.create(params_ds.format, params_ds.stream_filename, params_ds.calib_filename)
grabber = htpf.FlipInputGrabber(grabber_auto,params_ds.flip_images)
grabber_ldm = None

if len(params_ds.lm_meta_info)>sel_landmarks:
    print('Landmarks filename: ', params_ds.lm_meta_info[sel_landmarks].filename)
    grabber_ldm = ldm.LandmarksGrabber(params_ds.lm_meta_info[sel_landmarks].format,
                                       params_ds.lm_meta_info[sel_landmarks].filename,
                                       params_ds.lm_meta_info[sel_landmarks].calib_filename,
                                       model3d.model_name)

mmanager = core.MeshManager()
openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
model3d.setupMeshManager(mmanager)
decoder = model3d.createDecoder()
decoder.loadMeshTickets(mmanager)
renderer = ren.RendererOGLCudaExposed.get(2048, 2048)
renderer.bonesPerHypothesis = model3d.n_bones

dof = htpf.Model3dObjectiveFrameworkDecoding(mmanager)
dof.decoder = decoder
if model3d.model_type == mpf.Model3dType.Primitives: model3d.parts.genPrimitivesMap(dof.decoder)
else: model3d.parts.genBonesMap()
#landmarks = ldm.GetDefaultModelLandmarks(model3d, core.StringVector(['f_pinky.03.R', 'f_middle.03.R', 'f_ring.03.R', 'thumb.03.R', 'f_index.03.R']))
#landmarks_decoder = mpf.LandmarksDecoder()
#landmarks_decoder.decoder = decoder


# Gui Initialization
if visualize['client'] == 'opencv':
    gui = mtg.ModelTrackingGuiOpencv()
else:
    gui = mtg.ModelTrackingGuiZeromq()

# Main loop
continue_loop = True
points3d_det = None
f = params_ds.limits[0]
state = []
grabber.seek(f)
while continue_loop:
    gui_command = gui.recv_command()
    if gui_command.name == "quit":
        continue_loop = False

    if gui_command.name == "init":
        if results.has_state(f, model3d.model_name):
            state = core.DoubleVector(results.states[f][model3d.model_name])
        if visualize['client'] == 'blender':
            gui.send_init(blconv.FrameDataMBV(model3d, state, None, [params_ds.limits[0], f, params_ds.limits[1]],
                                              None, 0.001))
        else:
            gui_command.name = "frame"
            gui.next_frame = f

    if gui_command.name == "state":
        if visualize['client'] == 'blender':
            state_gui = gui.recv_state(model3d, state)
            if state_gui is not None:
                state = state_gui
                print('Received state: ', state)
                results.states[f][model3d.model_name] = state

    if gui_command.name == "frame":
        f_gui = gui.recv_frame()
        if f_gui is not None:
            f = f_gui
            if (f > params_ds.limits[1]) or (f < 0): break
            print('frame:', f)
            grabber.seek(f)
            images, calibs = grabber.grab()
            depth = images[0]
            rgb = images[1]
            if (len(rgb.shape) == 2):
                images[1] = cv2.merge((rgb, rgb, rgb))
                rgb = cv2.merge((rgb, rgb, rgb))

            cur_results_flag = False
            cur_results_flag = results.has_state(f, model3d.model_name)
            if cur_results_flag:
                state = core.DoubleVector(results.states[f][model3d.model_name])
            if grabber_ldm is not None:
                grabber_ldm.seek(f)
                points3d_det_names,  points3d_det, ldm_calib = grabber_ldm.acquire()


            frame_data = None
            if visualize['client'] == 'blender':
                frame_data = blconv.FrameDataMBV(model3d, state, calibs[0],
                                                 [params_ds.limits[0], f, params_ds.limits[1]], depth, 0.001)
            else:
                if cur_results_flag:
                    #points3d_ldm = landmarks_decoder.decode(state, landmarks)
                    viz = ru.visualize_overlay(renderer, mmanager, decoder, state, calibs[0],rgb,model3d.n_bones)#,points3d_ldm)
                else: viz = rgb
                frame_data = mtg.FrameDataOpencv(depth, None, viz, f)
            gui.send_frame(frame_data)

results.save(results_txt_out)
