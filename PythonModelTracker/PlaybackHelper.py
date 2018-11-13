import os
import cv2

import BlenderMBV.BlenderMBVLib.BlenderMBVConversions as blconv
import BlenderMBV.BlenderMBVLib.RenderingUtils as ru
import PythonModel3dTracker.PyMBVAll as mbv
import PyModel3dTracker as htpf


import PythonModel3dTracker.PythonModelTracker.Grabbers.AutoGrabber as AutoGrabber
import PythonModel3dTracker.PythonModelTracker.Grabbers.DatasetInfo as dsi
import PythonModel3dTracker.PythonModelTracker.Landmarks.LandmarksGrabber as LG
import PythonModel3dTracker.PythonModelTracker.Landmarks.Model3dLandmarks as M3DL
import PythonModel3dTracker.PythonModelTracker.ModelTrackingGui as mtg
import PythonModel3dTracker.PythonModelTracker.TrackingResults.ModelTrackingResults as mtr
from PythonModel3dTracker.ObjectDetection.RigidObjectOptimizer import RigidObjectOptimizer
import PythonModel3dTracker.Paths as Paths


class PlaybackHelper:

    def __init__(self,video_filename="",frames_filename=""):
        self.model3d = None
        self.params_ds = None
        self.grabber = None
        self.grabber_auto = None
        self.grabber_ldm = None
        self.results = None
        self.mmanager = None
        self.openmesh_loader = None
        self.decoder = None
        self.renderer = None
        self.dof = None
        self.save_dataset_json = False
        self.video_writer = None
        self.frames_filename = None

        if video_filename is not None:
            if len(video_filename) > 5:
                fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

        if frames_filename is not None:
            if len(frames_filename) > 5:
                self.frames_filename = frames_filename
                directory = os.path.dirname(frames_filename)
                if not os.path.exists(directory):
                    os.makedirs(directory)


    def __del__(self):
        if self.video_writer is not None: self.video_writer.release()
        mbv.Core.CachedAllocatorStorage.clear()

    def set_results(self,results_json,sel_landmarks=None):
        self.results = mtr.ModelTrackingResults()
        self.results.load(results_json)
        self.set_model(self.results.models[0])
        self.set_dataset(self.results.did,sel_landmarks)

    def set_model(self,model_name):
        assert model_name in Paths.model3d_dict
        model3d_xml = Paths.model3d_dict[model_name]['path']
        self.model3d = mbv.PF.Model3dMeta.create(str(model3d_xml))



    def set_dataset(self,dataset,sel_landmarks=None, calib_filename=None):
        #assert self.model3d is not None

        self.params_ds = dsi.DatasetInfo()
        self.params_ds.generate(str(dataset))
        if not os.path.isfile(self.params_ds.json_filename):
            self.save_dataset_json = True
        # if not (self.params_ds.did in paths.datasets_dict):
        #     paths.datasets_dict[self.params_ds.did] = self.params_ds.json_filename
        #     paths.save_datasets_dict(paths.datasets_dict)
        self.sel_landmarks = sel_landmarks
        self.params_ds.calib_filename = calib_filename



    def init_grabber(self):
        assert self.params_ds is not None
        print self.params_ds.calib_filename
        self.grabber = AutoGrabber.create(str(self.params_ds.format),
                                          mbv.Core.StringVector([str(s) for s in self.params_ds.stream_filename]),
                                          str(self.params_ds.calib_filename))
        #self.grabber = htpf.FlipInputGrabber(self.grabber_auto, self.params_ds.flip_images)
        self.grabber_ldm = None

        if self.sel_landmarks and self.params_ds.landmarks and (self.sel_landmarks in self.params_ds.landmarks):
            print('Landmarks filename: ', self.params_ds.landmarks[self.sel_landmarks]['filename'])
            self.grabber_ldm = LG.LandmarksGrabber(self.params_ds.landmarks[self.sel_landmarks]['format'],
                                                    self.params_ds.landmarks[self.sel_landmarks]['filename'],
                                                    self.params_ds.landmarks[self.sel_landmarks]['calib_filename'],
                                                    self.model3d.model_name)

    def init_mbv_rendering_stack(self):
        assert self.model3d is not None
        self.mmanager = mbv.Core.MeshManager()
        self.openmesh_loader = mbv.OM.OpenMeshLoader()
        self.mmanager.registerLoader(self.openmesh_loader)
        self.model3d.setupMeshManager(self.mmanager)
        self.decoder = self.model3d.createDecoder()
        self.decoder.loadMeshTickets(self.mmanager)
        self.exposed_renderer = \
            htpf.Model3dObjectiveFrameworkRendering. \
                generateDefaultRenderer(2048, 2048, "opengl",self.model3d.n_bones,
                                        mbv.Ren.RendererOGLBase.Culling.CullNone)#mbv.Ren.RendererOGLCudaExposed.get(2048, 2048)
        self.renderer = self.exposed_renderer.delegate
        self.renderer.bonesPerHypothesis = self.model3d.n_bones

        self.dof = htpf.Model3dObjectiveFrameworkDecoding(self.mmanager)
        self.dof.decoder = self.decoder
        if self.model3d.model_type == mbv.PF.Model3dType.Primitives:
            self.model3d.parts.genPrimitivesMap(self.dof.decoder)
        else:
            self.model3d.parts.genBonesMap()
            # landmarks = LG.GetDefaultModelLandmarks(model3d, mbv.Core.StringVector(['f_pinky.03.R', 'f_middle.03.R', 'f_ring.03.R', 'thumb.03.R', 'f_index.03.R']))
            #self.landmarks_decoder = mbv.PF.LandmarksDecoder()
            #self.landmarks_decoder.decoder = self.decoder

    def get_init_state(self,f):
        if self.model3d is None:
            state = []
        elif self.results is not None:
            if self.results.has_state(f, self.model3d.model_name):
                state = mbv.Core.DoubleVector(self.results.states[f][self.model3d.model_name])
            elif self.results.has_state(f+1, self.model3d.model_name):
                state = mbv.Core.DoubleVector(self.results.states[f+1][self.model3d.model_name])
        elif (self.model3d.model_name in self.params_ds.initialization) and \
             (len(self.params_ds.initialization[self.model3d.model_name]) == self.model3d.n_dims):
            state = mbv.Core.DoubleVector(self.params_ds.initialization[self.model3d.model_name])
        else:
            state = self.model3d.default_state
        return state

    def playback_loop(self,visualize):
        #assert self.model3d is not None
        assert self.params_ds is not None
        self.init_grabber()
        if self.model3d is not None: self.init_mbv_rendering_stack()
        # Gui Initialization
        if visualize['client'] == 'opencv':
            gui = mtg.ModelTrackingGuiOpencv()
        else:
            gui = mtg.ModelTrackingGuiZeromq()


        f = self.params_ds.limits[0]
        state = self.get_init_state(f)
        self.grabber.seek(f)
        continue_loop = True
        while continue_loop:
            gui_command = gui.recv_command()
            if gui_command.name == "quit":
                continue_loop = False

            if gui_command.name == "background":
                background_filename = os.path.join(self.params_ds.json_dir, self.params_ds.did+'_bg.png')
                self.params_ds.background = background_filename
                self.save_dataset_json = True
                print('Saving background to <{0}>.'.format(background_filename))
                cv2.imwrite(background_filename,depth)
                pass

            if gui_command.name == "init":
                if visualize['client'] == 'blender':
                    gui.send_init(blconv.getFrameDataMBV(self.model3d, state, None, None, None, None,
                                                         [self.params_ds.limits[0], f, self.params_ds.limits[1]],
                                                        None, 0.001))
                else:
                    gui_command.name = "frame"
                    gui.next_frame = f

            if gui_command.name == "optimize":
                print('Received opt request.')
                state_gui = gui.recv_state(self.model3d, state)
                if state_gui is not None:
                    state = state_gui
                    print('Received opt request, state: ', state)
                    optimizer = RigidObjectOptimizer(self.mmanager, self.exposed_renderer, self.decoder, self.model3d)
                    state = optimizer.optimize(images, calibs, state )
                    if visualize['client'] == 'blender':

                        frame_data = blconv.getFrameDataMBV(model3dmeta=self.model3d,state=state,
                                                            landmark_sets=['LandmarkObs'],
                                                            landmark_names=[points3d_det_names],
                                                            landmark_positions=[points3d_det],
                                                            mbv_camera=calibs[0],
                                                            frames=[self.params_ds.limits[0], f, self.params_ds.limits[1]],
                                                            images=[depth, rgb],scale = 0.001)
                        gui.send_frame(frame_data)

            if gui_command.name == "state":
                if visualize['client'] == 'blender':
                    state_gui = gui.recv_state(self.model3d, state)
                    if state_gui is not None:
                        state = state_gui
                        print('Received state: ', state)
                        if self.results is not None:
                            print('Setting results state.')
                            self.results.states[f][self.model3d.model_name] = state.__pythonize__()
                        else:
                            print('Setting dataset initialization.')
                            self.params_ds.initialization[self.model3d.model_name] = state.__pythonize__()
                            self.params_ds.limits[0] = f
                            self.save_dataset_json = True


            if gui_command.name == "frame":
                f_gui = gui.recv_frame()
                if f_gui is not None:
                    f = f_gui
                    if (f > self.params_ds.limits[1]) or (f < 0): break
                    print('frame:', f)
                    self.grabber.seek(f)
                    images, calibs = self.grabber.grab()
                    depth = images[0]
                    rgb = images[1]
                    if (len(rgb.shape) == 2):
                        #images[1] = cv2.merge((rgb, rgb, rgb))
                        rgb = cv2.merge((rgb, rgb, rgb))
                    if (rgb.shape[0:2]) != (depth.shape[0:2]):
                        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]) )
                    images[1] = rgb
                    #depth_filt = dmu.Filter3DRect(depth, bb, self.state[2], 500)

                    cur_results_flag = False
                    if self.results is not None:
                        cur_results_flag = self.results.has_state(f, self.model3d.model_name)
                        if cur_results_flag:
                            state = mbv.Core.DoubleVector(self.results.states[f][self.model3d.model_name])

                    if self.grabber_ldm is not None:
                        self.grabber_ldm.seek(f)
                        points3d_det_names, points3d_det, ldm_calib = self.grabber_ldm.acquire()
                    else:
                        points3d_det = []
                        points3d_det_names = []
                        ldm_calib = None


                    # Pack landmarks
                    disp_landmark_sets = []
                    disp_landmark_names = []
                    disp_landmarks = []
                    lnames = []
                    if (self.model3d is not None) and (self.results is not None):
                        lnames, landmarks = self.results.get_model_landmarks(self.model3d.model_name)
                        landmarks = landmarks[f]
                    if (len(lnames) > 0) and (len(points3d_det_names) > 0):
                        l_names_cor, l_cor, g_names_cor, g_cor = \
                            M3DL.GetCorrespondingLandmarks(self.model3d.model_name, lnames, landmarks[f],
                                self.params_ds.landmarks[self.sel_landmarks]['format'], points3d_det_names, points3d_det)
                        disp_landmark_sets = ['LandmarksModel', 'LandmmarksObs']
                        disp_landmark_names = [l_names_cor, g_names_cor]
                        disp_landmarks = [l_cor, g_cor]
                    elif len(lnames) > 0:
                        disp_landmark_sets = ['LandmarksModel']
                        disp_landmark_names = [lnames]
                        disp_landmarks = [landmarks]

                    elif len(points3d_det_names) > 0:
                        disp_landmark_sets = ['LandmarksObs']
                        disp_landmark_names = [ points3d_det_names ]
                        disp_landmarks = [ points3d_det ]

                    frame_data = None
                    if visualize['client'] == 'blender':
                        frame_data = blconv.getFrameDataMBV(self.model3d, state,
                                                            disp_landmark_sets, disp_landmark_names, disp_landmarks,
                                                            calibs[0], [self.params_ds.limits[0], f, self.params_ds.limits[1]],
                                                            [depth, rgb], 0.001)
                    else:
                        if self.model3d is not None:
                            viz = ru.visualize_overlay(self.renderer, self.mmanager, self.decoder, state, calibs[0], rgb,
                                                       self.model3d.n_bones, disp_landmarks)
                        else:
                            viz = rgb
                        frame_data = mtg.FrameDataOpencv(depth, None, viz, f)
                        if self.video_writer is not None: self.video_writer.write(viz)
                        if self.frames_filename is not None: cv2.imwrite(self.frames_filename.format(f),viz)
                    gui.send_frame(frame_data)
        if self.save_dataset_json:
            self.params_ds.save(self.params_ds.json_filename)
