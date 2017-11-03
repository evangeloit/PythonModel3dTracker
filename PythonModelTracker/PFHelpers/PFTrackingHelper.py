import PyCeresIK as IK
import PyMBVRendering as ren
import numpy as np
import os

import BlenderMBVLib.BlenderMBVConversions as blconv
import BlenderMBVLib.RenderingUtils as ru
import PyMBVCore as core
import PyMBVAcquisition as acq
import PyMBVLibraries as lib
import PyMBVOpenMesh as mbvom
import PyMBVParticleFilter as mpf
import PyMBVPhysics as phys
import PyModel3dTracker as htpf
import cv2

import PythonModelTracker.AutoGrabber as AutoGrabber
import PythonModelTracker.DatasetInfo as dsi
import PythonModelTracker.DepthMapUtils as dmu
import PythonModelTracker.LandmarksGrabber as ldm
import PythonModelTracker.ModelTrackingGui as mtg
import PythonModelTracker.ModelTrackingResults as mtr
import PythonModelTracker.OpenPoseGrabber as opg
import PythonModelTracker.PFHelpers.PFInitialization as pfi
import PythonModelTracker.PFLevmar as pfl
import PathsPM3DT as Paths


class PFTracking:
    def __init__(self, model3d, model_class):
        core.ScopeReportTimer.setReportDepth(-1)
        
        self.enable_metaopt = False
        self.pf_listener_flag = False

        self.model_class = model_class
        self.model3d = model3d
        
        self.params_ds = None

        self.init_state = None
        self.background_image = None
        self.landmarks_flag = False
        self.smart_pf_flag = False
        self.grabber_ldm = None
        self.landmarks = None
        self.landmarks_decoder = None
        self.state = None

        self.grabber = None
        self.mmanager = None
        self.renderer = None
        self.decoder = None
        self.openmesh_loader = None
        #self.obj_combination = None
        self.objective_weights = None
        self.results = None

    def __del__(self):
        core.CachedAllocatorStorage.clear()



    def load_dataset(self,dataset):
        self.params_ds = dsi.DatasetInfo()
        self.params_ds.generate(dataset)
        self.results = mtr.ModelTrackingResults(did=self.params_ds.did)

        if self.model3d.model_name in self.params_ds.initialization:
            ds_init_state = self.params_ds.initialization[self.model3d.model_name]
        else: ds_init_state = []
        if len(ds_init_state) == len(self.model3d.default_state):
            self.init_state = core.DoubleVector(ds_init_state)
        else:
            self.init_state = self.model3d.default_state
            print('Invalid dataset init state: ', ds_init_state)
            print('Setting init state from model default: ', self.model3d.default_state)

        if self.params_ds.background:
            self.background_image = cv2.imread(self.params_ds.background, 2 | 4)

        if self.params_ds.landmarks and len(self.params_ds.landmarks)>0:
            print('Landmarks filename: ', self.params_ds.landmarks['detections']['filename'])
            print('Landmarks calib_filename: ', self.params_ds.landmarks['detections']['calib_filename'])
            self.grabber_ldm = ldm.LandmarksGrabber(self.params_ds.landmarks['detections']['format'],
                                                    self.params_ds.landmarks['detections']['filename'],
                                                    self.params_ds.landmarks['detections']['calib_filename'],
                                                    self.model3d.model_name)


        print('Opening dataset:', self.params_ds.stream_filename)
        grabber_auto = AutoGrabber.create(str(self.params_ds.format),
                                               core.StringVector([str(s) for s in self.params_ds.stream_filename]),
                                               str(self.params_ds.calib_filename))

        self.grabber = grabber_auto
        #self.grabber = htpf.FlipInputGrabber(grabber_auto, self.params_ds.flip_images)
        return  self.params_ds


    def init_pf(self, pf_params,meta_mult, pf_listener_flag):
        self.rng = mpf.RandomNumberGeneratorOpencv()
        if pf_params['pf']['smart_pf']:
            self.smart_pf = pfl.SmartPF(self.rng,self.model3d,pf_params['pf'])
            self.pf = self.smart_pf.pf
            self.smart_pf_flag = True
        else:
            self.pf = pfi.CreatePF(self.rng, self.model3d, pf_params['pf'])
        self.pf.state = PFTracking.mult_meta(self.model3d.dim_types, self.init_state, meta_mult)
        self.state = self.pf.state
        self.pf_listener_flag = pf_listener_flag
        if pf_listener_flag:
            self.pf.listener = htpf.ParticleFilterVisualizer()

    def init_objective(self, weighted_part_mult, bgfg_type, objective_weights,depth_cutoff):
        self.mmanager = core.MeshManager()
        self.openmesh_loader = mbvom.OpenMeshLoader()
        self.mmanager.registerLoader(self.openmesh_loader)
        self.model3d.setupMeshManager(self.mmanager)
        #self.obj_combination = htpf.ObjectiveCombination()
        self.objective_weights = objective_weights
        obj_weight_vec = core.DoubleVector()

        # Model specific parameters.
        if self.model_class == "Human": bgfg_type = 'depth'

        if objective_weights['rendering'] > 0 :
            if weighted_part_mult != 1:
                self.model3dobj = PFTracking.generate_model3dobjectiveframework_weighted(self.mmanager,self.model3d,depth_cutoff,
                                                                                  bgfg_type,self.pf,weighted_part_mult)
            else:
                self.model3dobj = PFTracking.generate_model3dobjectiveframework(self.mmanager,self.model3d,depth_cutoff,bgfg_type)
                rois = htpf.RenderingObjectives()
                roi = PFTracking.generate_rendering_objective_kinect(depth_cutoff)
                rois.append(roi)
                self.model3dobj.appendRenderingObjectivesGroup(rois)
            #self.obj_combination.addObjective(self.model3dobj.getPFObjective(), objective_weights['rendering'])
            obj_weight_vec.append(objective_weights['rendering'])
            self.renderer = self.model3dobj.renderer
            self.decoder = self.model3dobj.decoder
        else:
            self.renderer = htpf.Model3dObjectiveFrameworkRendering.generateDefaultRenderer(
                1024, 1024, "opengl", self.model3d.n_bones, ren.RendererOGLBase.Culling.CullFront)
            self.model3dobj = PFTracking.generate_decoding_model3dobjectiveframework(self.mmanager, self.model3d)

        if objective_weights['openpose']: self.grabber_ldm = opg.OpenPoseGrabber()
        if (objective_weights['primitives'] > 0 and (self.grabber_ldm is not None)) or \
            objective_weights['collisions'] > 0:

            if (objective_weights['primitives'] > 0) and (self.grabber_ldm is not None):
                self.landmarks_flag = True
                #self.model3dobj_dec.decoding_objectives = PFTracking.generate_landmarksdistobjective(depth_cutoff)
                self.model3dobj.appendDecodingObjectivesGroup(PFTracking.generate_landmarksdistobjective(depth_cutoff))
                #self.model3dobj.appendDecodingObjectivesGroup(
                #    PFTracking.generate_filteredlandmarksdistobjectives(depth_cutoff,self.pf,self.model3d))
                #self.obj_combination.addObjective(self.model3dobj_dec.getPFObjective(), objective_weights['primitives'])
                if self.decoder is None: self.decoder = self.model3dobj.decoder
                self.landmarks_decoder = mpf.LandmarksDecoder()
                self.landmarks_decoder.decoder = self.decoder
                obj_weight_vec.append(objective_weights['primitives'])

            if objective_weights['collisions'] > 0:
                self.col_det = lib.CollisionDetection(self.mmanager)
                cyl_shape = phys.CylinderShapeZ()
                cyl_shape.scale = core.Vector3(1, 1, 1)
                cyl_shape.length = 2
                cyl_shape.radius = 1
                sphere_shape = phys.SphereShape()
                sphere_shape.radius = 1
                sphere_shape.scale = core.Vector3(1, 1, 1)
                meshes = core.MeshTicketList()
                self.mmanager.enumerateMeshes(meshes)
                for m in meshes:
                    mesh_filename = self.mmanager.getMeshFilename(m)
                    if 'sphere_collision' in mesh_filename:
                        print('Registering {0} from mesh {1}:{2}.'.format('sphere', m,mesh_filename))
                        self.col_det.registerShape(mesh_filename, sphere_shape)
                    if 'cylinder_collision' in mesh_filename:
                        print('Registering {0} from mesh {1}:{2}.'.format('cylinder',m, mesh_filename))
                        self.col_det.registerShape(mesh_filename, cyl_shape)
                doi = htpf.CollisionObjective.create(self.col_det)
                dois = htpf.DecodingObjectives()
                dois.append(doi)
                self.model3dobj.appendDecodingObjectivesGroup(dois)
                if self.decoder is None: self.decoder = self.model3dobj.decoder
                obj_weight_vec.append(objective_weights['collisions'])

        if self.decoder is None: self.decoder = self.model3d.createDecoder()
        self.model3dobj.objective_combination.weights = obj_weight_vec
        self.objective = self.model3dobj.getPFObjective()
        self.parallel_objective = mpf.PFObjectiveCast.toParallel(self.objective)

        if self.smart_pf_flag:
            primitive_names = ldm.LandmarksGrabber.getPrimitiveNamesfromLandmarkNames(
                self.model3d.parts.parts_map['all'],
                self.model3d.model_name)
            self.landmarks = ldm.GetDefaultModelLandmarks(self.model3d, primitive_names)
            self.smart_pf.ba = pfl.SmartPF.CreateBA(self.model3d, self.decoder, self.landmarks,
                                                    IK.ModelAwareBundleAdjuster.MHCUSTOM_TO_COCO)
            # self.smart_pf.model3dobj = self.model3dobj
            # self.objective = opt.ParallelObjective(self.smart_pf.Objective)
        
        

    def init_metaoptimizer(self,pf_params):
        self.mo = pfi.CreateMetaOptimizer(self.model3d,"meta",pf_params['meta_opt'])
        self.mf = pfi.CreateMetaFitter(self.model3d,"meta",pf_params['meta_fit'])
        self.mf.setFrameSetupFunctions(mpf.ObservationsSet(self.model3dobj.setObservations),
                                       mpf.FocusCamera(self.model3dobj.setFocus))
        self.enable_metaopt = True

    def run_metaoptimizer(self,bb):
        self.mo.optimize(self.pf, self.parallel_objective)
        #print("State mo:", self.pf.state[7])
        #if len(self.mf.state)>7: print("State mf before:", self.mf.state[7])
        self.mf.push(self.model3dobj.observations, self.pf.state, bb)
        self.mf.update(self.parallel_objective)
        #if len(self.mf.state) > 7: print("State mf after:",self.mf.state[7])
        self.pf.aux_models.pf.setSubState(self.mf.state, self.model3d.partitions.partitions["meta"])

    def load_primitives_objective(self,points3d_det_names, points3d_det):
        primitive_names = ldm.LandmarksGrabber.getPrimitiveNamesfromLandmarkNames(points3d_det_names,self.model3d.model_name)
        self.landmarks = ldm.GetDefaultModelLandmarks(self.model3d, primitive_names)
        for g in self.model3dobj.decoding_objectives:
            for d in g.data():
                if (type(d) is htpf.LandmarksDistObjective) or\
                   (type(d) is htpf.FilteredLandmarksDistObjective):
                    d.setObservations(self.landmarks, points3d_det[0])
                    #print("Observations/landmarks num:",len(d.observations), len(d.landmarks))



    def loop(self,visualize):
        gui = None
        if visualize['enable']:
            if visualize['client'] == 'blender':
                gui = mtg.ModelTrackingGuiZeromq()
            else:
                gui = mtg.ModelTrackingGuiOpencv(visualize=visualize, init_frame=self.params_ds.limits[0])
        else: gui = mtg.ModelTrackingGuiNone(self.params_ds.limits[0])


        self.grabber.seek(self.params_ds.limits[0])
        # Main loop
        print("entering loop")
        continue_loop = True
        f = self.params_ds.limits[0]
        while continue_loop:
            gui_command  = gui.recv_command()
            if gui_command.name == "quit":
                continue_loop = False

            if gui_command.name == "state":
                if visualize['client'] ==  'blender':
                    state_gui = gui.recv_state(self.model3d, self.state)
                    if state_gui is not None:
                        self.state = core.DoubleVector(state_gui)
                        self.pf.state = self.state


            if gui_command.name == "init":
                if visualize['client'] ==  'blender':
                    gui.send_init(blconv.getFrameDataMBV(self.model3d, self.state,
                                                  None, [self.params_ds.limits[0], f,
                                                         self.params_ds.limits[1]], None, 0.001))
                else:
                    gui_command.name = "frame"

            if gui_command.name == "frame":
                f_gui = gui.recv_frame()
                if f_gui is not None:
                    f = f_gui
                    if (f > self.params_ds.limits[1]) or (f < 0): break
                    print('frame:',f)
                    self.grabber.seek(f)
                    images, calibs = self.grabber.grab()
                    if self.landmarks_flag:
                        self.grabber_ldm.seek(f)
                        points3d_det_names, points3d_det, ldm_calib = self.grabber_ldm.acquire(images, calibs)
                        if self.smart_pf_flag:
                            self.smart_pf.calib = ldm_calib
                            self.smart_pf.keypoints3d = points3d_det[0]
                            self.smart_pf.keypoints2d = self.grabber_ldm.keypoints2d[0]
                            self.pf.particles = mpf.ParticlesMatrix(self.smart_pf.DynamicModel(self.pf.particles.particles))
                        #else:

                    else:
                        points3d_det = None

                    depth = images[0]
                    depth_filt = depth
                    rgb = images[1]
                    if (len(rgb.shape) == 2):
                        images[1] = cv2.merge((rgb, rgb, rgb))
                        rgb = cv2.merge((rgb, rgb, rgb))

                    if f > self.params_ds.limits[0]:
                        # 3d point observations related stuff.
                        if self.landmarks_flag:
                            self.load_primitives_objective(points3d_det_names, points3d_det)
                        else: points3d_det = None

                        # background sub
                        if (self.background_image is not None):
                            images = PFTracking.background_subtraction(images, self.background_image, 2)

                        if self.objective_weights['rendering'] > 0:
                            self.model3dobj.observations = images
                            self.model3dobj.virtual_camera = calibs[0]
                            bb = self.model3dobj.computeBoundingBox(self.state, .2)
                            self.model3dobj.focus_rect = bb
                            self.model3dobj.preprocessObservations()
                            depth_filt = dmu.Filter3DRect(depth_filt, bb, self.state[2], 500)

                        self.pf.track(self.state, self.objective)
                        if self.enable_metaopt: self.run_metaoptimizer(bb)



                    frame_data = None
                    if visualize['enable']:
                        if visualize['client'] == 'blender':
                            frame_data = blconv.getFrameDataMBV(self.model3d, self.state, calibs[0],
                                                             [self.params_ds.limits[0], f,
                                                              self.params_ds.limits[1]],
                                                              [depth_filt, rgb],
                                                              0.001)
                        else:
                            labels = None

                            #openpose_viz = self.grabber_ldm.op.render(rgb)
                            #cv2.imshow('op',openpose_viz)
                            # opviz_filename = "/home/mad/Development/Results/Human_tracking/Levmar/openpose/{0:05d}.png".format(f)
                            # cv2.imwrite(filename=opviz_filename,img=openpose_viz)

                            if f > self.params_ds.limits[0] and visualize['labels'] and self.objective_weights['rendering'] > 0:
                                labels = self.model3dobj.observations[1]
                                labels[labels > 0] = np.iinfo(labels.dtype).max
                            points3d_ldm = None
                            if self.landmarks_flag and (self.landmarks is not None) and (self.landmarks_decoder is not None):
                                points3d_ldm = self.landmarks_decoder.decode(self.state, self.landmarks)
                            viz = ru.visualize_overlay(self.renderer, self.mmanager, self.decoder, self.state, calibs[0], rgb,
                                                       self.model3d.n_bones)#,[points3d_det[0], points3d_ldm])
                            if self.pf_listener_flag:
                                self.pf.listener.viz_single_model_overlay = mpf.VizSingleModelOverlay(self.model3dobj.visualize)
                                self.pf.listener.visualize(rgb)
                            frame_data = mtg.FrameDataOpencv(depth, labels, viz, f)
                    else: frame_data = 'None'
                    gui.send_frame(frame_data)

            if f > self.params_ds.limits[1]: continue_loop = False



            self.results.add(f,self.model3d.model_name,self.state)



    def save_trajectory(self,filename):
        if len(filename)  > 3:
            self.results.save(filename)
        else:
            print('Cannot save trajectory, invalid filename:<{0}>.'.format(filename))


    @staticmethod
    def background_subtraction(images, background_image, thres):
        if (background_image is not None):
            #print("bg:",np.min(background_image), np.max(background_image))
            #print("input:", np.min(images[0]), np.max(images[0]))
            #cv2.imshow("input_depth",images[0])
            #cv2.imshow("bg_depth", images[0])
            #cv2.waitKey(0)
            depth_mask = ((background_image - images[0]) > thres)
            images[0] = depth_mask * images[0]
        return images

    @staticmethod
    def mult_meta(dim_types, state, mult_value):
        meta = [i == mpf.DimType.Scale for i in dim_types]
        for i, (s, m) in enumerate(zip(state, meta)):
            if m:
                state[i] = s * mult_value
        return state

    @staticmethod
    def generate_model3dobjectiveframework(mmanager, model3d, depth_cutoff, bgfg_type):
        model3dobj = htpf.Model3dObjectiveFrameworkRendering(mmanager)
        model3dobj.decoder = model3d.createDecoder()  # htpf.Model3dObjectiveFrameworkDecoding.generateDefaultDecoder(model3d.model_collada)
        model3dobj.renderer = \
            htpf.Model3dObjectiveFrameworkRendering. \
                generateDefaultRenderer(2048, 2048, "opengl",
                                        model3d.n_bones,
                                        ren.RendererOGLBase.Culling.CullFront)
        model3dobj.tile_size = (128,128)
        if bgfg_type == 'skin':
            model3dobj.bgfg = htpf.Model3dObjectiveFrameworkRendering.generateDefaultBGFG("media/hands_faceP.dat", 40,
                                                                                          50)
        else:
            model3dobj.bgfg = htpf.Model3dObjectiveFrameworkRendering.generate3DBoxBGFG(depth_cutoff)
        if model3d.model_type == mpf.Model3dType.Primitives:
            model3d.parts.genPrimitivesMap(model3dobj.decoder)
        else:
            model3d.parts.genBonesMap()
        return model3dobj

    @staticmethod
    def generate_rendering_objective_kinect(depth_cutoff):
        # RenderingObjective Initialization
        rendering_objective = htpf.RenderingObjectiveKinect()
        rendering_objective.architecture = htpf.Architecture.cuda
        rendering_objective.depth_cutoff = depth_cutoff
        return rendering_objective

    @staticmethod
    def generate_model3dobjectiveframework_weighted(mmanager, model3d, depth_cutoff, bgfg_type, pf, part_multiplier):
        rois = htpf.RenderingObjectives()
        N = len(pf.aux_models_vec)
        rof = PFTracking.generate_model3dobjectiveframework(mmanager, model3d, depth_cutoff, bgfg_type)
        for i, aux_model in enumerate(pf.aux_models_vec):

            roi = htpf.RenderingObjectiveKinectWeighted()
            roi.model_parts = model3d.parts
            part_weights = mpf.PartWeightsMap()
            for p in model3d.parts.parts_map:
                part_weights[p.key()] = 1
            part_weights[aux_model.part_name] = part_multiplier
            roi.part_weights = part_weights
            roi.architecture = htpf.Architecture.cuda
            roi.depth_cutoff = depth_cutoff
            rois.append(roi)

        rof.appendRenderingObjectivesGroup(rois)

        return rof

    @staticmethod
    def generate_decoding_model3dobjectiveframework(mmanager, model3d):
        # N = len(pf.aux_models_vec)
        dof = htpf.Model3dObjectiveFrameworkDecoding(mmanager)
        dof.decoder = model3d.createDecoder()
        if model3d.model_type == mpf.Model3dType.Primitives:
            model3d.parts.genPrimitivesMap(dof.decoder)
        else:
            model3d.parts.genBonesMap()

        return dof

    @staticmethod
    def generate_landmarksdistobjective(max_dist):
        doi = htpf.LandmarksDistObjective()

        doi.max_dist = max_dist
        dois = htpf.DecodingObjectives()
        dois.append(doi)
        return dois

    @staticmethod
    def generate_filteredlandmarksdistobjectives(max_dist,pf,model3d):
        dois = htpf.DecodingObjectives()
        for i, aux_model in enumerate(pf.aux_models_vec):
            doi = htpf.FilteredLandmarksDistObjective()
            doi.max_dist = max_dist
            doi.accepted_landmarks = model3d.parts.getPartPrimitives(str(aux_model.part_name))
            #roi.model_parts = model3d.parts
            dois.append(doi)
        return dois

    @staticmethod
    def get_model(model_name):
        assert model_name in Paths.model3d_dict
        model_class = Paths.model3d_dict[model_name]['class']
        model3d_xml = os.path.join(Paths.models, Paths.model3d_dict[model_name]['path'])
        model3d = mpf.Model3dMeta.create(str(model3d_xml))
        return model3d, model_class





