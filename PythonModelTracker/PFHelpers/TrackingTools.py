import cv2
import os

import PyCeresIK
import PythonModel3dTracker.PyMBVAll as mbv
import PyModel3dTracker as m3dt

import BlenderMBV.BlenderMBVLib.BlenderMBVConversions as blconv

import PythonModel3dTracker.Paths as Paths
import PythonModel3dTracker.PythonModelTracker.DatasetInfo as dsi
import PythonModel3dTracker.PythonModelTracker.LandmarksGrabber as LG
import PythonModel3dTracker.PythonModelTracker.Model3dUtils as M3DU
import PythonModel3dTracker.PythonModelTracker.ModelTrackingGui as mtg
import PythonModel3dTracker.PythonModelTracker.ModelTrackingResults as mtr
import PythonModel3dTracker.PythonModelTracker.PFHelpers.PFInitialization as pfi
import PythonModel3dTracker.PythonModelTracker.AutoGrabber as AutoGrabber
import PythonModel3dTracker.PythonModelTracker.PFLevmar as pfl
import PythonModel3dTracker.PythonModelTracker.OpenPoseGrabber as opg





class ModelTools:
    @staticmethod
    def GenModel(model_name):
        assert model_name in Paths.model3d_dict
        model_class = Paths.model3d_dict[model_name]['class']
        model3d_xml = os.path.join(Paths.models,
                                   Paths.model3d_dict[model_name]['path'])

        model3d = mbv.PF.Model3dMeta.create(str(model3d_xml))
        if model3d.model_type == mbv.PF.Model3dType.Skinned: model3d.parts.genBonesMap()
        return model3d, model_class

class DatasetTools:
    @staticmethod
    def GenInitState(params_ds, model3d):

        if model3d.model_name in params_ds.initialization:
            ds_init_state = params_ds.initialization[model3d.model_name]
        else:
            ds_init_state = []
        if len(ds_init_state) == len(model3d.default_state):
            init_state = mbv.Core.DoubleVector(ds_init_state)
        else:
            init_state = model3d.default_state
            print('Invalid dataset init state: ', ds_init_state)
            print('Setting init state from model default: ', model3d.default_state)
        return init_state

    @staticmethod
    def GenBackground(params_ds):
        background_image = None
        if params_ds.background:
            background_image = cv2.imread(params_ds.background, 2 | 4)
        return background_image


    @staticmethod
    def GenGrabbers(params_ds,model3d, openpose_grabber = False):
        grabber = DatasetTools.GenGrabber(params_ds)
        grabber_ldm = DatasetTools.GenLandmarksGrabber(params_ds, model3d, openpose_grabber)
        return grabber, grabber_ldm

    @staticmethod
    def GenLandmarksGrabber(params_ds, model3d, openpose_grabber = False):
        grabber_ldm = None
        if openpose_grabber:
            grabber_ldm = opg.OpenPoseGrabber(model_op_path=Paths.models_openpose)
        else:
            if params_ds.landmarks and len(params_ds.landmarks) > 0:
                print('Landmarks filename: ', params_ds.landmarks['detections']['filename'])
                print('Landmarks calib_filename: ', params_ds.landmarks['detections']['calib_filename'])
                grabber_ldm = LG.LandmarksGrabber(params_ds.landmarks['detections']['format'],
                                                   params_ds.landmarks['detections']['filename'],
                                                   params_ds.landmarks['detections']['calib_filename'],
                                                   model3d.model_name)
        return grabber_ldm

    @staticmethod
    def Load(dataset):
        params_ds = dsi.DatasetInfo()
        params_ds.generate(dataset)
        return params_ds

    @staticmethod
    def GenGrabber(params_ds):
        print('Opening dataset:', params_ds.stream_filename)
        grabber = AutoGrabber.create(str(params_ds.format),
                                               mbv.Core.StringVector([str(s) for s in params_ds.stream_filename]),
                                               str(params_ds.calib_filename))

        return grabber

class ParticleFilterTools:

    @staticmethod
    def GenPF(pf_params, model3d, decoder=None, rng=None):
        if pf_params['pf']['smart_pf']: return ParticleFilterTools.GenSmartPF(pf_params, model3d, decoder, rng)
        else: return ParticleFilterTools.GenRegularPF(pf_params, model3d, rng)

    @staticmethod
    def GenSmartPF(pf_params, model3d, decoder, rng=None):
        if rng is None: rng = mbv.PF.RandomNumberGeneratorOpencv()
        landmarks_source = pf_params['pf']['smart_pf_model']
        primitive_names = LG.LandmarksGrabber.getPrimitiveNamesfromLandmarkNames(
            opg.OpenPoseGrabber.landmark_names[landmarks_source],landmarks_source, model3d.model_name)

        if pf_params['pf']['smart_pf_interpolate_bones'] and pf_params['pf']['smart_pf_interpolate_num'] > 1:
            lnames, landmarks = \
                M3DU.GetInterpModelLandmarks(model3d=model3d,default_bones=primitive_names,
                    interpolated_bones=pf_params['pf']['smart_pf_interpolate_bones'],
                    n_interp=pf_params['pf']['smart_pf_interpolate_num'])
        else:
            lnames, landmarks = M3DU.GetDefaultModelLandmarks(model3d, primitive_names)
        for i,l in enumerate(landmarks): print i, l.name, l.linked_geometry, l.bone_id, l.pos, l.ref_frame

        smart_pf = pfl.SmartPF(rng, model3d, pf_params['pf'])
        smart_pf.lnames = lnames
        smart_pf.landmarks = landmarks
        smart_pf.ba = pfl.SmartPF.CreateBA(model3d, decoder, landmarks)
        return smart_pf, rng


    @staticmethod
    def GenRegularPF(pf_params, model3d, rng=None):
        if rng is None:
            rng = mbv.PF.RandomNumberGeneratorOpencv()
        pf = pfi.CreatePF(rng, model3d, pf_params['pf'])
        pf.state = ParticleFilterTools.MultMeta(model3d.dim_types,
                                                pf_params['pf']['init_state'],
                                                pf_params['meta_mult'])

        if pf_params['pf_listener_flag']:
            pf.listener = m3dt.ParticleFilterVisualizer()
        return pf,rng

    @staticmethod
    def MultMeta(dim_types, state, mult_value):
        meta = [i == mbv.PF.DimType.Scale for i in dim_types]
        for i, (s, m) in enumerate(zip(state, meta)):
            if m:
                state[i] = s * mult_value
        return state


class ObjectiveTools:
    @staticmethod
    def GenModel3dObjectiveFrameworkRendering(mmanager, model3d, depth_cutoff, bgfg_type):
        model3dobj = m3dt.Model3dObjectiveFrameworkRendering(mmanager)
        model3dobj.decoder = model3d.createDecoder()  # m3dt.Model3dObjectiveFrameworkDecoding.generateDefaultDecoder(model3d.model_collada)
        model3dobj.renderer = \
            m3dt.Model3dObjectiveFrameworkRendering. \
                generateDefaultRenderer(2048, 2048, "opengl",
                                        model3d.n_bones,
                                        mbv.Ren.RendererOGLBase.Culling.CullFront)
        model3dobj.tile_size = (128, 128)
        if bgfg_type == 'skin':
            model3dobj.bgfg = m3dt.Model3dObjectiveFrameworkRendering.generateDefaultBGFG("media/hands_faceP.dat", 40,
                                                                                          50)
        else:
            model3dobj.bgfg = m3dt.Model3dObjectiveFrameworkRendering.generate3DBoxBGFG(depth_cutoff)
        if model3d.model_type == mbv.PF.Model3dType.Primitives:
            model3d.parts.genPrimitivesMap(model3dobj.decoder)
        else:
            model3d.parts.genBonesMap()
        return model3dobj

    @staticmethod
    def GenRenderingObjectiveKinect(depth_cutoff):
        # RenderingObjective Initialization
        rois = m3dt.RenderingObjectives()
        roi = m3dt.RenderingObjectiveKinect()
        roi.architecture = m3dt.Architecture.cuda
        roi.depth_cutoff = depth_cutoff
        rois.append(roi)
        return rois

    @staticmethod
    def GenRenderingObjectiveKinectWeighted(depth_cutoff,model3d,pf,part_multiplier):
        rois = m3dt.RenderingObjectives()
        for i, aux_model in enumerate(pf.aux_models_vec):

            roi = m3dt.RenderingObjectiveKinectWeighted()
            roi.model_parts = model3d.parts
            part_weights = mbv.PF.PartWeightsMap()
            for p in model3d.parts.parts_map:
                part_weights[p.key()] = 1
            part_weights[aux_model.part_name] = part_multiplier
            roi.part_weights = part_weights
            roi.architecture = m3dt.Architecture.cuda
            roi.depth_cutoff = depth_cutoff
            rois.append(roi)
        return rois


    @staticmethod
    def GenModel3dObjectiveFrameworkDecoding(mmanager, model3d):
        # N = len(pf.aux_models_vec)
        dof = m3dt.Model3dObjectiveFrameworkDecoding(mmanager)
        dof.decoder = model3d.createDecoder()
        if model3d.model_type == mbv.PF.Model3dType.Primitives:
            model3d.parts.genPrimitivesMap(dof.decoder)
        else:
            model3d.parts.genBonesMap()

        return dof

    @staticmethod
    def GenLandmarksDistObjective(max_dist):
        doi = m3dt.LandmarksDistObjective()

        doi.max_dist = max_dist
        dois = m3dt.DecodingObjectives()
        dois.append(doi)
        return dois

    @staticmethod
    def GenFilteredLandmarksDistObjective(max_dist, pf, model3d):
        dois = m3dt.DecodingObjectives()
        for i, aux_model in enumerate(pf.aux_models_vec):
            doi = m3dt.FilteredLandmarksDistObjective()
            doi.max_dist = max_dist
            doi.accepted_landmarks = model3d.parts.getPartPrimitives(str(aux_model.part_name))
            # roi.model_parts = model3d.parts
            dois.append(doi)
        return dois


    @staticmethod
    def SetupRenderingObjective(model3dobj,images,calibs,state):
        model3dobj.observations = images
        model3dobj.virtual_camera = calibs[0]
        bb = model3dobj.computeBoundingBox(state, .2)
        model3dobj.focus_rect = bb
        model3dobj.preprocessObservations()
        #depth_filt = dmu.Filter3DRect(depth_filt, bb, state[2], 500)

    @staticmethod
    def SetupLandmarkDistObjective(model3dobj, points3d_det_names, points3d_det, model3d):
        primitive_names = LG.LandmarksGrabber.getPrimitiveNamesfromLandmarkNames(points3d_det_names,
                                                                                  model3d.model_name)
        landmarks = M3DU.GetDefaultModelLandmarks(model3d, primitive_names)
        for g in model3dobj.decoding_objectives:
            for d in g.data():
                if (type(d) is m3dt.LandmarksDistObjective) or \
                        (type(d) is m3dt.FilteredLandmarksDistObjective):
                    d.setObservations(landmarks, points3d_det)
                    # print("Observations/landmarks num:",len(d.observations), len(d.landmarks))

    @staticmethod
    def GenCollisionsObjective(mmanager):
        col_det = mbv.Lib.CollisionDetection(mmanager)
        cyl_shape = mbv.Phys.CylinderShapeZ()
        cyl_shape.scale = mbv.Core.Vector3(1, 1, 1)
        cyl_shape.length = 2
        cyl_shape.radius = 1
        sphere_shape = mbv.Phys.SphereShape()
        sphere_shape.radius = 1
        sphere_shape.scale = mbv.Core.Vector3(1, 1, 1)
        meshes = mbv.Core.MeshTicketList()
        mmanager.enumerateMeshes(meshes)
        for m in meshes:
            mesh_filename = mmanager.getMeshFilename(m)
            if 'sphere_collision' in mesh_filename:
                print('Registering {0} from mesh {1}:{2}.'.format('sphere', m, mesh_filename))
                col_det.registerShape(mesh_filename, sphere_shape)
            if 'cylinder_collision' in mesh_filename:
                print('Registering {0} from mesh {1}:{2}.'.format('cylinder', m, mesh_filename))
                col_det.registerShape(mesh_filename, cyl_shape)
        doi = m3dt.CollisionObjective.create(col_det)
        dois = m3dt.DecodingObjectives()
        dois.append(doi)
        return dois


    @staticmethod
    def GenMeshManager(model3d):
        mmanager = mbv.Core.MeshManager()
        openmesh_loader = mbv.OM.OpenMeshLoader()
        mmanager.registerLoader(openmesh_loader)
        model3d.setupMeshManager(mmanager)
        return mmanager

    @staticmethod
    def GenObjective(mmanager,model3d,params):

        renderer = None
        decoder = None
        model3dobj = None

        if params['enable']:
            objective_weights = params['objective_weights']
            obj_weight_vec = mbv.Core.DoubleVector()
            # Model specific parameters.
            #if model_class == "Human": bgfg_type = 'depth'

            if objective_weights['rendering'] > 0:
                model3dobj = ObjectiveTools.GenModel3dObjectiveFrameworkRendering(mmanager, model3d, params['depth_cutoff'],
                                                                               params['bgfg_type'])
                renderer = model3dobj.renderer
                # if params['weighted_part_mult'] != 1:
                #     rois = ObjectiveTools.GenRenderingObjectiveKinectWeighted(params['depth_cutoff'],model3d,pf,
                #                                                               params['weighted_part_mult'])
                # else:
                rois = ObjectiveTools.GenRenderingObjectiveKinect(params['depth_cutoff'])
                model3dobj.appendRenderingObjectivesGroup(rois)
                # obj_combination.addObjective(model3dobj.getPFObjective(), objective_weights['rendering'])
                obj_weight_vec.append(objective_weights['rendering'])
            elif (objective_weights['primitives'] > 0) or (objective_weights['collisions'] > 0):
                model3dobj = ObjectiveTools.GenModel3dObjectiveFrameworkDecoding(mmanager, model3d)
                decoder = model3dobj.decoder

                if (objective_weights['primitives'] > 0):
                    model3dobj.appendDecodingObjectivesGroup(ObjectiveTools.GenLandmarksDistObjective(params['depth_cutoff']))
                    # model3dobj.appendDecodingObjectivesGroup(
                    #    PFTracking.generate_filteredlandmarksdistobjectives(depth_cutoff,pf,model3d))
                    # obj_combination.addObjective(model3dobj_dec.getPFObjective(), objective_weights['primitives'])
                    obj_weight_vec.append(objective_weights['primitives'])

                if objective_weights['collisions'] > 0:
                    dois = ObjectiveTools.GenCollisionsObjective(mmanager)
                    model3dobj.appendDecodingObjectivesGroup(dois)
                    obj_weight_vec.append(objective_weights['collisions'])
            else:
                model3dobj = m3dt.Model3dObjectiveFramework()

            model3dobj.objective_combination.weights = obj_weight_vec
            #objective = model3dobj.getPFObjective()
            #parallel_objective = mbv.PF.PFObjectiveCast.toParallel(objective)

        return model3dobj, decoder, renderer


# def init_metaoptimizer(self, pf_params):
#     self.mo = pfi.CreateMetaOptimizer(self.model3d, "meta", pf_params['meta_opt'])
#     self.mf = pfi.CreateMetaFitter(self.model3d, "meta", pf_params['meta_fit'])
#     self.mf.setFrameSetupFunctions(mbv.PF.ObservationsSet(self.model3dobj.setObservations),
#                                    mbv.PF.FocusCamera(self.model3dobj.setFocus))
#     self.enable_metaopt = True
#
# def run_metaoptimizer(self, bb):
#     self.mo.optimize(self.pf, self.parallel_objective)
#     # print("State mo:", self.pf.state[7])
#     # if len(self.mf.state)>7: print("State mf before:", self.mf.state[7])
#     self.mf.push(self.model3dobj.observations, self.pf.state, bb)
#     self.mf.update(self.parallel_objective)
#     # if len(self.mf.state) > 7: print("State mf after:",self.mf.state[7])
#     self.pf.aux_models.pf.setSubState(self.mf.state, self.model3d.partitions.partitions["meta"])


class TrackingLoopTools:
    @staticmethod
    def Grab(f,params_ds,grabbers):

        if f%50 == 0: print 'frame:', f
        grabber = grabbers[0]
        grabber_ldm = grabbers[1]
        grabber.seek(f)
        images, calibs = grabber.grab()
        if grabber_ldm is not None:
            grabber_ldm.seek(f)
            #points3d_det_names, points3d_det, ldm_calib \
            landmark_observations = grabber_ldm.acquire(images, calibs)

        else:
            landmark_observations = None
            #points3d_det_names = None
            #points3d_det = None
            #ldm_calib = None

        depth = images[0]
        depth_filt = depth
        rgb = images[1]
        if (len(rgb.shape) == 2):
            images[1] = cv2.merge((rgb, rgb, rgb))
            rgb = cv2.merge((rgb, rgb, rgb))
        observations = {}
        observations['images'] = images
        observations['calibs'] = calibs
        observations['landmarks'] = landmark_observations
        return observations


    @staticmethod
    def GenGui(visualize,params_ds):
        if visualize['enable']:
            if visualize['client'] == 'blender':
                gui = mtg.ModelTrackingGuiZeromq()
            else:
                gui = mtg.ModelTrackingGuiOpencv(visualize=visualize, init_frame=params_ds.limits[0])
        else:
            gui = mtg.ModelTrackingGuiNone(params_ds.limits[0])
        return gui

    @staticmethod
    def SetupObjetive(state, observations, model3d, model3dobj, objective_params):
        images = observations['images']
        calibs = observations['calibs']
        landmark_observations = observations['landmarks']
        #images, calibs, landmark_observations = observations  # points3d_det_names, points3d_det, ldm_calib

        if (objective_params['objective_weights']['primitives'] > 0) and (landmark_observations is not None):
            points3d_det_names = landmark_observations[0]
            points3d_det = landmark_observations[1]
            # ldm_calib = landmark_observations[2]
            ObjectiveTools.SetupLandmarkDistObjective(model3dobj, points3d_det_names, points3d_det, model3d)
        if objective_params['objective_weights']['rendering'] > 0:
            ObjectiveTools.SetupRenderingObjective(model3dobj, images, calibs, state)
        return model3dobj.getPFObjective()


    @staticmethod
    def SetupSmartPFObjective(observations, smart_pf, pf_params):
        # images = observations['images']
        # calibs = observations['calibs']
        landmark_observations = observations['landmarks']
        points3d_det_names = landmark_observations[0]
        points3d_det = landmark_observations[1][0]
        points2d_det = landmark_observations[2][0]
        ldm_calib = landmark_observations[3]
        if pf_params['smart_pf_interpolate_bones'] and pf_params['smart_pf_interpolate_num'] >1:
            smart_pf_model = pf_params['smart_pf_model']
            interpolate_set = pf_params['smart_pf_interpolate_bones']
            n_interp = pf_params['smart_pf_interpolate_num']
            points3d_det_names, points3d_det, points2d_det = \
                M3DU.GetInterpKeypointsModel(smart_pf_model, smart_pf.model3d, points3d_det_names, points3d_det, points2d_det,
                                             interpolate_set, n_interp)
        #for i,(p, k3, k2) in enumerate(zip(points3d_det_names, points3d_det, points2d_det)): print i, p, k3, k2
        smart_pf.calib = ldm_calib
        smart_pf.keypoints3d = points3d_det
        smart_pf.keypoints2d = points2d_det

        observations['landmarks'] = (points3d_det_names, [points3d_det], [points2d_det])

        return mbv.Opt.ParallelObjective(smart_pf.Objective)

    @staticmethod
    def loop(params_ds,model3d,grabbers,pf,pf_params,model3dobj,objective_params,
             visualizer, visualize_params):

        results = mtr.ModelTrackingResults(did=params_ds.did)
        gui = TrackingLoopTools.GenGui(visualize_params, params_ds)

        grabbers[0].seek(params_ds.limits[0])
        # Main loop
        #print("entering loop")
        continue_loop = True
        f = params_ds.limits[0]
        state = pf_params['init_state']
        while continue_loop:
            gui_command = gui.recv_command()
            if gui_command.name == "quit":
                continue_loop = False

            if gui_command.name == "state":
                if visualize_params['client'] == 'blender':
                    state_gui = gui.recv_state(model3d, state)
                    if state_gui is not None:
                        state = mbv.Core.DoubleVector(state_gui)
                        pf.state = state

            if gui_command.name == "init":
                if visualize_params['client'] == 'blender':
                    gui.send_init(blconv.getFrameDataMBV(model3dmeta=model3d, state=state,
                                                         frames=[params_ds.limits[0], f, params_ds.limits[1]],
                                                         scale=0.001))
                else:
                    gui_command.name = "frame"

            if gui_command.name == "frame":
                f_gui = gui.recv_frame()
                if f_gui is not None:
                    f = f_gui
                    if (f > params_ds.limits[1]) or (f < 0): break
                    observations = TrackingLoopTools.Grab(f, params_ds, grabbers)


                    if pf_params['smart_pf']:
                        objective = TrackingLoopTools.SetupSmartPFObjective(observations, pf, pf_params)
                        landmarks = pf.landmarks
                    else: landmarks = []
                    if objective_params['enable']:
                        objective = TrackingLoopTools.SetupObjetive(state,observations,model3d,
                                                                    model3dobj,objective_params)
                    pf.track(state, objective)


                    disp_landmark_sets, disp_landmark_names, disp_landmarks = \
                        TrackingLoopTools.PackLandmarks(state, model3dobj.decoder, landmarks, observations)

                    if visualize_params['enable']:
                        if visualize_params['client'] == 'blender':
                            frame_data = blconv.getFrameDataMBV(model3dmeta=model3d, state=state,
                                                                mbv_camera=observations['calibs'][0],
                                                                frames=[params_ds.limits[0], f, params_ds.limits[1]],
                                                                images=observations[0], scale=0.001)
                        else:
                            viz = visualizer.visualize_overlay(state, observations['calibs'][0],
                                                               observations['images'][1], disp_landmarks)
                            frame_data = mtg.FrameDataOpencv(rgb=viz, n_frame=f)
                    else:
                        frame_data = 'None'
                    gui.send_frame(frame_data)

            if f > params_ds.limits[1]: continue_loop = False

            results.add(f, model3d.model_name, state)

        #if res_filename is not None:
        #    results.save(res_filename)
        return results
        #mbv.Core.CachedAllocatorStorage.clear()

    @staticmethod
    def PackLandmarks(state, decoder, landmarks, observations):
        landmark_observations = observations['landmarks']
        detnames = landmark_observations[0]
        detpoints = landmark_observations[1][0]
        # Pack landmarks
        disp_landmark_sets = []
        disp_landmark_names = []
        disp_landmarks = []

        lnames = [l.name for l in landmarks]
        landmarks_decoder = mbv.PF.LandmarksDecoder()
        landmarks_decoder.decoder = decoder
        landmark_points = landmarks_decoder.decode(state, landmarks)

        if (len(lnames) > 0) and (len(detnames) > 0):
            disp_landmark_sets = ['LandmarksModel', 'LandmmarksObs']
            disp_landmark_names = [lnames, detnames]
            disp_landmarks = [landmark_points, detpoints]
        elif len(lnames) > 0:
            disp_landmark_sets = ['LandmarksModel']
            disp_landmark_names = [lnames]
            disp_landmarks = [landmark_points]

        elif len(detnames) > 0:
            disp_landmark_sets = ['LandmarksObs']
            disp_landmark_names = [detnames]
            disp_landmarks = [detpoints]

        return disp_landmark_sets, disp_landmark_names, disp_landmarks



@staticmethod
def background_subtraction(images, background_image, thres):
    if (background_image is not None):
        # print("bg:",np.min(background_image), np.max(background_image))
        # print("input:", np.min(images[0]), np.max(images[0]))
        # cv2.imshow("input_depth",images[0])
        # cv2.imshow("bg_depth", images[0])
        # cv2.waitKey(0)
        depth_mask = ((background_image - images[0]) > thres)
        images[0] = depth_mask * images[0]
    return images


@staticmethod
def get_model(model_name):
    assert model_name in Paths.model3d_dict
    model_class = Paths.model3d_dict[model_name]['class']
    model3d_xml = Paths.model3d_dict[model_name]['path']
    model3d = mbv.PF.Model3dMeta.create(str(model3d_xml))
    return model3d, model_class





