import os.path

import PyMBVCore as core
import PyMBVDecoding as dec
import PyMBVOpenMesh as mbvom
import PyMBVParticleFilter as mpf
import PyModel3dTracker as htpf
import PythonModelTracker.PFSettings as pfs

import PythonModelTracker.LandmarksGrabber as ldm
import PythonModelTracker.ModelTrackingResults as mtr


#os.chdir(os.environ['hts'])

def GetRoditakModelLandmarks(model3d, landmark_names):
    # pf.Landmark3dInfoVec()
    landmarks = mpf.Landmark3dInfoSkinned.create_multiple(landmark_names,
                                                         landmark_names,
                                                         mpf.ReferenceFrame.RFGeomLocal,
                                                         core.Vector3fStorage([core.Vector3(0, 0, 0)]),
                                                         model3d.parts.bones_map)
    endeff_bones = ['f_index.03.R','f_middle.03.R','f_pinky.03.R','f_ring.03.R','thumb.03.R']
    endeff_names = [b+'_ee' for b in endeff_bones]
    landmarks_ee = mpf.Landmark3dInfoSkinned.create_multiple(core.StringVector(endeff_names),
                                                         core.StringVector(endeff_bones),
                                                         mpf.ReferenceFrame.RFGeomLocal,
                                                         core.Vector3fStorage([core.Vector3(0, 15.1, 0)]),
                                                         model3d.parts.bones_map)
    for ee in landmarks_ee:
        landmarks.append(ee)
    print('landmarks:')
    for l in landmarks:
        print(l.name,l.pos)
    transform_node = dec.TransformNode()
    mpf.LoadTransformNode(model3d.transform_node_filename, transform_node)
    landmarks_decoder = mpf.LandmarksDecoder()
    landmarks_decoder.convertReferenceFrame(mpf.ReferenceFrame.RFModel, transform_node, landmarks)
    return landmarks

def SaveRoditakJoints(dir,frame,names,points):
    fname = dir + 'joints{:04d}.txt'.format(frame)
    print('Saving joints to {0}, frame {1}.'.format(fname,frame))

    point_order = ['hand.R', 'palm_pinky.R', 'f_pinky.01.R', 'f_pinky.02.R', 'f_pinky.03.R', 'f_pinky.03.R_ee', 'palm_middle.R', 'f_middle.01.R',
     'f_middle.02.R', 'f_middle.03.R', 'f_middle.03.R_ee', 'palm_ring.R', 'f_ring.01.R', 'f_ring.02.R', 'f_ring.03.R', 'f_ring.03.R_ee',
     'thumb.01.R', 'thumb.02.R', 'thumb.03.R', 'thumb.03.R_ee', 'palm_index.R', 'f_index.01.R', 'f_index.02.R', 'f_index.03.R',
     'f_index.03.R_ee']
    p_dict = {}
    for n,p in zip(names,points):
        p_dict[n] = p

    of = open(fname,'w')

    for pn in point_order:
        cur_p = p_dict[pn]
        cur_p_str = '{0} {1} {2} \n'.format(cur_p.x,cur_p.y,cur_p.z)
        #print(pn, cur_p_str)
        of.write(cur_p_str)
    of.close()


wait_time = 1

for file in os.listdir(Paths.results+ "/Hand_tracking/rds/"):
    if file.endswith(".json"):
        results_txt = os.path.join(Paths.results + "/Hand_tracking/rds/", file)

        parts = os.path.splitext(file)
        run = parts[0][-1]
        results_dir = Paths.results + "/Hand_tracking/rds/{0}/{1}/res_all_joints/".format(parts[0][0:len(parts[0]) - 1], run)
        print(results_txt,results_dir)



        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_txt_out = ""#"rs/Human_tracking/kostas_good_01_out.json"

        #Loading state vectors from txt file.
        results_flag = False
        assert os.path.isfile(results_txt)
        results = mtr.ModelTrackingResults()
        results.load(results_txt)

        model_name = results.models[0]
        assert model_name in pfs.model3d_dict
        model_class = pfs.model3d_dict[model_name][0]
        model3d_xml = pfs.model3d_dict[model_name][1]
        model3d = mpf.Model3dMeta.create(model3d_xml)
        datasets_xml = results.datasets_xml
        did = results.did
        sel_landmarks = 0   #see datasets_xml for available landmarks.


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


        dof = htpf.Model3dObjectiveFrameworkDecoding(mmanager)
        dof.decoder = decoder
        if model3d.model_type == mpf.Model3dType.Primitives: model3d.parts.genPrimitivesMap(dof.decoder)
        else: model3d.parts.genBonesMap()
        landmarks = GetRoditakModelLandmarks(model3d, core.StringVector([b.key() for b in model3d.parts.bones_map]))
        landmarks_decoder = mpf.LandmarksDecoder()
        landmarks_decoder.decoder = decoder


        # Main loop
        continue_loop = True
        points3d_det = None
        f = params_ds.limits[0]
        state = core.DoubleVector(params_ds.getInitState(model3d.model_name))
        grabber.seek(f)
        while continue_loop:
            if (f > params_ds.limits[1]) or (f < 0): break
            #print('frame:', f)
            cur_results_flag = results.has_state(f, model3d.model_name)
            if cur_results_flag:
                state = core.DoubleVector(results.states[f][model3d.model_name])
            #print(state)
            points3d_ldm = landmarks_decoder.decode(state, landmarks)
            SaveRoditakJoints(results_dir,f,[n.name for n in landmarks],points3d_ldm)
            f +=1





