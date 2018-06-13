import PyMBVRendering as ren
import os.path

import BlenderMBVLib.RenderingUtils as ru
import PyMBVCore as core
import PyMBVOpenMesh as mbvom
import PyMBVParticleFilter as mpf
import PyModel3dTracker as htpf
import PythonModel3dTracker.PythonModelTracker.PFSettings as pfs
import cv2

import PythonModel3dTracker.PythonModelTracker.Landmarks.LandmarksGrabber as ldm
import PythonModel3dTracker.PythonModelTracker.TrackingResults.ModelTrackingResults as mtr

os.chdir(os.environ['bmbv']+'/Scripts')

wait_time = 0
results_txt = Paths.results + "Hand_tracking/roditak-hand_skinned.json"
results_txt_out = ""#"rs/Human_tracking/kostas_good_01_out.json"

#Loading state vectors from txt file.
results_flag = False
if os.path.isfile(results_txt):
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
    results_flag = True

else:
    # Dataset selection.
    model_class = "Hand"
    datasets_xml = {"Hand": Paths.ds_info + "/ht_datasets.xml",
                    "Human": Paths.ds_info + "/bt_datasets.xml"}[model_class]
    dids = {'Hand': ["roditak","iasonas","seq0","seq1","big_01"][0],
           'Human':["kostas_bad_01","mhad_s09_a01","mhad_s09_a02","mhad_s09_a03","mhad_s09_a04","mhad_s09_a05","mhad_s09_a06","mhad_s09_a07",
                     "mhad_s09_a08", "mhad_s09_a09", "mhad_s09_a10", "mhad_s09_a11"][0]
           }
    did = dids[model_class]
sel_landmarks = 0   #see datasets_xml for available landmarks.

output_dir = Paths.results + "/{0}_tracking/{1}/".format(model_class, did)
output_video = ["", "{0}{1}.avi".format(output_dir,did)][1]
output_frames = ["", "{0}{1}.png".format(output_dir,"{:04d}")][1]
output_depth = ["", "{0}depth_{1}.png".format(output_dir,"{:04d}")][1]

video_writer = None
if len(output_video) > 5:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video, fourcc, 20.0, (640, 480))


ds = htpf.HTDataset(datasets_xml)
params_ds = ds.getDatasetInfo(did)

grabber_auto = htpf.AutoGrabber.create(params_ds.format, params_ds.stream_filename, params_ds.calib_filename)
grabber = htpf.FlipInputGrabber(grabber_auto,params_ds.flip_images)
grabber_ldm = None
if results_flag:
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

    # doi = htpf.LandmarksDistObjective()
    # doi.max_dist = 500
    # dois = htpf.DecodingObjectives()
    # dois.append(doi)
    # dof.decoding_objectives = dois
    # landmarks_decoder = mpf.LandmarksDecoder()
    # landmarks_decoder.decoder = decoder


# Main loop
grabber.seek(params_ds.limits[0])
for i in range(params_ds.limits[0],params_ds.limits[1]):
    cur_results_flag = False
    if results_flag:
        cur_results_flag = results.has_state(i, model3d.model_name)
        if cur_results_flag:
            state = results.states[i][model3d.model_name]
            if grabber_ldm is not None:
                points3d_det_names, points3d_det, ldm_calib = grabber_ldm.acquire()
                # print("Frame {0}, point names:{1}".format(i, points3d_det_names))

    images, calibs = grabber.grab()



    depth = images[0]
    rgb = images[1]
    viz = cv2.putText(rgb, "frame {0}".format(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if cur_results_flag:
        viz = ru.visualize_overlay(renderer,mmanager,decoder,core.DoubleVector(state),calibs[0],viz,model3d.n_bones)

    if grabber_ldm is not None:
        viz = ru.disp_landmarks(calibs[0], viz, points3d_det, (255, 0, 0))
        # TODO: fix primitives names mismatch between bvh/damien.
        # primitive_names = grabber_ldm.getPrimitiveNamesfromLandmarkNames(points3d_det_names)
        # landmarks = ldm.GetDefaultModelLandmarks(model3d, primitive_names)
        # for d in dof.decoding_objectives:
        #     d.setObservations(landmarks, points3d_det)
        # if cur_results_flag:
        #     cur_state_vecs = core.ParamVectors()
        #     cur_state_vecs.append(state)
        #     landmarks_dist = dof.evaluate(cur_state_vecs,0)
        #     landmarks_model = landmarks_decoder.decode(state,landmarks)
        #     viz = ru.disp_landmarks(calibs[0], viz, landmarks_model, (0,0,255))
        #     #print("Frame {0}, lanmarks_dist:{1}".format(i,landmarks_dist))

    #displaying the depth and rgb images
    cv2.imshow("rgb",viz)
    cv2.imshow("depth", depth)
    key = cv2.waitKey(wait_time)
    if chr(key & 255) == 'q': break
    if chr(key & 255) == 'p': wait_time = 0
    if chr(key & 255) == 'c': wait_time = 10
    if chr(key & 255) == 's':
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        print(output_frames.format(i), output_depth.format(i))
        cv2.imwrite(output_frames.format(i),viz)
        cv2.imwrite(output_depth.format(i), depth)

    if video_writer is not None:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        video_writer.write(viz)

if video_writer is not None:
    video_writer.release()
