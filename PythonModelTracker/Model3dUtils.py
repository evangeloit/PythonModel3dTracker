import numpy as np
import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.PythonModelTracker.LandmarksGrabber as LG
import PythonModel3dTracker.PythonModelTracker.DepthMapUtils as DMU



def GetLandmarkPos(lname, landmarks):
    for l in landmarks:
        if lname == l.linked_geometry: return l.pos
    return None


def GetNode(tf, tfname):
    if tf.name == tfname:
        return tf
    else:
        for c in tf.children:
            c = GetNode(c, tfname)
            if c is not None: return c
    return None


def GetNodeChildren(model3d):
    lnames, landmarks = GetDefaultModelLandmarks(model3d)
    tf = mbv.Dec.TransformNode()
    mbv.PF.LoadTransformNode(model3d.transform_node_bones_filename, tf)
    node_children = {}
    for l in landmarks:
        cur_node = GetNode(tf, l.linked_geometry)
        node_children[l.linked_geometry] = [c.name for c in cur_node.children]
    return node_children


def GetBoneLengths(model3d):
    lnames, landmarks = GetDefaultModelLandmarks(model3d)
    tf = mbv.Dec.TransformNode()
    mbv.PF.LoadTransformNode(model3d.transform_node_bones_filename, tf)
    bone_lengths = {}
    for l in landmarks:
        bone_lengths[l.linked_geometry] = GetBoneLength(l.linked_geometry, tf, landmarks)
    return bone_lengths



def GetBoneLength(bone_name, tf, landmarks):
    b0 = GetNode(tf, bone_name)
    if b0 is None: return 0
    if len(b0.children) != 1: return 0
    b1 = b0.children[0]
    p0 = GetLandmarkPos(b0.name, landmarks)
    p1 = GetLandmarkPos(b1.name, landmarks)
    length = mbv.Core.glm.distance(p0, p1)
    return length


def GetDefaultModelLandmarks(model3d, landmark_names=None):
    # pf.Landmark3dInfoVec()
    if landmark_names is None:
        landmark_names = model3d.parts.parts_map['all']
    if model3d.model_type == mbv.PF.Model3dType.Primitives:
        landmarks = mbv.PF.Landmark3dInfoPrimitives.create_multiple(landmark_names,
                                                                landmark_names,
                                                                mbv.PF.ReferenceFrame.RFGeomLocal,
                                                                mbv.Core.Vector3fStorage([mbv.Core.Vector3(0, 0, 0)]),
                                                                model3d.parts.primitives_map)
    else:
        # print 'Landmark names:',landmark_names
        # print 'Bones:',model3d.parts.bones_map
        landmarks = mbv.PF.Landmark3dInfoSkinned.create_multiple(landmark_names,
                                                              landmark_names,
                                                             mbv.PF.ReferenceFrame.RFGeomLocal,
                                                             mbv.Core.Vector3fStorage([mbv.Core.Vector3(0, 0, 0)]),
                                                             model3d.parts.bones_map)

        transform_node = mbv.Dec.TransformNode()
        mbv.PF.LoadTransformNode(model3d.transform_node_filename, transform_node)
        landmarks_decoder = mbv.PF.LandmarksDecoder()
        landmarks_decoder.convertReferenceFrame(mbv.PF.ReferenceFrame.RFModel, transform_node, landmarks)

    return landmark_names, landmarks


def GetCorrespondingLandmarks(model_name, ldm_model_names, ldm_model, ldm_obs_source, ldm_obs_names, ldm_obs):
    lnames_cor = LG.LandmarksGrabber.getPrimitiveNamesfromLandmarkNames(ldm_obs_names, ldm_obs_source, model_name)
    idx_obs = [i for i, g in enumerate(lnames_cor) if g != 'None']
    idx_model = [ldm_model_names.index(g) for g in lnames_cor if g != 'None']

    names_model_cor = [ldm_model_names[l] for l in idx_model]
    if ldm_model is None:
        ldm_model_cor = None
    else:
        ldm_model_cor = [ldm_model[l] for l in idx_model]
    names_obs_cor = [ldm_obs_names[l] for l in idx_obs]
    ldm_obs_cor = [ [float(ldm_obs[l].data[0, 0]), float(ldm_obs[l].data[1, 0]),
                     float(ldm_obs[l].data[2, 0])] for l in idx_obs]
    return names_model_cor, ldm_model_cor, names_obs_cor, ldm_obs_cor


def GenerateModelLandmarksfromObservationLandmarks(model3d, ldm_obs_source, ldm_obs_names = None):
    ldm_obs_source = str(ldm_obs_source)
    model_name = str(model3d.model_name)
    prim_dict = LG.primitives_dict[(ldm_obs_source, model_name)]

    if ldm_obs_names is None:
        ldm_obs_names = [l for l in prim_dict]
    if (ldm_obs_source, model_name) in LG.model_landmark_positions:
        pos_dict = LG.model_landmark_positions[(ldm_obs_source, model_name)]
    else:
        pos_dict = None


    model_primitive_names = mbv.Core.StringVector([LG.primitives_dict[(ldm_obs_source, model_name)][n] for n in ldm_obs_names ])

    model_landmark_names = mbv.Core.StringVector(ldm_obs_names)
    #print model_primitive_names
    if pos_dict:
        model_landmark_positions = mbv.Core.Vector3fStorage([mbv.Core.Vector3(lp) for n,lp in pos_dict.items()])
    else:
        model_landmark_positions = mbv.Core.Vector3fStorage([mbv.Core.Vector3([0,0,0])] * len(ldm_obs_names))
    #print len(model_landmark_positions), len(model_landmark_names), len(model_primitive_names)

    model_landmarks = mbv.PF.Landmark3dInfoSkinned.create_multiple(model_landmark_names,
                                                                   model_primitive_names,
                                                                   mbv.PF.ReferenceFrame.RFGeomLocal,
                                                                   model_landmark_positions,
                                                                   model3d.parts.bones_map)
    transform_node = mbv.Dec.TransformNode()
    mbv.PF.LoadTransformNode(model3d.transform_node_filename, transform_node)
    landmarks_decoder = mbv.PF.LandmarksDecoder()
    landmarks_decoder.convertReferenceFrame(mbv.PF.ReferenceFrame.RFModel, transform_node, model_landmarks)
    return model_landmark_names, model_landmarks


def GetInterpModelLandmarks(model3d, default_bones=None, interpolated_bones=None, n_interp=5):
    assert model3d.model_type == mbv.PF.Model3dType.Skinned

    if interpolated_bones is None: interpolated_bones = []
    if default_bones is None: default_bones = model3d.parts.parts_map['all']
    default_bones = [d for d in default_bones if d not in interpolated_bones]

    bone_lengths = GetBoneLengths(model3d)

    landmark_names = []
    bone_names = []
    positions = []
    for b in default_bones:
        landmark_names.append(str(b))
        bone_names.append(str(b))
        positions.append(mbv.Core.Vector3(0, 0, 0))

    for b in interpolated_bones:
        cur_y = np.linspace(0., bone_lengths[b], n_interp, endpoint=True)
        for i, y in enumerate(cur_y):
            lname = "{0}_{1:02d}".format(b, i)
            landmark_names.append(lname)
            bone_names.append(str(b))
            positions.append(mbv.Core.Vector3(0, y, 0))

    landmarks = mbv.PF.Landmark3dInfoSkinned.create_multiple(mbv.Core.StringVector(landmark_names),
                                                             mbv.Core.StringVector(bone_names),
                                                             mbv.PF.ReferenceFrame.RFGeomLocal,
                                                             mbv.Core.Vector3fStorage(positions),
                                                             model3d.parts.bones_map)
    transform_node = mbv.Dec.TransformNode()
    mbv.PF.LoadTransformNode(model3d.transform_node_filename, transform_node)
    landmarks_decoder = mbv.PF.LandmarksDecoder()
    landmarks_decoder.convertReferenceFrame(mbv.PF.ReferenceFrame.RFModel, transform_node, landmarks)

    return landmark_names, landmarks


def p2d_interp(p0, p1, n):
    X = np.linspace(p0.x, p1.x, n, endpoint=True)
    Y = np.linspace(p0.y, p1.y, n, endpoint=True)
    points = mbv.Core.Vector2fStorage()
    for x,y in zip(X,Y): points.append(mbv.Core.Vector2(x,y))
    return points

def p3d_interp(p0, p1, n):
    X = np.linspace(p0.x, p1.x, n, endpoint=True)
    Y = np.linspace(p0.y, p1.y, n, endpoint=True)
    Z = np.linspace(p0.z, p1.z, n, endpoint=True)
    points = mbv.Core.Vector3fStorage()
    for x,y,z in zip(X,Y,Z): points.append(mbv.Core.Vector3(x,y,z))
    return points


def GetInterpKeypointsModel(landmark_source, model3d, point_names,  keypoints2d, interpolate_set, n_interp=5):
    point_pairs = GetConsecutiveKeypointPairs(point_names, landmark_source, model3d, interpolate_set)
    point_names_, keypoints2d_ = \
        GetInterpKeypoints(point_names=point_names,
                           keypoints2d=keypoints2d, point_pairs=point_pairs, n_interp=n_interp)
    return point_names_, keypoints2d_


def GetInterpKeypointsModelSets(landmark_source, model3d, point_names, keypoints2d, interpolate_set, n_interp=5):
    point_pairs = GetConsecutiveKeypointPairs(point_names, landmark_source, model3d, interpolate_set)
    point_set_names_, point_names_, keypoints2d_ = \
        GetInterpKeypointsSets(point_names=point_names,
                               keypoints2d=keypoints2d, point_pairs=point_pairs, n_interp=n_interp)
    return point_set_names_, point_names_, keypoints2d_


def GetConsecutiveKeypointPairs(point_names, landmark_source, model3d, selected_bones):
    children = GetNodeChildren(model3d)
    bone_names = LG.LandmarksGrabber.getPrimitiveNamesfromLandmarkNames(point_names, landmark_source, model3d.model_name)
    bone_names= [b for b in bone_names]
    point_pairs = []
    for b0 in selected_bones:
        b1 = children[b0][0]
        p0 = point_names[bone_names.index(b0)]
        p1 = point_names[bone_names.index(b1)]
        point_pairs.append( (p0, p1) )
    return point_pairs


def GetInterpKeypoints(point_names, keypoints2d, point_pairs=[], n_interp=5):
    _, point_names_sets, keypoints2d_sets = \
        GetInterpKeypointsSets(point_names, keypoints2d, point_pairs, n_interp)

    point_names_ = [p for ps in point_names_sets for p in ps]
    keypoints2d_ = mbv.Core.Vector2fStorage([p for ps in keypoints2d_sets for p in ps])
    #keypoints3d_ = mbv.Core.Vector3fStorage([p for ps in keypoints3d_sets for p in ps])
    return point_names_, keypoints2d_


def GetInterpKeypointsSets(point_names, keypoints2d, point_pairs=[], n_interp=5):
    #children = GetNodeChildren(model3d)
    kp2d_dict = {}
    for n, p2d in zip(point_names, keypoints2d): kp2d_dict[n] = p2d
    #kp3d_dict = {}
    #for n, p3d in zip(point_names, keypoints3d): kp3d_dict[n] = p3d

    interpolate_set = [p0 for (p0,p1) in point_pairs]
    default_set = [n for n in point_names if n not in interpolate_set]

    point_set_names_ =[]
    point_names_ = []
    keypoints2d_ = []
    #keypoints3d_ = []

    for n in default_set:
        point_set_names_.append(n)
        point_names_.append([n])
        k2d = mbv.Core.Vector2fStorage([kp2d_dict[n]])
        keypoints2d_.append(k2d)
        #k3d = DMU.UnprojectPoints(k2d, camera, depth)
        #keypoints3d_.append(k3d)

    for n0,n1 in point_pairs:
        keypoints2d_cur = mbv.Core.Vector2fStorage()
        #keypoints3d_cur = mbv.Core.Vector3fStorage()
        point_names_cur = []
        point_set_names_.append(n0)

        p0 = kp2d_dict[n0]
        p1 = kp2d_dict[n1]
        if (p0.x > 0) and (p0.y > 0) and (p1.x > 0) and (p1.y > 0):
            cur_p2d = p2d_interp(p0, p1, n_interp)
            for i, p in enumerate(cur_p2d):
                lname = "{0}_{1:02d}".format(n0, i)
                point_names_cur.append(lname)
                keypoints2d_cur.append(p)
        else:
            point_names_cur.append(n0)
            keypoints2d_cur.append(p0)


        #keypoints3d_cur = DMU.UnprojectPoints(keypoints2d_cur,camera,depth)

        #p0 = kp3d_dict[n0]
        #p1 = kp3d_dict[n1]
        #cur_p3d = p3d_interp(p0, p1, n_interp)
        #for p in cur_p3d: keypoints3d_cur.append(p)
        keypoints2d_.append(keypoints2d_cur)
        #keypoints3d_.append(keypoints3d_cur)
        point_names_.append(point_names_cur)
    return point_set_names_, point_names_, keypoints2d_


