import PythonModel3dTracker.PyMBVAll as mbv
import PythonModel3dTracker.PythonModelTracker.LandmarksGrabber as LG


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
    lnames, landmarks = LG.GetDefaultModelLandmarks(model3d)
    tf = mbv.Dec.TransformNode()
    mbv.PF.LoadTransformNode(model3d.transform_node_bones_filename, tf)
    node_children = {}
    for l in landmarks:
        cur_node = GetNode(tf, l.linked_geometry)
        node_children[l.linked_geometry] = [c.name for c in cur_node.children]
    return node_children


def GetBoneLengths(model3d):
    lnames, landmarks = LG.GetDefaultModelLandmarks(model3d)
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
        landmarks = mbv.PF.Landmark3dInfoSkinned.create_multiple(landmark_names,
                                                              landmark_names,
                                                             mbv.PF.ReferenceFrame.RFGeomLocal,
                                                             mbv.Core.Vector3fStorage([mbv.Core.Vector3(0, 0, 0)]),
                                                             model3d.parts.bones_map)
        #print(landmark_names)
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
    ldm_model_cor = [ldm_model[l] for l in idx_model]
    names_obs_cor = [ldm_obs_names[l] for l in idx_obs]
    ldm_obs_cor = [ [float(ldm_obs[l].data[0, 0]), float(ldm_obs[l].data[1, 0]),
                     float(ldm_obs[l].data[2, 0])] for l in idx_obs]
    return names_model_cor, ldm_model_cor, names_obs_cor, ldm_obs_cor


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
        landmark_names.append(b)
        bone_names.append(b)
        positions.append(mbv.Core.Vector3(0, 0, 0))

    for b in interpolated_bones:
        cur_y = np.linspace(0., bone_lengths[b], n_interp, endpoint=False)
        for i, y in enumerate(cur_y):
            lname = "{0}_{1:02d}".format(b, i)
            landmark_names.append(lname)
            bone_names.append(b)
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


