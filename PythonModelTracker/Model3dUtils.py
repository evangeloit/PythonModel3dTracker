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




