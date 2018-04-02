import PyMBVCore as core
import PyMBVDecoding as dec
import PyMBVParticleFilter as pf
import PyHandTracker as ht
import numpy as np
np.set_printoptions(precision=1)
import cv2
import copy
import matplotlib.pyplot as plt
import Armature
import AssimpModelInfo
import AssimpLoad
import AssimpImport
import AssimpModelInspect
import RenderingUtils as ru
import AngleTransformations as at
import LandmarksGrabber as lg

def Vector3fStoragetoNumpyArray(mesh_vertices):
    n_vertices = len(mesh_vertices)
    v_xyzw = np.ndarray((4,n_vertices))
    for i,v in enumerate(mesh_vertices):
        v_xyzw[0,i] = v.x
        v_xyzw[1,i] = v.y
        v_xyzw[2,i] = v.z
        v_xyzw[3,i] = 1
    return v_xyzw

#Not required in this version.
def replace_n2p_transform(node,offset):
    if node.type == dec.TransformType.Custom:
        cur_t = AssimpImport.convt_mbv_np(node.defaultValues)
        print 'n2p old:\n',cur_t
        cur_t[1, 3] *= 0.5
        offset_t = np.eye(4,4).astype(np.double)
        offset_t[0][3] = offset[0]
        offset_t[1][3] = offset[1]
        offset_t[2][3] = offset[2]
        offset_t[3][3] = 1
        cur_t = np.dot(offset_t,cur_t)
        print 'n2p new:\n',cur_t
        node.defaultValues = core.DoubleVector(AssimpImport.convt_np_mbv(cur_t))
        #node.meshFile = './models3d/cylinder_zc.obj'
        #print node.name, node.meshFile, ':\n', cur_t

#Replaces the skinned model m2b transform with scaling and translation so that one
#  edge of the cylinder is aligned with the bone origin.
def replace_m2b_transform(node,scale):
    if node.type == dec.TransformType.Custom and len(node.meshFile)>0:
        cur_t = np.eye(4,4).astype(np.double)
        cur_t[0][0] = scale.x
        cur_t[1][1] = scale.y
        cur_t[2][2] = scale.z
        cur_t[1][3] = scale.y
        cur_t[3][3] = 1
        node.defaultValues = core.DoubleVector(AssimpImport.convt_np_mbv(cur_t))
        node.meshFile = './models3d/cylinder_zc.obj'
        #print node.name, node.meshFile, ':\n', cur_t


AssimpLoader = AssimpLoad.AssimpLoader
atree = Armature.ArmatureTree

models_dir = "./models3d"
models_list = ["arm3", "mh_body_male_meta", "hand_skinned", "hand_skinned_meta", "hand_skinned_devices", "simple_arm"]
sel_model = 1
model_name = models_list[sel_model]
#model_filename = "{0}{1}.dae".format(models_dir, models_list[sel_model])
model_xml = "{0}/{1}/{1}.xml".format(models_dir, models_list[sel_model])

model3d_skinned = pf.Model3dMeta.create(model_xml)
model3d_skinned.parts.genBonesMap()
bones_map = model3d_skinned.parts.bones_map
# model_parts = model3d.parts
# model_parts.loadPartsMap(model_xml)
# model_parts.loadBonesMap(model_xml)


assimp_loader = AssimpLoader()
assimp_loader.load(model3d_skinned)

mbv_rootnode = assimp_loader.mbv_rootnode
mbv_bones_rootnode = assimp_loader.mbv_bones_rootnode
mesh_loader = assimp_loader.mesh_loader
n_dims = model3d_skinned.n_dims
n_bones = model3d_skinned.n_bones

print 'Loaded model from <', model_xml, '>'
print 'Bones num:', n_bones
print 'Dim num:', n_dims

mmanager = core.MeshManager()
mmanager.registerLoader(mesh_loader)
ticket = mmanager.loadMesh(model3d_skinned.mesh_filename)
mesh = mmanager.getMesh(ticket)
vert_pos_range = mesh.getVertexPositions()
vert_pos_storage = core.Vector3fStorage()
for p in vert_pos_range:
    vert_pos_storage.append(p)
vert_bones = mesh.getVertexFeature('bone')



#atree.procNodes(mbv_rootnode,replace_m2b_transform)
nodenames_list = []
atree.getNodeList(mbv_rootnode,nodenames_list)

all_vertices = Vector3fStoragetoNumpyArray(vert_pos_storage)
rng = np.max(all_vertices,1) - np.min(all_vertices,1)
print np.max(all_vertices,1), np.min(all_vertices,1)
print 'range all:',rng, len(vert_pos_storage)


for bt in bones_map:
    n2p = bt.key()
    m2b = bt.key() + '_m2b'
    i = bt.data()
    print i, m2b, n2p

    cur_vidx = []
    cur_vertices = core.Vector3fStorage()
    for vidx,vbidx in enumerate(vert_bones):
        if vbidx.x == i:
            cur_vidx.append(vidx)
            cur_vertices.append(vert_pos_storage[vidx])

    node_m2b = atree.getNode(mbv_rootnode,m2b)
    node_n2p = atree.getNode(mbv_rootnode,n2p)
    bones_node = atree.getNode(mbv_bones_rootnode,n2p)
    if len(bones_node.children) == 1:
        n2p_child = AssimpImport.convt_mbv_np(bones_node.children[0].defaultValues)
        cur_bone_length = n2p_child[1,3]
    else: cur_bone_length = -1
    if len(cur_vertices) >0:
        m2b_tr = AssimpImport.convt_mbv_np(node_m2b.defaultValues)
        vertices_pos = np.dot(m2b_tr,Vector3fStoragetoNumpyArray(cur_vertices))

        mx = np.max(vertices_pos,1)
        mn = np.min(vertices_pos,1)
        rng = mx - mn
        print n2p, 'offset n2p/vertices:', cur_bone_length, ' --- ', rng[1]
        if cur_bone_length > 0: rng[1] = cur_bone_length
        scale = core.Vector3(0.5*rng[0], 0.5*rng[1],0.5*rng[2])
        print 'l:',rng[1],'w:',rng[0],'w:',rng[2],'vertices:',len(cur_vertices), 'scale:', scale.data[:,0]
    else:
        if cur_bone_length > 0:
            scale = core.Vector3(1,0.5*cur_bone_length,1)
        else:
            scale = core.Vector3(1,1,1)

    replace_m2b_transform(node_m2b,scale)
    #replace_n2p_transform(node_n2p,[0,0.5*rng[1],0])
    #AssimpModelInspect.plot_vertices(vertices_pos)

model3d_prim = pf.Model3dMetaPrimitives(model3d_skinned)

model3d_prim.model_name = model3d_skinned.model_name + "_prim"
#model3d_prim.model_path = model3d_skinned.model_path + "_prim"
model3d_prim.save()
pf.SaveTransformNode(model3d_prim.transform_node_filename, mbv_rootnode)






