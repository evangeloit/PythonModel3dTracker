import PyMBVCore as core
import PyMBVDecoding as dec
import PyMBVRendering as ren
import PyMBVAcquisition as acq
import PyMBVLibraries as lib
import PyMBVParticleFilter as pf
import GenerateStateTransforms as gst
import AssimpLoad

import numpy as np
np.set_printoptions(precision=1)

import cv2
import copy
import matplotlib.pyplot as plt
import RenderingUtils as ru
import AngleTransformations as at

model_xml = "models3d/hand_skinned/hand_skinned.xml"
model3d_dst = pf.Model3dMeta.create(model_xml)
assimp_loader = AssimpLoad.AssimpLoader()
assimp_loader.load(model_xml)

mbv_rootnode = assimp_loader.mbv_rootnode
mesh_loader = assimp_loader.mesh_loader
n_bones = model3d_dst.n_bones
mmanager = core.MeshManager()
mmanager.registerLoader(mesh_loader)
decoder = dec.GenericDecoderGPU(mbv_rootnode)
print 'Loading model from <', model3d_dst.model_xml, '>'
#decoder.loadFromFile(model3d_dst.model_xml,False)
decoder.loadMeshTickets(mmanager)


dim_groups = [[7,9,10,11,13,14,15,17,18,19,21,22],[8,12,16,20],[23,25,26]]
model3d_src, stransformer = gst.GenerateStateTransform_GroupDims(model3d_dst,dim_groups)
src_state = model3d_src.default_state
stdecoder = dec.StateTransformDecoder(decoder)
stdecoder.state_transformer = stransformer

src_indices = []
dst_indices = []
for d in range(model3d_dst.n_dims):
    cur_st = stransformer.getStateTransform(d)
    src_indices.append(cur_st.params.idx_src)
    dst_indices.append(cur_st.params.idx_dst)

print "n_dims_src:", model3d_src.n_dims
print "model3d_dst default state:\n", model3d_dst.default_state.data.transpose()
print "model3d_dst low_bounds:\n", model3d_dst.low_bounds.data.transpose()
print "model3d_dst high_bounds:\n", model3d_dst.high_bounds.data.transpose()
print "src indices:\n", src_indices
print "dst indices:\n", dst_indices
print "model3d_src default state:\n", model3d_src.default_state.data.transpose()
print "model3d_src low_bounds:\n", model3d_src.low_bounds.data.transpose()
print "model3d_src high_bounds:\n", model3d_src.high_bounds.data.transpose()




# Setting camera
cam_meta = lib.RGBDAcquisitionSimulation.getDefaultCalibration()
cam_frust = cam_meta.camera
# cam_frust = core.CameraFrustum()
# cam_frust.position = core.Vector3(0,0,0)
# cam_frust.zNear = 100
# cam_frust.zFar = 20000
# cam_frust.nearPlaneSize = core.Vector2(100,100)
view_mat = cam_frust.Graphics_getViewTransform()
proj_mat = cam_frust.Graphics_getProjectionTransform()
print 'view mat:\n', view_mat.data
print 'proj mat:\n', proj_mat.data
renderer = ren.RendererOGLCudaExposed.get()

plt.figure(figsize=(15, 15))
steps = 10
for i in range(steps):
    # Setting param Vector

    # state[0] += 10
    # state[1] += 10
    src_state[2] = 500
    rot_q = at.quaternion_from_euler(0,0,0)
    src_state[3] = rot_q[1]
    src_state[4] = rot_q[2]
    src_state[5] = rot_q[3]
    src_state[6] = rot_q[0]
    src_state[7] += 0.1

    multi_pos = core.ParamVectors([src_state] * 4)
    rows = 2
    cols = 2

    decoding = stdecoder.quickDecode(multi_pos)
    ru.render(renderer, mmanager, decoding,  view_mat, proj_mat,rows, cols, [640, 480],
              renderer.Culling.CullFront, n_bones)
    positions, normals, colors, issue, instance = ru.genMaps(renderer)

    plt.imshow(normals[:, :, 0:3].astype(np.float32) * 0.5 + 0.5)
    plt.waitforbuttonpress(0.1)
    # from scipy.misc import imsave
    # imsave('body.png', normals * 0.5 + 0.5)

plt.close()
