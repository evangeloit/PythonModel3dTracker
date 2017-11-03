import PyMBVRendering as ren
import matplotlib.pyplot as plt
import numpy as np

import BlenderMBVLib.RenderingUtils as ru
import PyMBVCore as core
import PyMBVLibraries as lib
import PyMBVParticleFilter as pf
import cv2

sel_part = 'little'
model3d_xml = Paths.model3d_dict['hand_skinned']['path']
model3d = pf.Model3dMeta.create(str(model3d_xml))

mmanager = core.MeshManager()
model3d.setupMeshManager(mmanager)

decoder = model3d.createDecoder()
decoder.loadMeshTickets(mmanager)


model_parts = model3d.parts
model_parts.load(str(model3d_xml))
model_parts.genBonesMap()



#Testing ModelParts visibility.
decoding = decoder.quickDecode(model3d.default_state)
mesh_tkt = decoding.keys()[0]
model_parts.mesh = mmanager.getMesh(mesh_tkt)
cam_meta = lib.RGBDAcquisitionSimulation.getDefaultCalibration()
cam_frust = cam_meta.camera
view_mat = cam_frust.Graphics_getViewTransform()
proj_mat = cam_frust.Graphics_getProjectionTransform()
renderer = ren.RendererOGLCudaExposed.get()

#Init figure
plt.figure(figsize=(15, 15))
state = model3d.default_state
default_decoding = decoder.quickDecode(state)
model_parts_visibility = pf.ModelPartsVisibility(model_parts)

#Viz loop
for i in range(0,10,1):
    # Setting param Vector
    state[0] += 3 * float(i)
    state[1] += 3 * float(i)
    state[2] = 1000#10 * float(i)
    state[3] += 0.1
    for j in range(7,model3d.n_dims,1):
        state[j] += 0.1


    #decoding
    decoding = decoder.quickDecode(state)


    #Rendering model
    tile_dims = [1, 1]
    ru.render(renderer, mmanager, decoding,  cam_meta,tile_dims, renderer.Culling.CullFront, model3d.n_bones)
    positions, normals, colors, issue, instance, vertex_id = ru.genMaps(renderer)
    #bones=ru.genBoneIDs(renderer, decoding ,mmanager)
    labels = [issue.astype(np.ushort), vertex_id.astype(np.int32)]



    # Generating Mask Using TiledRendering/Thrust on gpu.
    WriteFlag = ren.Renderer.WriteFlag
    Channel = ren.MapExposer.Channel
    WriteFlagExtra = ren.RendererOGLBase.WriteFlagExtra
    renderer.beginMapResources(WriteFlagExtra.WriteAll)
    iss = renderer.getMap(WriteFlag.WriteID)
    #ins = renderer.getMap(WriteFlag.WriteID, Channel.Y)
    vid = renderer.getMap(WriteFlagExtra.WriteVertexIDs)
    labels_tl = ren.TiledRenderingVector()
    labels_tl.append(iss)
    labels_tl.append(vid)
    mask_mat_cuda_tl = model_parts.genPartMask(sel_part, labels_tl)
    mask_mat_cuda = ren.DownloadToCvMat(mask_mat_cuda_tl)
    mask_mat_cuda[mask_mat_cuda > 0] = np.iinfo(mask_mat_cuda.dtype).max

    # Generating Mask Using cv::Mat on cpu.
    mask_mat_cpu = model_parts.genPartMask(sel_part, labels)


    cur_vis_ratio_cpu = model_parts_visibility.calcVisibilityRatio(labels, sel_part)

    cur_vis_ratio_cuda = model_parts_visibility.calcVisibilityRatioGPU(labels_tl, sel_part)

    renderer.endMapResources()

#    print "Unique bone ids in bones map:", np.unique(model3d.bones), 'No of non zero mask elements:',np.sum(mask_mat)
    print('Visibility ratio ', sel_part, ' cpu:', cur_vis_ratio_cpu, ', cuda:',cur_vis_ratio_cuda.data)


    normals_disp = normals[:, :, 0:3].astype(np.float32) * 0.5 + 0.5

    #Visualizing
    # Visualizing
    cv2.imshow("normals", normals_disp)
    cv2.imshow("mask_mat_cpu", mask_mat_cpu)
    cv2.imshow("mask_mat_cuda", mask_mat_cuda)
    cv2.waitKey(0)


plt.close()
