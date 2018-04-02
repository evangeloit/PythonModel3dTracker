import PyMBVCore as core
import PyMBVDecoding as dec
import PyMBVAcquisition
import PyMBVParticleFilter as pf
import PyMBVRendering as ren
import PyMBVLibraries as lib
import matplotlib.pyplot as plt
import RenderingUtils as ru
import numpy as np
import cv2
import copy
import time

tile_size = [32,32]
tile_dims = [64,64]
sel_part = 'little'
model3d_xml = 'models3d_samples/hand_std/hand_std.xml'
model3d = pf.Model3dMeta.create(model3d_xml)

mmanager = core.MeshManager()
decoder = model3d.createDecoder()
model3d.setupMeshManager(mmanager)
decoder.loadMeshTickets(mmanager)


model_parts = model3d.parts
model_parts.load(model3d_xml)
model_parts.genPrimitivesMap(decoder)
iir = model_parts.getInstanceIssueIDsRange()
print iir
print 'Iss ids:',iir.iss_ids
print 'Min inst ids:',iir.min_inst_ids
print 'Max inst ids:',iir.max_inst_ids

print 'Primitives Map:', model_parts.primitives_map
print 'Parts Map:', model_parts.parts_map

#Testing ModelParts visibility.
decoding = decoder.quickDecode(model3d.default_state)
cam_meta = lib.RGBDAcquisitionSimulation.getDefaultCalibration()
cam_frust = cam_meta.camera
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
    state[2] += 3 * float(i)
    state[3] += 0.1
    for j in range(7,model3d.n_dims,1):
        state[j] += 0.1

    multi_state = core.ParamVectors()
    for h in range(tile_dims[0]*tile_dims[1]):
        multi_state.append(state)
    decoding = decoder.quickDecode(multi_state)


    #Rendering model
    #ru.render(renderer, mmanager, decoding,  cam_meta,tile_dims, renderer.Culling.CullFront, model3d.n_bones)
    ru.render_tiles(renderer, mmanager, decoding, cam_meta, tile_dims, renderer.Culling.CullFront, model3d.n_bones,tile_size)
    positions, normals, colors, issue, instance, vertex_id = ru.genMaps(renderer)
    labels = [issue.astype(np.ushort), instance.astype(np.ushort)]




    #Generating Mask Using TiledRendering/Thrust on gpu.
    WriteFlag = ren.Renderer.WriteFlag
    Channel = ren.MapExposer.Channel
    WriteFlagExtra = ren.RendererOGLBase.WriteFlagExtra
    renderer.beginMapResources(WriteFlagExtra.WriteAll)
    iss = renderer.getMap(WriteFlag.WriteID, Channel.X)
    ins = renderer.getMap(WriteFlag.WriteID, Channel.Y)
    labels_tl = ren.TiledRenderingVector()
    labels_tl.append(iss)
    labels_tl.append(ins)
    mask_mat_cuda_tl = model_parts.genPartMask(sel_part,labels_tl)
    mask_mat_cuda = ren.DownloadToCvMat(mask_mat_cuda_tl)
    mask_mat_cuda[mask_mat_cuda>0] = np.iinfo(mask_mat_cuda.dtype).max
    #print np.min(mask_mat_cuda), np.max(mask_mat_cuda), np.sum(mask_mat_cuda)


    #Generating Mask Using cv::Mat on cpu.
    mask_mat_cpu = model_parts.genPartMask(sel_part,labels)

    t1 = time.clock()
    cur_vis_ratio_cpu = model_parts_visibility.calcVisibilityRatio(labels,sel_part)
    t2 = time.clock()
    cur_vis_ratio_cuda = model_parts_visibility.calcVisibilityRatioGPU(labels_tl, sel_part)
    t3 = time.clock()
    renderer.endMapResources()

    #print 'Visibility ratio ', sel_part, ' cpu:', cur_vis_ratio_cpu, ', cuda:',cur_vis_ratio_cuda.data
    print 'Calc time cpu-cuda:', t2-t1, t3-t2
    normals_disp = normals[:, :, 0:3].astype(np.float32) * 0.5 + 0.5

    #Visualizing
    cv2.imshow("normals",normals_disp)
    #cv2.imshow("mask_mat_cpu",mask_mat_cpu)
    #cv2.imshow("mask_mat_cuda",mask_mat_cuda)
    cv2.waitKey(0)



