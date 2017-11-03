import PyMBVCore as core
import PyMBVOpenMesh as mbvom
import PyMBVDecoding as dec
import PyMBVRendering as ren
import PyMBVAcquisition as acq
import PyMBVLibraries as lib
import numpy as np
#import AngleTransformations as at

np.set_printoptions(precision=1)
import cv2
import copy
import matplotlib.pyplot as plt
import RenderingUtils as ru


"""
Renders a primitives based model and saves the rendered maps (images).
"""

core.ScopeReportTimer.setReportDepth(-1)
#Model path.
model_xml = "media/hand_right_low_RH.xml"


#Initializing decoder and mesh manager.
mmanager = core.MeshManager()
openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
decoder = dec.GenericDecoder()
print 'Loading model from <', model_xml, '>'
decoder.loadFromFile(model_xml,False)
decoder.loadMeshTickets(mmanager)
n_dims = decoder.get_dimensions()
print 'Dim num:',n_dims

# Setting camera
cam_meta = lib.RGBDAcquisitionSimulation.getDefaultCalibration()
cam_frust = cam_meta.camera
view_mat = cam_frust.Graphics_getViewTransform()
proj_mat = cam_frust.Graphics_getProjectionTransform()
print 'view mat:\n', view_mat.data
print 'proj mat:\n', proj_mat.data
renderer = ren.RendererOGLCudaExposed.get()
rendering_helper = lib.RenderingHelper()
rendering_helper.decoder = decoder
rendering_helper.renderer = renderer

#Init Parameter Vector.
init_pos = core.ParamVector([0,0,1000,0.,0.,0.,1.,0.,1.57,0.,0,0.,3.14,0.,0.,0.,1.57,0.,0,0,1.57,0,0,0.,1.57,0.,0])


#Rendering Loop.
plt.figure(figsize=(15, 15))
steps = 10
for i in range(steps):
    # Manually changing the parameter vector.
    #Setting the global position.
    init_pos[0] = 10 * float(i)
    init_pos[1] = 10 * float(i)
    init_pos[2] = 1000
    #Setting the rotation using euler angles (for convenience) and then transforming to quaternion.
    euler_angles = [0,i*3.14/(3*float(steps)),3.14]
    # rot_q = at.quaternion_from_euler(euler_angles[0],euler_angles[1],euler_angles[2])
    # init_pos[3] = rot_q[1]
    # init_pos[4] = rot_q[2]
    # init_pos[5] = rot_q[3]
    # init_pos[6] = rot_q[0]

    #Rendering multiple hypotheses in tiles.
    multi_pos = core.ParamVectors([init_pos] )
    rows = 1
    cols = 1

    #Decoding, rendering and getting the maps from renderer.
    decoding = decoder.quickDecode(multi_pos)
    ru.render(renderer, mmanager, decoding, cam_meta, [rows, cols], renderer.Culling.CullFront, 0)
    #renderer.uploadViewMatrices(core.MatrixVector([cam_meta.camera.Graphics_getViewTransform()]))
    #renderer.uploadProjectionMatrices(core.MatrixVector([cam_meta.camera.Graphics_getProjectionTransform()]))
    #rendering_helper.render(ren.RendererOGLBase.WriteFlagExtra.WriteAll,rows,cols,640,480,multi_pos,mmanager)
    positions, normals, colors, issue, instance, V = ru.genMaps(renderer)

    #Displaying the foreground.
    # plt.imshow(issue > 0)

    #Displaying the normals.
    plt.imshow(normals * 0.5 + 0.5)

    plt.waitforbuttonpress()

    #Saving the image to the disk.
    # from scipy.misc import imsave
    # imsave('body.png', normals * 0.5 + 0.5)

plt.close()
