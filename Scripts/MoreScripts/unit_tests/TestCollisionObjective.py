import PyMBVRendering as ren
import numpy as np
# import CustomRenderingObjectiveImplementation as croi
import os

import PyMBVCore as core
import PyMBVLibraries as lib
import PyMBVOpenMesh as mbvom
import PyMBVParticleFilter as pf
import PyMBVPhysics as phys
import PyModel3dTracker as mt

os.chdir(os.environ['bmbv']+"/Scripts/")

np.set_printoptions(precision=3)
import cv2
import BlenderMBVLib.RenderingUtils as ru

"""
Renders a primitives based model and checks for collisions.
 WARNING: The scale is not estimated correctly.
"""
wait_time = 0
steps = 10
tile_dims = [1, 1]
tile_size = [1280,960]
euler_mult = [0.,0.3,0.]
params_ids = [10, 20]
params_step = [0.05,-0.05]
#Model path.
sel_model = ["human_ext_collisions","hand_std_collisions"][0]
if sel_model == "human_ext_collisions":
    model_xml = Paths.models + "/human_ext/{}.xml".format(sel_model)
    selected_shapes = ['sphere_collision','cylinder_collision']
    selected_shapes_count = [5,2]
elif sel_model == "hand_std_collisions":
    model_xml = Paths.models + "/hand_std/{}.xml".format(sel_model)
    selected_shapes = ['cylinder_low1']
    selected_shapes_count = [12]
else:
    quit()


#Initializing decoder and mesh manager.
model_3d = pf.Model3dMeta.create(model_xml)
mmanager = core.MeshManager()

dof = mt.Model3dObjectiveFrameworkDecoding(mmanager)
dof.decoder = model_3d.createDecoder()

openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
# decoder = dec.GenericDecoder()
# print('Loading model from <', model_3d.model_collada, '>')
# decoder.loadFromFile(model_3d.model_collada,False)
dof.decoder.loadMeshTickets(mmanager)
meshes = core.MeshTicketList()
mmanager.enumerateMeshes(meshes)
for m in meshes:
    print(mmanager.getMeshFilename(m))

n_dims = dof.decoder.get_dimensions()
print('Dim num:',n_dims)

# Setting camera
cam_meta = lib.RGBDAcquisitionSimulation.getDefaultCalibration()
cam_meta.width = tile_size[0]
cam_meta.height = tile_size[1]

cam_frust = cam_meta.camera
print(cam_frust.zNear,cam_frust.zFar, cam_frust.position, cam_frust.orientation, cam_frust.nearPlaneSize)
view_mat = cam_frust.Graphics_getViewTransform()
proj_mat = cam_frust.Graphics_getProjectionTransform()
print('view mat:\n', view_mat.data)
print('proj mat:\n', proj_mat.data)
renderer = ren.RendererOGLCudaExposed.get()

#Initializing collision detector.
codet = lib.CollisionDetection(mmanager)
cyl_shape = phys.CylinderShapeZ()
cyl_shape.scale = core.Vector3(1,1,1)
cyl_shape.length = 1.8
cyl_shape.radius = 1
sphere_shape = phys.SphereShape()
sphere_shape.radius = 1
sphere_shape.scale = core.Vector3(1,1,1)

for s in selected_shapes:
    for m in meshes:
        mesh_filename = mmanager.getMeshFilename(m)
        if s in mesh_filename:
            if 'sphere' in mesh_filename:
                print('Registering ', mesh_filename)
                codet.registerShape(mesh_filename,sphere_shape)
            if 'cylinder' in mesh_filename:
                print('Registering ', mesh_filename)
                codet.registerShape(mesh_filename,cyl_shape)

dois = mt.DecodingObjectives()
doi = mt.CollisionObjective.create(codet)
#doi = mt.LandmarksDistObjective()
#doi = croi.MyCustomDecodingObjectiven()
dois.append(doi)
dof.appendDecodingObjectivesGroup(dois)

#Init Parameter Vector.
init_pos = model_3d.default_state
init_pos[2] = 3500

#Rendering Loop.
for i in range(steps):
    # Manually changing the parameter vector.
    for p,s in zip(params_ids,params_step):
        init_pos[p] += s
    #Setting the rotation using euler angles (for convenience) and then transforming to quaternion.
    # euler_angles = [3.14+euler_mult[0]*i,euler_mult[1]*i,euler_mult[2]*i]
    # rot_q = at.quaternion_from_euler(euler_angles[0],euler_angles[1],euler_angles[2])
    # init_pos[3] = rot_q[0]
    # init_pos[4] = rot_q[1]
    # init_pos[5] = rot_q[2]
    # init_pos[6] = rot_q[3]

    #print('init_pos', init_pos.data[:,0])

    #Rendering multiple hypotheses in tiles.
    multi_pos = core.ParamVectors([init_pos]*(tile_dims[0]*tile_dims[1]) )

    res = dof.evaluate(multi_pos,0)
    print("CollisionObjectiveFramework res vec: ", res)
    #Decoding, rendering and getting the maps from renderer.
    decoding = dof.decoder.quickDecode(multi_pos)
    res = doi.evaluate(decoding)
    print("CollisionObjective res vec: ", res)
    ru.render(renderer, mmanager, decoding,  cam_meta,tile_dims, renderer.Culling.CullFront, 0)
    positions, normals, colors, issue, instance, vertexn_ids = ru.genMaps(renderer)
    viz_img = normals[:, :, 0:3].astype(np.float32) * 0.5 + 0.5
    covec = core.DoubleVector()
    codet.queryCollisions2(decoding,covec)
    if len(covec) > 0:
        print('Penetration depth CollisionDetection:',covec.data[:,0], 'len:', len(covec))


    #Displaying the normals.n
    cv2.imshow("viz",viz_img)
    cv2.waitKey(wait_time)

core.CachedAllocatorStorage.clear()
