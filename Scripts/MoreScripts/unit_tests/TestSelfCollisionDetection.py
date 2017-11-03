import PyMBVRendering as ren
import numpy as np

import PyMBVCore as core
import PyMBVDecoding as dec
import PyMBVLibraries as lib
import PyMBVOpenMesh as mbvom
import PyMBVParticleFilter as pf
import PyMBVPhysics as phys

np.set_printoptions(precision=3)
import cv2
import BlenderMBVLib.RenderingUtils as ru
import PythonModelTracker.BulletCollisionUtils as bcu
import os
os.chdir(os.environ['bmbv']+"/Scripts/")


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
params_step = [-0.05,0.05]
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
openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
decoder = dec.GenericDecoder()
print('Loading model from <', model_3d.model_collada, '>')
decoder.loadFromFile(model_3d.model_collada,False)
decoder.loadMeshTickets(mmanager)
meshes = core.MeshTicketList()
mmanager.enumerateMeshes(meshes)
for m in meshes:
    print(mmanager.getMeshFilename(m))

n_dims = decoder.get_dimensions()
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


#Creating Bullet World
bw = phys.BulletWorld()
bullet_bodies = []

for i, s in enumerate(selected_shapes):
    bullet_bodies.append([])
    for j in range(selected_shapes_count[i]):
        if 'sphere' in s:
            bullet_bodies[i].append( phys.BulletBody(phys.BulletShapeImplementation(sphere_shape)))
        if 'cylinder' in s:
            bullet_bodies[i].append( phys.BulletBody(phys.BulletShapeImplementation(cyl_shape)))
        bw.addBody(bullet_bodies[i][j])



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


    #Decoding, rendering and getting the maps from renderer.
    decoding = decoder.quickDecode(multi_pos)
    ru.render(renderer, mmanager, decoding,  cam_meta,tile_dims, renderer.Culling.CullFront, 0)
    positions, normals, colors, issue, instance, vertex_ids = ru.genMaps(renderer)
    viz_img = normals[:, :, 0:3].astype(np.float32) * 0.5 + 0.5
    covec = core.DoubleVector()
    codet.queryCollisions2(decoding,covec)
    if len(covec) > 0:
        print('Penetration depth CollisionDetection:',covec, 'len:', len(covec))


    for obj in decoding:
        for ss_idx,ss in enumerate(selected_shapes):
            if ss in mmanager.getMeshFilename(obj.key()):
                for bb_idx,bb in enumerate(bullet_bodies[ss_idx]):
                    dec_p,dec_s,dec_q = core.DecomposeMatrix(obj.data().matrices[bb_idx])
                    bb.position = dec_p
                    bb.orientation = core.Quaternion(-dec_q.w,dec_q.x,dec_q.y,dec_q.z)
                    bb.scale = dec_s
                    #print('scale:', dec_s.data.T, 'pos:', dec_p.data.T, 'rot:', dec_q.data.T)

    penetration_depth = 0
    for ss_idx1,ss1 in enumerate(bullet_bodies):
        for bb_idx1,bb1 in enumerate(bullet_bodies[ss_idx1]):
            for ss_idx2,ss2 in enumerate(bullet_bodies):
                for bb_idx2,bb2 in enumerate(bullet_bodies[ss_idx2]):
                    if (bb_idx1 != bb_idx2) or (ss_idx1 != ss_idx2):
                        #print('cur_penetration:',bw.computePenetration(bb1,bb2))
                        penetration_depth += bw.computePenetration(bb1,bb2)
    #if len(covec) > 0:
    #    print('Penetration depth CollisionDetection, BulletWorld, ratio:', \
    #        covec[0], penetration_depth, penetration_depth/(covec[0]+0.000001))
    bdd = bcu.BulletDebugDraw()
    bw.debugDraw(bdd)
    #print('Min-max 3d points:',np.min(bdd.points3da,axis=0), np.max(bdd.points3db,axis=0))
    bcu.ProjectDrawLinesOpencv(bdd.points3da,bdd.points3db,cam_meta,viz_img)

    #Displaying the normals.
    cv2.imshow("viz",viz_img)
    cv2.waitKey(wait_time)

core.CachedAllocatorStorage.clear()
