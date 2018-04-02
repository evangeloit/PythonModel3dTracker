import PyMBVCore as core
import PyMBVDecoding as dec
import PyMBVRendering as ren
import PyMBVAcquisition as acq
import PyMBVLibraries as lib
import numpy as np
import AngleTransformations as at
import PyHandTracker
import PyMBVPhysics as phys
import PySynergyTrackerRGBD as st
import PyMBVParticleFilter as pf

np.set_printoptions(precision=3)
import cv2
import copy
import RenderingUtils as ru
import BulletCollisionUtils as bcu


"""
Renders a skinned model and checks for collisions.
 WARNING: The scale is not estimated correctly.
"""
wait_time = 0
steps = 24
tile_dims = [1, 1]
tile_size = [1240,800]
euler_mult = [0.,0.25,0.]
params_ids = []
params_step = [-0.03,+0.03]
#Model path.
sel_model = 1
if sel_model == 0:
    model_xml = "modeqls3d/human_ext/human_ext_collisions.xml"
    selected_shapes = ['sphere_head','sphere_elbow','sphere_wrist','cylinder_body0','cylinder_body1']
    selected_shapes_count = [1,2,2,1,1]
elif sel_model == 1:
    model_xml = "models3d/hand_skinned/hand_skinned_prim.xml"
    selected_shapes = ['cylinder_zc']
    selected_shapes_count = [20]

#Initializing decoder and mesh manager.
model_3d = pf.Model3dMeta.create(model_xml)
mmanager = core.MeshManager()
decoder = model_3d.createDecoder()
model_3d.setupMeshManager(mmanager)
print 'Dim num:',model_3d.n_dims, 'Bones num: ', model_3d.n_bones


decoder.loadMeshTickets(mmanager)
meshes = core.MeshTicketList()
mmanager.enumerateMeshes(meshes)
for m in meshes:
    print mmanager.getMeshFilename(m)



# Setting camera
cam_meta = lib.RGBDAcquisitionSimulation.getDefaultCalibration()
cam_meta.width = tile_size[0]
cam_meta.height = tile_size[1]


cam_frust = cam_meta.camera
cam_frust.nearPlaneSize = core.Vector2(-cam_frust.nearPlaneSize.x,-cam_frust.nearPlaneSize.y)
print cam_frust.zNear,cam_frust.zFar, cam_frust.position, cam_frust.orientation, cam_frust.nearPlaneSize
view_mat = cam_frust.Graphics_getViewTransform()
proj_mat = cam_frust.Graphics_getProjectionTransform()
print 'view mat:\n', view_mat.data
print 'proj mat:\n', proj_mat.data
cam_meta.camera = cam_frust
renderer = ren.RendererOGLCudaExposed.get()

#Initializing collision detector.
codet = st.CollisionDetection(mmanager)
cyl_shape = phys.CylinderShape()
cyl_shape.scale = core.Vector3(1,1,1)
cyl_shape.length = 1
cyl_shape.radius = 1
sphere_shape = phys.SphereShape()
sphere_shape.radius = 1
sphere_shape.scale = core.Vector3(1,1,1)

for s in selected_shapes:
    for m in meshes:
        if s in mmanager.getMeshFilename(m):
            if 'sphere' in s:
                print 'Registering sphere ' , mmanager.getMeshFilename(m)
                codet.registerShape(mmanager.getMeshFilename(m),sphere_shape)
            else:
                print 'Registering cylinder ' , mmanager.getMeshFilename(m)
                codet.registerShape(mmanager.getMeshFilename(m),cyl_shape)

#Creating Bullet World
bw = phys.BulletWorld()
bullet_bodies = []

for i, s in enumerate(selected_shapes):
    bullet_bodies.append([])
    for j in range(selected_shapes_count[i]):
        if 'sphere' in s:
            bullet_bodies[i].append( phys.BulletBody(phys.BulletShapeImplementation(sphere_shape)))
        else:
            bullet_bodies[i].append( phys.BulletBody(phys.BulletShapeImplementation(cyl_shape)))
        bw.addBody(bullet_bodies[i][j])



#Init Parameter Vector.
init_pos = model_3d.default_state#core.ParamVector([0,0,3000,0,0.164012,0,0.986458,0,0.0002,0,0,0,0,0.706329,0,0.707883,0,0,0,1,0,0,0,0.706329,0,0.707883,0,0,0,1,0,0,0.0279562,0.0279781,0,0.999218,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,288.6,289.6,91.1,135,259.2,260,322.4,322.6,200.04,59.98,59.98,55,50.02,50.02,50.02,200.04,350,110.04,200])

#Rendering Loop.
for i in range(steps):
    # Manually changing the parameter vector.
    for p,s in zip(params_ids,params_step):
        init_pos[p] += s
    #Setting the rotation using euler angles (for convenience) and then transforming to quaternion.
    euler_angles = [euler_mult[0]*i,euler_mult[1]*i,euler_mult[2]*i]
    rot_q = at.quaternion_from_euler(euler_angles[0],euler_angles[1],euler_angles[2])
    init_pos[3] = rot_q[0]
    init_pos[4] = rot_q[1]
    init_pos[5] = rot_q[2]
    init_pos[6] = rot_q[3]



    #Rendering multiple hypotheses in tiles.
    multi_pos = core.ParamVectors([init_pos]*(tile_dims[0]*tile_dims[1]) )


    #Decoding, rendering and getting the maps from renderer.
    decoding = decoder.quickDecode(multi_pos)
    ru.render(renderer, mmanager, decoding,  cam_meta,tile_dims, renderer.Culling.CullFront, model_3d.n_bones)
    positions, normals, colors, issue, instance = ru.genMaps(renderer)
    viz_img = normals[:, :, 0:3].astype(np.float32) * 0.5 + 0.5
    covec = core.DoubleVector()
    codet.queryCollisions(decoding,covec)
    if len(covec) > 0:
        print 'Max penetration depth CollisionDetection:',max(covec.data), 'len:', len(covec)


    #body
    for obj in decoding:
        for ss_idx,ss in enumerate(selected_shapes):
            if ss in mmanager.getMeshFilename(obj.key()):
                print 'Decoding', ss
                for bb_idx,bb in enumerate(bullet_bodies[ss_idx]):
                    dec_p,dec_s,dec_q = core.DecomposeMatrix(obj.data().matrices[bb_idx])
                    bb.position = dec_p
                    print 'pos: ',dec_p.data[:,0],'  nscl: ',dec_s.data[:,0]
                    bb.orientation = core.Quaternion(dec_q.w,dec_q.x,dec_q.y,-dec_q.z)
                    #bb.orientation = core.Quaternion(rot_q[0],rot_q[1],rot_q[2],rot_q[3])
                    bb.scale = dec_s
                    #if bb_idx < len(bullet_bodies)-1:
                    #    print 'Penetration depth BulletWorld:',bw.computePenetration(bb,bullet_bodies[bb_idx+1])
    bdd = bcu.BulletDebugDraw()
    bw.debugDraw(bdd)
    #print 'Min-max 3d points:',np.min(bdd.points3da,axis=0), np.max(bdd.points3db,axis=0)
    bcu.ProjectDrawLinesOpencv(bdd.points3da,bdd.points3db,cam_meta,viz_img)

    #Displaying the normals.
    cv2.imshow("viz",viz_img)
    cv2.waitKey(wait_time)

