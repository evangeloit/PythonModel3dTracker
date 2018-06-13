import PyMBVCore as core
import PyMBVOpenMesh as mbvom
import PyMBVDecoding as dec
import PyMBVRendering as ren
import PyMBVAcquisition as acq
import PyMBVLibraries as lib
import numpy as np
import PythonModel3dTracker.PythonModelTracker.AngleTransformations as at
import PyHandTracker
import PyMBVPhysics as phys
import PySynergyTrackerRGBD as st

np.set_printoptions(precision=1)
import cv2
import copy
import BlenderMBVLib.RenderingUtils as ru
import PythonModel3dTracker.PythonModelTracker.BulletCollisionUtils as bcu

"""
Renders a primitives based model and checks for collisions.
"""

# Parameters
selected_shape = "sphere"
rendering_size = [2,2]


# Fixed values, not to be changed.
n_obj = 2   # only two objects are supported
n_dims_single = 7 # state dimensions for rigid bodies
assert(n_obj == 2)

if selected_shape == "cyl":
    #Model path.
    model_xml = "models3d_samples/hand_std/sphere_low.obj"
    shape = phys.SphereShape()
    shape.radius = 1
    shape.scale = core.Vector3(100,100,100)
    #dec_scale = core.Vector3(shape.radius,shape.radius,shape.radius)
    #dec_scale = core.Vector3(100,100,100)
else:
    model_xml = "models3d_samples/hand_std/cylinder_low.obj"
    shape = phys.CylinderShapeZ()
    # shape = phys.MeshShape()
    # shape.useConvex = True
    # mm = core.MeshManager()
    # shape.mesh = mm.getMesh(mm.loadMesh(model_xml))
    shape.scale = core.Vector3(50,50,50)
    shape.length = 2
    shape.radius = 1

#Initializing decoder and mesh manager.
mmanager = core.MeshManager()
openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
rigid_decoders = []
decoders = []
for d in range(n_obj):
    decoder1 = dec.SingleRigidDecoder()
    decoder1.meshFilename = model_xml
    decoder1.loadMeshTickets(mmanager)
    decoder1.enableRotation = True
    decoder1.enableTranslation = True
    decoder1.scale = shape.scale
    dec1 = dec.SlicedDecoder(decoder1)
    dec1.slice = core.UIntVector(range(n_dims_single*d,n_dims_single*(d+1),1))
    rigid_decoders.append(decoder1)
    decoders.append(dec1)
decoder = dec.DecoderCombination(decoders[0],decoders[1])

meshes = core.MeshTicketList()
mmanager.enumerateMeshes(meshes)
for m in meshes:
    print(mmanager.getMeshFilename(m))

n_dims = 2 * decoder1.getDimensionsRequired()
print('Dim num:',n_dims)

# Setting camera
cam_meta = lib.RGBDAcquisitionSimulation.getDefaultCalibration()
cam_frust = cam_meta.camera
renderer = ren.RendererOGLCudaExposed.get()

#Initializing collision detector.
codet = st.CollisionDetection(mmanager)
for m in meshes:
    if 'sphere' in mmanager.getMeshFilename(m):
        print('Registering {0} from mesh {1}.'.format('sphere', mmanager.getMeshFilename(m)))
        codet.registerShape(m,shape)
    if 'cylinder' in mmanager.getMeshFilename(m):
        print('Registering {0} from mesh {1}.'.format('cylinder', mmanager.getMeshFilename(m)))
        codet.registerShape(m,shape)

#Creating Bullet World
bw = phys.BulletWorld()
bullet_bodies = []
for b in range(n_obj):
    bullet_bodies.append(phys.BulletBody(phys.BulletShapeImplementation(shape)))
    bw.addBody(bullet_bodies[b])






#Init Parameter Vector.
state = [0,0,600,0.,0.,0.,1.,
         100,0,600,0.,0.,0.,1.]
n_hyp = rendering_size[0]*rendering_size[1]
#
#Rendering Loop.
steps = 12
for i in range(steps):
    # Manually changing the parameter vector.
    #Setting the global position.
    # state[0] += 3 * float(i)
    # state[1] += 3 * float(i)
    # state[2] += 3 * float(i)
    #Setting the rotation using euler angles (for convenience) and then transforming to quaternion.
    euler_angles = [0+0.1*i,0+0.2*i,0+0.3*i]
    rot_q1 = at.quaternion_from_euler(euler_angles[0],euler_angles[1],euler_angles[2]) # rot_q1(x,y,z,w)
    state[3] = rot_q1[3]
    state[4] = rot_q1[0]
    state[5] = rot_q1[1]
    state[6] = rot_q1[2]
    euler_angles = [0,0,1.6]
    rot_q2 = at.quaternion_from_euler(euler_angles[0],euler_angles[1],euler_angles[2])
    state[10] = rot_q2[3]
    state[11] = rot_q2[0]
    state[12] = rot_q2[1]
    state[13] = rot_q2[2]
    shape.scale = core.Vector3(100+10*i,100+10*i,100+10*i)

    #Decoding, rendering and getting the maps from renderer.
    for d in rigid_decoders: d.scale = shape.scale

    multi_state = core.ParamVectors()
    for i in range(n_hyp): multi_state.append(core.ParamVector(state))
    decoding = decoder.quickDecode(multi_state)
    ru.render(renderer, mmanager, decoding,  cam_meta,rendering_size, renderer.Culling.CullFront, 0)
    positions, normals, colors, issue, instance, vertex_id = ru.genMaps(renderer)

    covec = core.DoubleVector()
    codet.queryCollisions(decoding,covec)
    if len(covec) > 0:
        print('Max penetration depth CollisionDetection:',covec.data.T)

    #Displaying the foreground.
    #plt.imshow(issue > 0)
    viz_img = normals[:, :, 0:3].astype(np.float32) * 0.5 + 0.5


    for d in decoding:
        for j,(m,b) in enumerate(zip(d.data().matrices,bullet_bodies)):
            # b.position, b.scale, b.orientation = core.DecomposeMatrix(m)
            dec_p,dec_s,dec_q = core.DecomposeMatrix(m)
            print('Body ', j, 'quats(x,y,z,w) rot_q1:',rot_q1,'dec_q1:', np.array([dec_q.x, dec_q.y, dec_q.z, dec_q.w]))
            b.position = dec_p
            b.orientation = core.Quaternion(-dec_q.w, dec_q.x, dec_q.y, dec_q.z)  # core.Quaternion(w,x,y,z)
            b.scale = shape.scale


    bdd = bcu.BulletDebugDraw()
    bw.debugDraw(bdd)
    bcu.ProjectDrawLinesOpencv(bdd.points3da,bdd.points3db,cam_meta,viz_img)
    # print('Min-max 3d points:',np.min(bdd.points3da,axis=0), np.max(bdd.points3db,axis=0))
    penetration_depth = bw.computePenetration(bullet_bodies[0],bullet_bodies[1])
    # Getting detailed collision infor through CollisionVector
    # cvec1 = phys.CollisionVector()
    # cvec2 = phys.CollisionVector()
    # bw.checkPairCollision(bullet_bodies[0],bullet_bodies[1],cvec1)
    # bw.computePenetration(cvec2)
    # print('CollisionVec1:', cvec1[0].localPointA.data.T, cvec1[0].localPointB.data.T)
    # print('CollisionVec2:', cvec2[0].localPointA.data.T, cvec2[0].localPointB.data.T)
    if len(covec) > 0:
        print('Penetration depth CollisionDetection, BulletWorld, ratio:', \
            covec[0], penetration_depth, penetration_depth / (covec[0] + 0.000001))

    #Displaying the normals.
    cv2.imshow("viz",viz_img)
    cv2.waitKey(0)

