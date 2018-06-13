import PyMBVRendering as ren
import numpy as np
import os

import PyMBVCore as core
import PyMBVDecoding as dec
import PyMBVLibraries as lib
import PyMBVOpenMesh as mbvom
import PyMBVPhysics as phys

os.chdir(os.environ['bmbv']+"/Scripts/")


np.set_printoptions(precision=1)
import cv2
import BlenderMBVLib.RenderingUtils as ru
import PythonModel3dTracker.PythonModelTracker.BulletCollisionUtils as bcu

"""
Renders a primitives based model and checks for collisions.
"""

# Parameters
selected_shapes = ["sphere", "sphere", "cylinder"]
states = [[10,-200,600,0.,0.,0.,1.],
          [-30,-300,600,0.,0.,0.,1.],
          [0,-100,600,0.,0.,0.,1.]]
ground_plane_enable = True
rendering_size = [1,1]

n_obj = len(selected_shapes)
shapes = []
models = []
plane_shape = None
for s in selected_shapes:
    if s == "cylinder":
        models.append(Paths.models + "/hand_std/cylinder_low.obj")
        shape = phys.CylinderShapeZ()
        shape.scale = core.Vector3(50, 50, 50)
        shape.length = 2
        shape.radius = 1
    elif s == "sphere":
        models.append(Paths.models + "/hand_std/sphere_low.obj")
        shape = phys.SphereShape()
        shape.radius = 1
        shape.scale = core.Vector3(25, 25, 25)
    shapes.append(shape)




#Creating Bullet World
bw = phys.BulletWorld()
#bw.gravity = core.Vector3(0,10,0)
bullet_bodies = []
for i,s in enumerate(shapes):
    body = phys.BulletBody(phys.BulletShapeImplementation(s))
    body.mass = 1
    body.gravity = core.Vector3(0,10,0)
    body.position = core.Vector3(states[i][0],states[i][1],states[i][2])
    body.orientation = core.Quaternion(states[i][3],states[i][4],states[i][5],states[i][6])  # core.Quaternion(w,x,y,z)
    body.scale = shapes[i].scale
    bullet_bodies.append(body)
    bw.addBody(bullet_bodies[-1])
if ground_plane_enable :
    plane_shape = phys.PlaneShape()
    plane_shape.normal = core.Vector3(0, -1, 0)
    # plane_shape.scale = core.Vector3(1,1,1)
    # plane_shape.offset = 0
    plane_shape_body = phys.BulletBody(phys.BulletShapeImplementation(plane_shape))
    plane_shape_body.mass = 0
    plane_shape_body.position = core.Vector3(0, 0, 0)
    plane_shape_body.restitution = 1
    bw.addBody(plane_shape_body)
worlds = phys.WorldVector()
worlds.append(bw)
times = core.DoubleVector()
times.append(0.33)

sim = phys.BulletPhysicsSimulator()



#Initializing decoder and mesh manager.
mmanager = core.MeshManager()
openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
rigid_decoders = []
decoders = []
n_dims = 0
for d in range(n_obj):
    decoder_sr = dec.SingleRigidDecoder()
    decoder_sr.meshFilename = models[d]
    print(models[d])
    decoder_sr.loadMeshTickets(mmanager)
    decoder_sr.enableRotation = True
    decoder_sr.enableTranslation = True
    decoder_sr.scale = shapes[d].scale
    n_dims_prev = n_dims
    n_dims += decoder_sr.getDimensionsRequired()
    decoder_sl = dec.SlicedDecoder(decoder_sr)
    decoder_sl.slice = core.UIntVector(range(n_dims_prev,n_dims,1))
    rigid_decoders.append(decoder_sr)
    decoders.append(decoder_sl)

if n_obj > 1:
    decoder = dec.DecoderCombination(decoders[0],decoders[1])
    for i,d in enumerate(decoders):
        if i>1:
            decoder = dec.DecoderCombination(decoder, d)
else:
    decoder = decoders[0]

meshes = core.MeshTicketList()
mmanager.enumerateMeshes(meshes)
for m in meshes:
    print(mmanager.getMeshFilename(m))

print('Dim num:',n_dims)

# Setting camera
cam_meta = lib.RGBDAcquisitionSimulation.getDefaultCalibration()
cam_frust = cam_meta.camera
cam_frust.position = core.Vector3(0,0,0)
cam_meta.camera = cam_frust
renderer = ren.RendererOGLCudaExposed.get()



#Init Parameter Vector.
n_hyp = rendering_size[0]*rendering_size[1]
#
#Rendering Loop.
steps = 240
for i in range(steps):

    bdd = bcu.BulletDebugDraw()
    sim.simulate(worlds,times)

    for j,substate in enumerate(states):
        substate[0] = bullet_bodies[j].position.x
        substate[1] = bullet_bodies[j].position.y
        substate[2] = bullet_bodies[j].position.z
        substate[3] = bullet_bodies[j].orientation.w
        substate[4] = bullet_bodies[j].orientation.x
        substate[5] = bullet_bodies[j].orientation.y
        substate[6] = bullet_bodies[j].orientation.z

    state = [s for substate in states for s in substate]
    multi_state = core.ParamVectors()
    for j in range(n_hyp): multi_state.append(core.ParamVector(state))
    decoding = decoder.quickDecode(multi_state)
    ru.render(renderer, mmanager, decoding, cam_meta, rendering_size, renderer.Culling.CullFront, 0)
    positions, normals, colors, issue, instance, vertex_id = ru.genMaps(renderer)
    viz_img = normals[:, :, 0:3].astype(np.float32) * 0.5 + 0.5

    bw.debugDraw(bdd)
    bcu.ProjectDrawLinesOpencv(bdd.points3da,bdd.points3db,cam_meta,viz_img)

    #Displaying the normals.
    cv2.imshow("viz",viz_img)
    cv2.waitKey(99)

