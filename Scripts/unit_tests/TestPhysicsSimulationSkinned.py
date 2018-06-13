import PyMBVRendering as ren
import numpy as np
import os

import PyMBVCore as core
import PyMBVDecoding as dec
import PyMBVLibraries as lib
import PyMBVOpenMesh as mbvom
import PyMBVParticleFilter as mpf
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
steps = 64
waitkey = 33
selected_shapes = ["cylinder","box","sphere","sphere"]
states = [[10,-100,600,0.,0.,0.,1.,50,50,50],
          [0,100,600,0.,0.,0.,1.,150,10,150],
          [10, -200, 610, 0., 0., 0., 1., 10, 10, 10],
          [10, -300, 620, 0., 0., 0., 1., 10, 10, 10]]
masses = [3, 0, 1, 1]
ground_plane_enable = False
rendering_size = [1,1]

n_obj = len(selected_shapes)
shapes = []
models = []
plane_shape = None
for shape_name in selected_shapes:
    models.append(mpf.Model3dMeta.create(Paths.model3d_dict[shape_name][1]))



class BulletSimulator:
    def __init__(self):
        self.bw = phys.BulletWorld()
        self.shapes = []
        self.bodies = []
        self.worlds = phys.WorldVector()
        self.worlds.append(self.bw)
        self.times = core.DoubleVector()
        self.times.append(0.33)
        self.sim = phys.BulletPhysicsSimulator()


    def simulate(self):
        self.sim.simulate(self.worlds, self.times)

    def debug_draw(self, cam_meta, viz_img):
        bdd = bcu.BulletDebugDraw()
        bullet_sim.bw.debugDraw(bdd)
        bcu.ProjectDrawLinesOpencv(bdd.points3da, bdd.points3db, cam_meta, viz_img)

    def add_bodies(self,models,states,masses):
        for state_l, model, mass in zip(states, models, masses):
            state = core.DoubleVector(state_l)
            shape_name = model.model_name
            if shape_name == "cylinder":
                shape = phys.CylinderShape()
                shape.length = 1
                shape.radius = 1
            elif shape_name == "sphere":
                shape = phys.SphereShape()
                shape.radius = 1
            elif shape_name == "box":
                shape = phys.BoxShape()
                shape.width = 1
                shape.height = 1
                shape.length = 1
            shape.scale = model.getScale(state)
            self.shapes.append(shape)
            body = phys.BulletBody(phys.BulletShapeImplementation(shape))
            body.mass = mass
            body.gravity = core.Vector3(0, 10, 0)
            body.position = model.getPosition(state)
            body.orientation = model.getOrientation(state)#core.Quaternion(state[3], state[4], state[5], state[6])  # core.Quaternion(w,x,y,z)
            body.scale = shape.scale
            self.bodies.append(body)
            self.bw.addBody(self.bodies[-1])

        # if ground_plane_enable:
        #     plane_shape = phys.PlaneShape()
        #     plane_shape.normal = core.Vector3(0, -1, 0)
        #     # plane_shape.scale = core.Vector3(1,1,1)
        #     # plane_shape.offset = 0
        #     plane_shape_body = phys.BulletBody(phys.BulletShapeImplementation(plane_shape))
        #     plane_shape_body.mass = 0
        #     plane_shape_body.position = core.Vector3(0, 0, 0)
        #     plane_shape_body.restitution = 1
        #     bw.addBody(plane_shape_body)
        #


bullet_sim = BulletSimulator()
bullet_sim.add_bodies(models,states,masses)


#Initializing decoder and mesh manager.
mmanager = core.MeshManager()
openmesh_loader = mbvom.OpenMeshLoader()
mmanager.registerLoader(openmesh_loader)
rigid_decoders = []
decoders = []
n_dims = 0
for d in range(n_obj):
    decoder_sr = models[d].createDecoder()
    models[d].setupMeshManager(mmanager)
    decoder_sr.loadMeshTickets(mmanager)
    n_dims_prev = n_dims
    n_dims += models[d].n_dims

    if n_obj > 1:
        decoder_sl = dec.SlicedDecoder(decoder_sr)
        decoder_sl.slice = core.UIntVector(range(n_dims_prev,n_dims,1))
        rigid_decoders.append(decoder_sr)
        decoders.append(decoder_sl)
    else:
        decoders.append(decoder_sr)

if n_obj > 1:
    decoder = dec.DecoderCombination(decoders[0],decoders[1])
    for i,d in enumerate(decoders):
        if i>1:
            decoder = dec.DecoderCombination(decoder, d)
else:
    decoder = decoders[0]

meshes = core.MeshTicketList()
mmanager.enumerateMeshes(meshes)
print("meshes:")
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
for i in range(steps):

    bullet_sim.simulate()

    for j,substate in enumerate(states):
        substate[0] = bullet_sim.bodies[j].position.x
        substate[1] = bullet_sim.bodies[j].position.y
        substate[2] = bullet_sim.bodies[j].position.z
        substate[3] = bullet_sim.bodies[j].orientation.x
        substate[4] = bullet_sim.bodies[j].orientation.y
        substate[5] = bullet_sim.bodies[j].orientation.z
        substate[6] = bullet_sim.bodies[j].orientation.w

    state = [s for substate in states for s in substate]
    print(i, 'state:', state)
    multi_state = core.ParamVectors()
    for j in range(n_hyp): multi_state.append(core.ParamVector(state))
    decoding = decoder.quickDecode(multi_state)
    for d in decoding:
        print('decoding mat:',d.data().matrices[0])
    ru.render(renderer, mmanager, decoding, cam_meta, rendering_size, renderer.Culling.CullFront, 1)
    positions, normals, colors, issue, instance, vertex_id = ru.genMaps(renderer)
    viz_img = normals[:, :, 0:3].astype(np.float32) * 0.5 + 0.5

    bullet_sim.debug_draw(cam_meta,viz_img)

    #Displaying the normals.
    cv2.imshow("viz",viz_img)
    cv2.waitKey(waitkey)

