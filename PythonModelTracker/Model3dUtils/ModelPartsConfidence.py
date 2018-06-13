import PythonModel3dTracker.PyMBVAll as mbv
import PyModel3dTracker as M3DT
import BlenderMBV.BlenderMBVLib.RenderingUtils as RU


class ModelPartsConfidence:

    def __init__(self, model3d, model3dobj = None, mesh_manager = None, decoder = None, renderer = None, depth_cutoff = 500 ):
        self.model3d = model3d
        if model3dobj is None:
            self.mesh_manager = mesh_manager
            if mesh_manager is None:
                self.mesh_manager = mbv.Core.MeshManager()
                model3d.setupMeshManager(self.mesh_manager)

            self.model3dobj = M3DT.Model3dObjectiveFrameworkRendering(self.mesh_manager)
            if decoder is None:
                self.model3dobj.decoder = model3d.createDecoder()  # m3dt.Model3dObjectiveFrameworkDecoding.generateDefaultDecoder(model3d.model_collada)
            else:
                self.model3dobj.decoder = decoder
            if renderer is None:
                self.model3dobj.renderer = \
                    M3DT.Model3dObjectiveFrameworkRendering. \
                        generateDefaultRenderer(2048, 2048, "opengl",
                                                model3d.n_bones,
                                                mbv.Ren.RendererOGLBase.Culling.CullFront)
            else:
                self.model3dobj.renderer = renderer
        else:
            self.model3dobj = model3dobj
            self.mesh_manager = model3dobj.mesh_manager

        meshes = mbv.Core.MeshTicketList()
        self.mesh_manager.enumerateMeshes(meshes)
        model3d.parts.mesh = self.mesh_manager.getMesh(meshes[0])
        print self.mesh_manager.getMeshFilename(meshes[0])

        # self.model3dobj.tile_size = (128, 128)
        self.model3dobj.bgfg = M3DT.Model3dObjectiveFrameworkRendering.generate3DBoxBGFG(depth_cutoff)
        rois = M3DT.RenderingObjectives()
        roi = M3DT.RenderingObjectiveKinectParts()
        roi.model_parts = model3d.parts
        roi.architecture = M3DT.Architecture.cuda
        roi.depth_cutoff = depth_cutoff
        rois.append(roi)
        self.model3dobj.appendRenderingObjectivesGroup(rois)

    def process(self, images, calibs, state):
        states = mbv.Core.ParamVectors()
        for p in self.model3d.parts.parts_map:
            states.append(state)
        self.model3dobj.evaluateSetup(images, calibs[0], state, .2)
        obj_vals = self.model3dobj.evaluate(states, 0)
        return obj_vals

    def visualize(self, image, camera, state, obj_vals, excluded_parts=[]):
        part_colors = []
        for p, o in zip(self.model3d.parts.parts_map, obj_vals):
            print p.key(), o
            part_colors.append(255 - int(255 * o))

        viz = RU.visualize_parts(renderer=self.model3dobj.renderer.delegate, mmanager=self.model3dobj.mesh_manager,
                                 decoder=self.model3dobj.decoder, state=state, camera=camera, image=image,
                                 n_bones=self.model3d.n_bones, model_parts=self.model3d.parts,
                                 part_colors=part_colors, excluded_parts=excluded_parts)
        return viz