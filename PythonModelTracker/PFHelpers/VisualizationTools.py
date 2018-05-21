import PythonModel3dTracker.PyMBVAll as mbv
import BlenderMBV.BlenderMBVLib.RenderingUtils as ru



class Visualizer:
    def __init__(self, model3d, mesh_manager = None, decoder=None, renderer=None ):
        self.model3d = model3d
        if mesh_manager is None:
            mesh_manager = mbv.Core.MeshManager()
            openmesh_loader = mbv.OM.OpenMeshLoader()
            mesh_manager.registerLoader(openmesh_loader)
            model3d.setupMeshManager(mesh_manager)
        self.mesh_manager = mesh_manager

        if decoder is None:
            decoder = model3d.createDecoder()
            decoder.loadMeshTickets(mesh_manager)
            # if model3d.model_type == mbv.PF.Model3dType.Primitives:
            #     model3d.parts.genPrimitivesMap(decoder)
            # else:
            #     model3d.parts.genBonesMap()
        self.decoder = decoder

        if renderer is None:
            renderer = mbv.Ren.RendererOGLCudaExposed.get()
            renderer.n_bones = model3d.n_bones
            renderer.culling = mbv.Ren.RendererOGLBase.Culling.CullNone
        self.renderer = renderer



    def visualize_overlay(self, state, camera, image, points3d=None):
        viz = ru.visualize_overlay(self.renderer,self.mesh_manager,self.decoder,
                                     state,camera,image,self.model3d.n_bones, points3d)
        return viz

    def visualize_parts(self, state, camera, image, parts):
        viz = ru.visualize_parts(self.renderer, self.mesh_manager, self.decoder,
                                     state, camera, image, self.model3d.n_bones, parts)
        return viz


