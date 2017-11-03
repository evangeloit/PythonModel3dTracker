import PythonModelTracker.PyMBVAll as mbv
import PyModel3dTracker as pm3d
import BlenderMBVLib.RenderingUtils as ru
import numpy as np





class RigidObjectOptimizer:

    default_settings = {
        "particles" : 32,
        "generations" : 32,
        "depth_cutoff" : 50,
        "variances" : [50,50,50,0.1,0.1,0.1,0.1,0,0,0],
        "tile_size": (128,128)
    }

    def __init__(self,mmanager, renderer, decoder, model3d, settings = default_settings):
        self.mmanager = mmanager
        self.renderer = renderer.delegate
        self.exposed_renderer = renderer
        self.decoder = decoder
        self.model3d = model3d
        self.settings = settings


        self.model3dobj = self.gen_objective_framework()

        self.objective = mbv.PF.PFObjectiveCast.toParallel(self.model3dobj.getPFObjective())

    def __del__(self):
        mbv.Core.CachedAllocatorStorage.clear()
    

    def gen_objective_framework(self):
        roi = pm3d.RenderingObjectiveKinect()
        roi.architecture = pm3d.Architecture.cuda
        roi.fit_mode = pm3d.FitMode.model
        roi.depth_cutoff = self.settings['depth_cutoff']
        model3dobj = pm3d.Model3dObjectiveFrameworkRendering(self.mmanager)
        model3dobj.objective_combination = pm3d.ObjectiveResultsCombinationAverage()
        model3dobj.objective_combination.weights = mbv.Core.DoubleVector([-1.])
        model3dobj.decoder = self.decoder
        model3dobj.renderer = self.exposed_renderer
        model3dobj.tile_size = self.settings['tile_size']
        model3dobj.bgfg = pm3d.Model3dObjectiveFrameworkRendering.generate3DBoxBGFG(10*self.settings['depth_cutoff'])
        rois = pm3d.RenderingObjectives()
        rois.append(roi)
        model3dobj.appendRenderingObjectivesGroup(rois)
        return model3dobj

    def evaluate(self, imgs, clbs, state_):
        state = mbv.Core.DoubleVector(state_)
        states = mbv.Core.ParamVectors()
        states.append(state)
        camera = clbs[0]
        self.model3dobj.observations = imgs
        self.model3dobj.virtual_camera = camera
        bb = self.model3dobj.computeBoundingBox(state, .2)
        self.model3dobj.focus_rect = bb
        self.model3dobj.preprocessObservations()
        res = self.model3dobj.evaluate(states,0)
        print "Objective res:", res


    def optimize(self,imgs,clbs, state_):
        state = mbv.Core.DoubleVector(state_)
        camera = clbs[0]
        print 'state posest:', state
        self.model3dobj.observations = imgs
        self.model3dobj.virtual_camera = camera
        bb = self.model3dobj.computeBoundingBox(state, .2)
        self.model3dobj.focus_rect = bb
        self.model3dobj.preprocessObservations()

        m3d = self.model3d

        pso = mbv.Opt.PSOVariantOptimizer(self.settings['particles'],self.settings['generations'])
        pso.setBounds(m3d.low_bounds, m3d.high_bounds)
        pso.randomizationIndices = mbv.Core.UIntVector([0]*m3d.n_dims)
        # print m3d.low_bounds
        # print m3d.high_bounds
        sampler = mbv.Opt.ConstantPositionExplicitTrackingSampler()
        sampler.populationSize = self.settings['particles']
        sampler.variances = mbv.Core.DoubleVector(self.settings['variances'])
        sampler.randomSpecification = mbv.Opt.RandomSpecification([mbv.Opt.RandomModel.RandomGaussian]*m3d.n_dims)
        sampler.addToHistory(state)
        sampler.resample(pso)
        #print 'InitHypotheses:',pso.initialHypotheses

        pso.optimize(self.objective, m3d.n_dims, state)
        print 'state pso:', state
        return state

    def extract_visible(self,imgs,clbs,state_,thres = 10):
        state = mbv.Core.DoubleVector(state_)
        camera = clbs[0]
        depth = imgs[0]

        decoding = self.decoder.quickDecode(state)
        ru.render(self.renderer, self.mmanager, decoding, camera, [1, 1],
                  mbv.Ren.RendererOGLBase.Culling.CullNone, self.model3d.n_bones)
        positions, normals, colors, issue, instance, V = ru.genMaps(self.renderer)
        mask = (issue > 0)
        visible_depth = np.zeros_like(depth)
        visible_depth[mask] = depth[mask]
        ddiff_mask = (np.abs(visible_depth - positions[:,:,2]) > thres)
        visible_depth[ddiff_mask] = 0
        visible_points = mbv.Core.Vector3fStorage()
        camera.camera.PointCloud_fromDepthMap(visible_depth,visible_points)
        return visible_depth, visible_points

