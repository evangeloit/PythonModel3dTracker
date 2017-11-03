import PyMBVCore as core
import PyMBVRendering as ren
import PyMBVAcquisition
import PyMBVParticleFilter as pf
import PyModel3dTracker as htpf
#import RenderingUtils as ru
import cv2
import numpy as np

#Python RenderingObjective, deriving from HandTracking::RenderingObjectiveBase.
class MyCustomRenderingObjective(htpf.RenderingObjectiveBase):
    def evaluate(self,observations,renderings):
        print('evaluating, some serious python calculations in progress, please wait...')
        results = core.DoubleVector([4.0])
        return results

    def getRenderingMapIDs(self):
        WriteFlag = ren.Renderer.WriteFlag
        Channel = ren.MapExposer.Channel
        ids = htpf.RenderingMapIDList()
        ids.append(htpf.RenderingMapID(WriteFlag.WriteID, Channel.X))
        ids.append(htpf.RenderingMapID(WriteFlag.WriteID, Channel.Y))
        ids.append(htpf.RenderingMapID(WriteFlag.WritePosition, Channel.Z))
        return ids

    def getObservationsTypes(self):
        otl = htpf.ObservationTypeList()
        otl.append(htpf.ObservationType.ot_depth)
        otl.append(htpf.ObservationType.ot_labels)
        return otl

#Python RenderingObjective, deriving from HandTracking::RenderingObjectiveBase.
class MyCustomDecodingObjective(htpf.DecodingObjectiveBase):
    def evaluate(self,decoding):
        print('evaluating decoding obj, some serious python calculations in progress, please wait...')
        results = core.DoubleVector([4.0])
        return results
