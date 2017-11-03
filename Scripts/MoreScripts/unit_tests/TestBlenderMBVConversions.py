import PyMBVCore as core
import PyMBVParticleFilter as mpf
import BlenderMBVLib.BlenderMBVConversions as bmc
import os
os.chdir(os.environ['bmbv']+"/Scripts/")

m3d = mpf.Model3dMeta.create('data/models3d/hand_skinned/hand_skinned.xml')
bm3d = bmc.BlenderModel3dMeta(m3d,m3d.default_state)


mbv_camera = core.CameraMeta()

frame_data = bmc.FrameDataMBV(m3d,m3d.default_state,mbv_camera,[0, 10, 100])

for nn,n in frame_data.blender_model3dmeta.dims.items():
    for t,v in n.items():
        print(nn, t,v)
