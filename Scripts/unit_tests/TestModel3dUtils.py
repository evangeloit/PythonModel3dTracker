import PythonModel3dTracker.PyMBVAll as mbv
from Models3D.Models3dDict import model3d_dict
import PythonModel3dTracker.PythonModelTracker.Model3dUtils.Model3dUtils as m3dutils

model_name = "mh_body_male_custom"
model3d = mbv.PF.Model3dMeta.create(str(model3d_dict[model_name]["path"]))
model3d.parts.genBonesMap()

bone_lengths = m3dutils.GetBoneLengths(model3d)

for b,l in bone_lengths.items():
    if l>0: print b, l
