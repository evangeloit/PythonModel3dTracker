import PyMBVCore as core
import PyMBVParticleFilter as pf

sel_part = 'little'
model3d_xml = Paths.model3d_dict['mh_body_male_custom']['path']
model3d = pf.Model3dMeta.create(str(model3d_xml))

default_state = model3d.default_state

print(default_state)
model3d.setPosition(default_state, core.Vector3(10,20,30))
model3d.setOrientation(default_state, core.Quaternion(0.1,0.75,0.2,0.0))
model3d.setScale(default_state, core.Vector3(1.7,2.7,3.7))
print(default_state)
