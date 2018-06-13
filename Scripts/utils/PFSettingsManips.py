import PyMBVParticleFilter as pf

import PythonModel3dTracker.PythonModelTracker.PFHelpers.PFSettings as pfs

model_name = 'mh_body_male'

model_xml = Paths.model3d_dict[model_name]['path']
model_class = Paths.model3d_dict[model_name]['class']
pf_settings_setup,pf_settings,hmf_architectures,model_settings = pfs.LoadAll(model_name,model_class)

model3d = pf.Model3dMeta.create(model_xml)

print('Loaded model from <', model_xml, '>')
print('Bones num:', model3d.n_bones)
print('Dim num:', model3d.n_dims)
print('\n\n----Source Model Dimensions Info----\n')
pfs.PrintStdDev(model3d, model_settings['default']['std_dev'])
pfs.PrintStdDevCond(model3d, model_settings['default']['std_dev_cond'])

#pf_settings['model']['hand_skinned_l'] = pf_settings['model'][model_name]

#Save(pf_settings,hmf_architectures,model_settings)


# for m in pf_settings['model']:
#     settings_filename = os.path.join(paths.settings, m+'.json')
#     fp = open(settings_filename,'w')
#     model_settings = {}
#     model_settings['default'] = pf_settings['model'][m]
#     json.dump(model_settings,fp,indent=2,sort_keys=True)
#     fp.close()


