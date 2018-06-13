import PyMBVParticleFilter as pf
import os

import PyMBVParticleFilter as pf
import PythonModel3dTracker.PythonModelTracker.PFSettings as pfs
import PythonModel3dTracker.PythonModelTracker.PFSettings.Hand.Architecture as architecture_a
import PythonModel3dTracker.PythonModelTracker.PFSettings.Hand.hand_skinned as settings_a
import PythonModel3dTracker.PythonModelTracker.PFSettings.Human.Architecture as architecture_h
import PythonModel3dTracker.PythonModelTracker.PFSettings.Human.mh_body_male_meta_glbscl as settings_h
import PythonModel3dTracker.PythonModelTracker.PFSettings.Object.Architecture as architecture_o
import PythonModel3dTracker.PythonModelTracker.PFSettings.Object.cylinder as settings_o
import PythonModel3dTracker.PythonModelTracker.PFSettings.PFSettingsGen as pfg

mdict = Paths.model3d_dict
s_dict = {}
s_dict["generic"] = {}
s_dict["generic"]["pf"] = {}
s_dict["generic"]["meta_opt"] = {}
s_dict["generic"]["meta_fit"] = {}
s_dict["generic"]["pf"]["type"] = pfg.pf_type
s_dict["generic"]["pf"]["n_particles"] = pfg.pf_n_particles
s_dict["generic"]["pf"]["like_variance"] = pfg.pf_like_variance
s_dict["generic"]["pf"]["arch_type"] = "2levels"

#meta optimizer params
s_dict["generic"]["meta_opt"]["n_generations"]=10
s_dict["generic"]["meta_opt"]["n_particles"]=16
s_dict["generic"]["meta_opt"]["std_dev"]=0.1


#meta fitter params
s_dict["generic"]["meta_fit"]["max_hist_frames"]=50
s_dict["generic"]["meta_fit"]["max_hist_meta"]=20
s_dict["generic"]["meta_fit"]["n_skip_frames"]=5

s_dict["model"] = {}
for m in mdict:
    m3d = pf.Model3dMeta.create(mdict[m][1])
    if mdict[m][0] == 'Human': architecture = architecture_h
    if mdict[m][0] == 'Hand': architecture = architecture_a
    if mdict[m][0] == 'Object': architecture = architecture_o
    s_dict["model"][m] = {}
    s_dict["model"][m]['std_dev_cond'] = {}
    for a in architecture.arch_types:
        s = pfs.GetSettings(model3d=m3d,model_class=mdict[m][0],sel_arch=a)
        if s is not None:
            for partition, params in s[0].model_params_map.items():
                s_dict["model"][m]['std_dev'] = params.std_dev
                s_dict["model"][m]['std_dev_cond'][partition] = params.std_dev_cond
#
fp = open(os.path.join(Paths.settings, 'pf_settings.json'), 'w')
json.dump(s_dict,fp,indent=2)
fp.close()


def convert_architectures_json():
    arch_dict = {}
    arch_dict['Human'] = {}
    arch_dict['Hand'] = {}
    arch_dict['Object'] = {}


    arch_dict['Human']['arch_types'] = {}
    for a in architecture_h.arch_types:
        arch = architecture_h.get_architercture(a, 0, 0,settings_h.pfhmf_std_dev,settings_h.pfhmf_std_dev_cond)
        arch_dict['Human']['arch_types'][a] = arch[1]
        arch_dict['Human']['model_parts_map'] = arch[2]

    arch_dict['Hand']['arch_types'] = {}
    for a in architecture_a.arch_types:
        arch = architecture_a.get_architercture(a, 0, 0,settings_a.pfhmf_std_dev,settings_a.pfhmf_std_dev_cond)
        arch_dict['Hand']['arch_types'][a] = arch[1]
        arch_dict['Hand']['model_parts_map'] = arch[2]

    arch_dict['Object']['arch_types'] = {}
    for a in architecture_o.arch_types:
        arch = architecture_o.get_architercture(a, 0, 0,settings_o.pfhmf_std_dev,settings_o.pfhmf_std_dev_cond)
        arch_dict['Object']['arch_types'][a] = arch[1]
        arch_dict['Object']['model_parts_map'] = arch[2]

    fp = open(os.path.join(Paths.settings, 'architecture.json'), 'w')
    json.dump(arch_dict,fp,indent=2)
    fp.close()

# Convert ModelDict to dict of dicts.
def convert_modeldict(mdict, output_path):
    print(mdict)
    ndict = {}
    for m in mdict:
        ndict[m] = {}
        ndict[m]['class'] = mdict[m][0]
        ndict[m]['path'] = mdict[m][1]
    Paths.save_model3d_dict(ndict, output_path)
