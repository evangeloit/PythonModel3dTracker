import json
import numpy as np
import os
import PathsPM3DT as Paths

np.set_printoptions(precision=2)
np.set_printoptions(suppress=False)
import copy


def Load(model_name=None,model_class=None,hmf_arch_type=None,pf_settings_filename=None, arch_filename=None):
    pf_settings_setup,_,_,_ = LoadAll(model_name,model_class,hmf_arch_type,pf_settings_filename,arch_filename)
    return pf_settings_setup

def LoadAll(model_name=None,model_class=None,hmf_arch_type=None,pf_settings_filename=None, arch_filename=None):

    if pf_settings_filename is None:
        pf_settings_filename = os.path.join(Paths.settings, 'pf_settings.json')
    fp = open(pf_settings_filename)
    pf_settings = json.load(fp)
    fp.close()

    if arch_filename is None:
        arch_filename = os.path.join(Paths.settings, 'hmf_architectures.json')
    fp = open(arch_filename)
    hmf_architectures = json.load(fp)
    fp.close()

    if model_name is not None:
        model_settings_filename = os.path.join(Paths.settings, 'model/', model_name + '.json')
        fp = open(model_settings_filename)
        model_settings = json.load(fp)
        fp.close()

    pf_settings_setup = copy.deepcopy(pf_settings)
    if hmf_arch_type is not None: pf_settings_setup['pf']['arch_type'] = hmf_arch_type
    if model_name is not None:
        pf_settings_setup['pf']['std_dev'] = model_settings["default"]["std_dev"]
        pf_settings_setup['pf']['std_dev_cond'] = model_settings["default"]["std_dev_cond"]
    if model_class is not None:
        pf_settings_setup['pf']['children_map'] = hmf_architectures[model_class]["arch_types"][pf_settings_setup['pf']['arch_type']]
        pf_settings_setup['pf']['model_parts_map'] = hmf_architectures[model_class]['model_parts_map']
    return pf_settings_setup, pf_settings, hmf_architectures, model_settings


def Save(pf_settings=None, hmf_architectures=None, model_settings=None,
         settings_filename=None, arch_filename=None, model_settings_filename=None):
    if pf_settings is not None:
        if settings_filename is None:
            settings_filename = os.path.join(Paths.settings, 'pf_settings_new.json')
        fp = open(settings_filename, 'w')
        json.dump(pf_settings, fp, indent=2, sort_keys=True)
        fp.close()

    if hmf_architectures is not None:
        if arch_filename is None:
            arch_filename = os.path.join(Paths.settings, 'hmf_architectures_new.json')
        fp = open(arch_filename, 'w')
        json.dump(hmf_architectures, fp, indent=2, sort_keys=True)
        fp.close()

    if model_settings is not None:
        if model_settings_filename is not None:
        #    model_settings_filename = os.path.join(paths.settings, 'model/', model_name + '.json')
            fp = open(model_settings_filename, 'w')
            json.dump(model_settings, fp, indent=2, sort_keys=True)
            fp.close()


def PrintStdDev(model3d,std_dev):
    print('d, name, type, axis, [low, default, high], pf_std_dev')
    for d in range(model3d.n_dims):
        print(d, model3d.dim_names[d],model3d.dim_types[d], model3d.dim_axes[d], \
            '[',model3d.low_bounds[d], model3d.default_state[d], model3d.high_bounds[d],']',std_dev[d])

def PrintStdDevCond(model3d,std_dev_cond_map):
    for partition in model3d.partitions.partitions:
        partition_name = partition.key()
        partition_mask = partition.data()

        if partition_name in std_dev_cond_map:
            std_dev_cond = std_dev_cond_map[partition_name]
            if len(std_dev_cond) == sum(partition_mask):
                print(partition_name)
                m_counter = 0
                print('d, m, name, type, axis, [low, default, high], pf_std_dev_cond')
                for d,m in enumerate(partition_mask):
                    if m:
                        print(d, m_counter, model3d.dim_names[d], model3d.dim_types[d], model3d.dim_axes[d],
                              '[', model3d.low_bounds[d], model3d.default_state[d], model3d.high_bounds[d], ']',
                              std_dev_cond[m_counter])
                        m_counter += 1


