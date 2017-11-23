import os
import json
import Models3D.Models3dDict as MD
import PythonModel3dTracker

package_path = os.environ['pm3dt'] #os.path.abspath(PythonModel3dTracker.__path__[0])
data = os.path.join(package_path, "Data")
models = MD.models
media = os.path.join(data, 'media/')
datasets = os.path.join(data, 'ds/')
results = os.path.join(data, 'rs/')
ds_info = data
objdetect = os.path.join(data, 'objdetect/')
settings = os.path.join(data, 'settings/')


def load_datasets_dict(load_path=None):
    if load_path is None:
        load_path = os.path.join(ds_info, 'datasets_dict.json')
    assert os.path.isfile(load_path)
    fp = open(load_path,'r')
    my_dict = json.load(fp)
    # json.dump(datasets_dict, fp, indent=4, sort_keys=True)
    fp.close()
    return my_dict


def save_datasets_dict(my_dict,save_path=None):
    if save_path is None:
        save_path = os.path.join(ds_info, 'datasets_dict.json')

    fp = open(save_path, 'w')
    json.dump(my_dict, fp, indent=4, sort_keys=True)
    fp.close()


model3d_dict = MD.model3d_dict
datasets_dict = load_datasets_dict()
for d in datasets_dict: datasets_dict[d] = os.path.join(datasets,datasets_dict[d])
