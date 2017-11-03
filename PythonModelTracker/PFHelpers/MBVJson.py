import json
import PyMBVCore as core

def Save(filename, mbv_obj):
    json_target = open(filename, 'w')
    json.dump(mbv_obj.__pythonize__(), json_target, sort_keys=True,indent=4)


def Load(filename, class_):
    json_target = open(filename,'r')
    camera_python = json.load(json_target)
    camera = class_()
    camera.__depythonize__(camera_python)
    return camera