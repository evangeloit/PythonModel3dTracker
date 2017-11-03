# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os

import PythonModelTracker.DatasetInfo as dsi

os.chdir(os.environ['bmbv']+"/Scripts/")


def replace_path_part(di,old_path,new_path):
    di.gt_filename = di.gt_filename.replace(old_path, new_path,1)
    di.background = di.background.replace(old_path, new_path,1)
    di.calib_filename = di.calib_filename.replace(old_path, new_path,1)
    for (i, l) in di.landmarks.items():
        l['filename'] = l['filename'].replace(old_path, new_path,1)
        l['calib_filename'] = l['calib_filename'].replace(old_path, new_path,1)
    #di.stream_filename = [s.replace(old_path, new_path,1) for s in di.stream_filename]
    return di

def add_initialization(di):
    di.initialization = {}
    for f in di.gt.states:
        for m in di.gt.states[f]:
            di.initialization[m] = di.gt.states[f][m]
            print(f,m,di.gt.states[f][m])
    return di




cur_dir = Paths.datasets + "/human_tracking/mhad/"

for f_in in os.listdir(cur_dir):
    if not f_in.endswith("_gt.json"):
        if f_in.endswith(".json"):

            cur_path = os.path.join(cur_dir,f_in)
            print(cur_path)

            di = dsi.DatasetInfo()
            di.load(cur_path)
            #di = replace_path_part(di,'/','')
            di = add_initialization(di)
            print(di.initialization)


            # di.gt_filename = cur_dir + di.did + '_gt.json'
            #
            di.save(cur_path)





