# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os

import PyModel3dTracker as mt

import PythonModel3dTracker.PythonModelTracker.Grabbers.DatasetInfo as dsi
import PythonModel3dTracker.PythonModelTracker.TrackingResults.ModelTrackingResults as mtr

os.chdir(os.environ['bmbv']+"/Scripts/")

def add_initialization(di,gt):
    di.initialization = {}
    for f in gt.states:
        for m in gt.states[f]:
            di.initialization[m] = gt.states[f][m]
            print(f,m,gt.states[f][m])
    return di

ds_folder = os.path.join(Paths.ds_info, "ht_datasets.xml").__str__()
print(ds_folder)

ds = mt.HTDataset(ds_folder)
ds.print()
ds_ids = ds.getDatasetIDs()

#
for d in ds_ids:
    ds_info = ds.getDatasetInfo(d)
    #print(ds_info.did.__class__, ds_info.format.__class__, ds_info.stream_filename.__class__, ds_info.calib_filename.__class__, ds_info.flip_images.__class__,
    #      ds_info.limits.__class__, ds_info.getInitState('human_ext').__class__)
    ds_info_filename = os.path.join(Paths.ds_info, ds_info.did + '.json')
    gt_filename = os.path.join(Paths.ds_info, ds_info.did + '_gt.json').__str__()
    dataset_info = {
        "did":ds_info.did,
        "format":ds_info.format.__str__(),
        "stream_filename":[d for d in ds_info.stream_filename],
        "calib_filename":ds_info.calib_filename,
        "flip_images":ds_info.flip_images,
        "limits":ds_info.limits,
        "gt_filename":None,
        "json_dir":os.path.split(ds_info.stream_filename[0])[0] + '/'
    }

    # tdsi = DatasetInfo(dataset_info['did'],dataset_info['format'],dataset_info['stream_filename'],
    #                    dataset_info['calib_filename'],dataset_info['flip_images'])
    # tdsi.save(tds_info_filename)

    #Get meta landmarks info.
    dataset_info["landmarks"] = {}
    lm_meta = ds_info.getMetaDatasetInfo(mt.MetaType.MTLandmarks)
    for l in lm_meta:
        dataset_info["landmarks"][l.name] = {}
        dataset_info["landmarks"][l.name]["filename"] = l.filename
        dataset_info["landmarks"][l.name]["calib_filename"] = l.calib_filename
        dataset_info["landmarks"][l.name]["format"] = l.format

    # Get meta landmarks info.
    bg_meta = ds_info.getMetaDatasetInfo(mt.MetaType.MTBackground)
    for b in bg_meta:
        dataset_info["background"] = b.filename
    else:
        dataset_info["background"] = None

    ##Get gt info
    gt = mtr.ModelTrackingResults()
    gt.did = dataset_info['did']
    gt_meta = ds_info.getMetaDatasetInfo('gt_init',mt.MetaType.MTResParamvec)
    for m in gt_meta.getAvailableModels():
        for f in gt_meta.getAvailableFrames(m):
            print(m,f,gt_meta.getResult(m,f))
            gt.add(f,m,gt_meta.getResult(m,f))

    tdsi = dsi.DatasetInfo()
    tdsi.__dict__ = dataset_info
    add_initialization(tdsi,gt)
    print(tdsi.did, tdsi.json_dir)
    #Saving everything.
    tdsi.save(ds_info_filename)
    #gt.save(gt_filename)




