# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import PyMBVCore as Core
import PyMBVAcquisition as acq
import PyMBVDecoding as dec
import PyHandTracker as ht
import PyPFUtilsLib as pf

import cv2
import numpy as np
import csv
import LandmarksGrabber as ldm

# <codecell>

ds = pf.HTDataset("media/pf/bt_datasets.xml","human_ext")
ds.PrintDatasets()
ds.genFilenames("S09_A02",0,0,0)
print 'StreamFilename: ',ds.getStreamFilename()[0]
print 'CalibFilename: ',ds.getCalibFilename()
meta_info =  ds.getMetaDatasetInfo(pf.MetaType.MTLandmarks)
if (meta_info):
    print 'MetaInfo: ', meta_info.name, meta_info.type, meta_info.info
    meta_info_landmarks = pf.MetaDatasetInfoLandmarks(meta_info)
    print 'MetaInfoLandMarks: ', meta_info_landmarks.type,meta_info_landmarks.filename
grabber = pf.AutoGrabber(pf.StreamFormat.SFImage,ds.getStreamFilename(),ds.getCalibFilename())
grabber_ldm = ldm.LandmarksGrabber(meta_info_landmarks.filename, meta_info_landmarks.calib_filename)

# <codecell>

input_res_path = 'rs/body_track_000.txt'
track  = np.loadtxt(input_res_path, dtype=float, comments='#', delimiter=',') 


m_manager = Core.MeshManager()
decoder = dec.GenericDecoder()
decoder.loadFromFile('media/human_ext.xml')
decoder.loadMeshTickets(m_manager)

dist_calc = pf.FilteredPrimitivesDistObjective()
dist_calc.setDecoder(decoder)
#dist_calc.setAcceptedPrimitives(Core.StringVector(["arm_left_sphere_1"]))



    

# <codecell>

def getPrimitives3dpos(primitives,state_vec):
    translations = Core.Vector3fStorage()
    dist_calc.setPrimitives(primitives)
    dist_calc.decodeToTranslations(state_vec,translations)
    points3d_model = np.zeros([0,3])
    for i in translations:
        points3d_model = np.append(points3d_model,[[i.x, i.y, i.z]],axis=0)    
    return points3d_model
        

# <codecell>

i=0
quit = False
while(i<55):
    img, clb = grabber.grab()
    depth_img = np.copy(img[0])
    color_img = np.copy(img[1])
    intr = clb[0].camera.OpenCV_getIntrinsics((640,480))
    dist = np.zeros((1,4),float)
        
    points3d_det_names,points3d_det,clb_tmp = grabber_ldm.acquire()
    points2d_det,jacobian = cv2.projectPoints(points3d_det, np.zeros([1,3]),np.zeros([1,3]), intr, dist)
    print 'points2d_det:',points2d_det
        
    cur_state = Core.DoubleVector()
    for e in track[i]:
        cur_state.append(e)
    primitive_names = ldm.getPrimitiveNamesfromLandmarkNames(points3d_det_names)
    points3d_model = getPrimitives3dpos(primitive_names, cur_state)
    #print 'points3d_model:',points3d_model
    points2d_model,jacobian = cv2.projectPoints(points3d_model, np.zeros([1,3]),np.zeros([1,3]), intr, dist)
    print 'points2d_model:',points2d_model
    
    points3d_det_vec = Core.Vector3fStorage()
    for p in points3d_det:
        points3d_det_vec.append(Core.Vector3(p[0],p[1],p[2]))
        
    dist_calc.setObservations(primitive_names,points3d_det_vec)
    cur_dist = dist_calc.evaluate(cur_state);
    
    print "Frame: ",i, " dist:",cur_dist
    
       
    #color = (50000,50000,50000)
    #for p in points2d_model:
    #    x =  int(p[0][0])
    #    y =  int(p[0][1])    
    #    cv2.rectangle(depth_img, (x-1,y-1), (x+1,y+1), color,5)
    color = (100000,100000,100000)
    for p in points2d_det:
        x =  int(p[0][0])
        y =  int(p[0][1])    
        cv2.rectangle(depth_img, (x-1,y-1), (x+1,y+1), color,5)
    cv2.imshow("rgb",color_img)
    cv2.imshow("viz",depth_img)    
    cv2.waitKey(0)
    i+=1    


# <codecell>


# <codecell>


