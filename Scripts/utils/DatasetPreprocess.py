import PyMBVCore as core
import PyMBVAcquisition as acq
import PyMBVParticleFilter as pf
import PyPFUtilsLib as pful

import numpy as np
np.set_printoptions(precision=1)
import cv2
import matplotlib.pyplot as plt

#Input
datasets_xml = 'ds/bt_datasets.xml'
did = 'pdt'
limits = [0, 995]

datasets = pful.HTDataset()
datasets.load(datasets_xml)

ds_info = datasets.getDatasetInfo(did)
ds_info.limits = limits


#grabber = pful.AutoGrabber(ds_info.format,ds_info.stream_filename, ds_info.calib_filename)
grabber_depth = acq.AcquisitionFromImages(ds_info.stream_filename[0], ds_info.calib_filename)
grabber_rgb = acq.AcquisitionFromImages(ds_info.stream_filename[1], ds_info.calib_filename)

min_depth = []
max_depth = []
for f in range(ds_info.limits[1]):
    images_depth, calibs = grabber_depth.grab()
    images_rgb, calibs = grabber_rgb.grab()

    depth = images_depth[0]
    rgb = images_rgb[0]

    min_depth.append(np.min(depth))
    max_depth.append(np.max(depth))

    depth[depth >= max_depth[0]] = 0

    cv2.imshow("depth",depth)
    cv2.imshow("rgb",rgb)
    cv2.waitKey(33)

    depth_filename = grabber_depth.getImageFilename()
    #cv2.imwrite(depth_filename,depth)

print max_depth
X = range(ds_info.limits[1])
plt.plot(X,min_depth,label='min depth')
plt.plot(X,max_depth,label='max depth')

plt.ylim(min(min_depth)-10, max(max_depth)+10)
plt.legend(loc='upper left')
plt.show()


