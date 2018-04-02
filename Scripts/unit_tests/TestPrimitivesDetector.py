import PyMBVCore as core
import PyMBVAcquisition as acq
import PyMBVParticleFilter as mpf
import PyHandTrackerPF as htpf
import PFSettings as pfs
import cv
import cv2
import numpy as np


n_frames = 10  # Set the total number of frames.

datasets_xml = "./ds_info/bt_datasets.xml"
#did = ["dev_00","seq0","seq1","sensor"][1]
did = ["pdt", "s09_a02"][0]
ht_dataset = htpf.HTDataset(datasets_xml)
ds_info = ht_dataset.getDatasetInfo(did)

grabber = htpf.AutoGrabber.create(ds_info.format, ds_info.stream_filename, ds_info.calib_filename)


cyl_detector = htpf.CylindersDetector()

for i in range(n_frames):
    images, calibs = grabber.grab()
    depth = images[0]
    rgb = images[1]

    cylinders = cyl_detector.detect(images,calibs)

    for c in cylinders:
        c.viz(calibs[0],rgb, core.Vector3(255,0,0))

    p3dp = htpf.Points3dPair()
    p3dp.first = core.Vector3(0,0,1800)
    p3dp.second = core.Vector3(500, 0,1800)
    radius = 100
    cylinder = htpf.Primitive3dCylinder(radius,p3dp)
    cylinder.viz(calibs[0], rgb, core.Vector3(0, 255, 0))

    #displaying the depth and rgb images.
    cv2.imshow("depth",depth)
    cv2.imshow("rgb",rgb)
    key = cv2.waitKey(330)