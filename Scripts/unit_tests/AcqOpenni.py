import PyMBVCore as core
import PyMBVAcquisition as acq
import cv2

"""
Displays a dataset or the sensor feed and saves the RGB-D images.
Press 'q' to quit.
"""

# Option 1: creating grabber that reads from openni depth sensor
grabber_ni = acq.OpenNIGrabber(True, True, 'media/openni.xml')

# Option 2: creating grabber that reads from file oni.
# grabber_ni = acq.OpenNIGrabber(True, True, 'media/openni.xml', 'ds/seq0.oni')

grabber_ni.initialize()

#Set the delay in miliseconds. Set to 0: wait for key press to continue.
delay = 0

#Set the total number of frames.
n_frames = 10

#Main loop
for i in range(n_frames):
    images, calibs = grabber_ni.grab()
    depth = images[0]
    rgb = images[1]

    #displaying the depth and rgb images.
    cv2.imshow("depth",depth)
    cv2.imshow("rgb",rgb)
    key = cv2.waitKey(delay)


    # Saves the images to disk.
    # depth_filename = "depth_{0}.png".format(i)
    # rgb_filename = "rgb_{0}.png".format(i)
    # cv2.imwrite(depth_filename,depth)
    # cv2.imwrite(rgb_filename,rgb)

    #getting the camera
    cam = calibs[0].camera

    #extracting/printing the point cloud
    pcl = core.Vector3fStorage()
    cam.PointCloud_fromDepthMap(depth,pcl)
    #print pcl.data

    #displaying camera matrices
    #print cam.Graphics_getViewportTransform(640,480).data
