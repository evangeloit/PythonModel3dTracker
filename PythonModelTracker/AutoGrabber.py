import os
import PythonModel3dTracker.Paths as Paths
import PythonModel3dTracker.PythonModelTracker.PyMBVAll as mbv

valid_input_formats = ['SFSerializedAcq', 'SFOni', 'SFImage','ser','oni','image', None]
default_openni_xml = os.path.join(Paths.media, 'openni.xml')


def create_di(params_ds,openni_xml = default_openni_xml):
    grabber = create(params_ds.format,
                     params_ds.stream_filename,
                     params_ds.calib_filename,
                     openni_xml)
    return grabber


def create(input_format, input_stream, input_calib=None, openni_xml = default_openni_xml):
    assert input_format in valid_input_formats
    if input_format in ['SFOni', 'oni']:
        assert len(input_stream) == 1
        grabber = mbv.Acq.OpenNIGrabber(True,True,openni_xml,str(input_stream[0]),False)
        #grabber = mbv.Acq.OpenNI2Grabber(True, True, str(input_stream[0]), False)
        grabber.initialize()
    elif input_format in ['SFImage', 'image']:
        assert len(input_stream) == 2
        g1 = mbv.Acq.AcquisitionFromImages(str(input_stream[0]), str(input_calib))
        g2 = mbv.Acq.AcquisitionFromImages(str(input_stream[1]), str(input_calib))
        grabber = mbv.Acq.CombinedAcquisition(g1, g2)
    elif input_format in ['SFSerializedAcq','ser']:
        assert len(input_stream) == 1
        grabber = mbv.Acq.SerializedAcquisition(str(input_stream[0]))
    return grabber

