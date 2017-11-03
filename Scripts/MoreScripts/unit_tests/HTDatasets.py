import PyMBVCore as core
import PyMBVAcquisition as acq
import PyPFUtilsLib as pfutil

ds = pfutil.HTDataset()
ds.load("../../media/pf/ht_datasets_new.xml")
dids = ds.getDatasetIDs()
print dids

di = ds.getDatasetInfo('sensor')
print di.id, '  stream fname:<',di.stream_filename[0],'> , calib fname:<', di.calib_filename, \
    '> , flip:', di.flip, ' , format:',di.format