import json
import os
import re
import PythonModel3dTracker.Paths as Paths

class DatasetInfo:

    optional_attributes = ['landmarks','background','gt_filename']

    def __init__(self,did=None,format=None,stream_filename=None,calib_filename=None,flip_images=None,
                 limits=None,gt_filename=None,landmarks=None,background=None):
        self.did = did
        self.format = format
        self.stream_filename = stream_filename
        self.calib_filename = calib_filename
        self.flip_images = flip_images
        self.limits = limits
        self.gt_filename = gt_filename
        self.landmarks = landmarks
        self.background = background
        #self.gt = None
        self.json_dir = None
        self.initialization = None

    def save(self,filename):
        if len(filename) > 3:
            json_target = open(filename, 'w')
            cur_dict = self.__dict__
            cur_dict = DatasetInfo.strip_base_dir(cur_dict, self.json_dir)
            cur_dict.pop('gt',None)
            cur_dict.pop('json_dir', None)
            json.dump(cur_dict, json_target, indent=4, sort_keys=True)
            print('Saving DatasetInfo to: <{}>'.format(filename))
        else:
            print('Cannot save results, invalid filename: <{}>'.format(filename))

    def load(self,filename):
        json_source = open(filename,'r')
        self.__dict__ = json.load(json_source)
        self.json_dir = os.path.dirname(filename) + '/'
        for oa in DatasetInfo.optional_attributes:
            if not hasattr(self, oa): self.__dict__[oa] = None
        self.append_base_dir()
        #self.gt = mtr.ModelTrackingResults()
        #self.gt.load(self.gt_filename)

    def generate(self,dataset):
        if os.path.isfile(dataset):
            (dirname, filename) = os.path.split(dataset)
            (did, extension) = os.path.splitext(filename)
            print(dirname, filename, did, extension)
            if extension == '.json':
                self.load(dataset)
                json_info_path = dataset
            elif extension == '.oni':
                json_info_path = os.path.join(dirname, did + '.json')
                if os.path.isfile(json_info_path):
                    print('Loading dataset info from <{0}>.'.format(json_info_path))
                    self.load(json_info_path)
                else:
                    self.did = did
                    self.format = 'SFOni'
                    self.stream_filename = [dataset]
                    self.calib_filename = ""
                    self.flip_images = False
                    self.limits = [0, 1000]
                    self.initialization = {}
                    self.json_dir = dirname
            if not (did in Paths.datasets_dict):
                Paths.datasets_dict[did] = json_info_path
                Paths.save_datasets_dict(Paths.datasets_dict)
        elif os.path.isdir(dataset):

            (parent_path, did) = os.path.split(os.path.normpath(dataset))
            json_info_path = os.path.join(parent_path, did+'.json')
            if os.path.isfile(json_info_path):
                print('Loading dataset info from <{0}>.'.format(json_info_path))
                self.load(json_info_path)
            else:
                self.did = did
                self.format = 'SFImage'
                self.calib_filename = os.path.join(parent_path,"calib.txt")
                self.flip_images = False
                self.limits = [0, 0]
                self.initialization = {}
                self.json_dir = parent_path
                self.stream_filename = self.extract_stream_filename(dataset)
            if not (did in Paths.datasets_dict):
                Paths.datasets_dict[did] = json_info_path
                Paths.save_datasets_dict(Paths.datasets_dict)
        else:
            did = dataset
            self.load(os.path.join(Paths.datasets, Paths.datasets_dict[did]))  # ds.getDatasetInfo(did)

    def extract_stream_filename(self, path, dstype='mhad'):

        rgb_names = ['color', 'rgb', 'images']
        depth_names = ['depth', 'dpt']
        filelist = os.listdir(path)
        fname_templates = []
        fname_extensions = []
        dir_names = [None, None]

        #Check for directories with images.
        for f in filelist:
            f_full = os.path.join(path, f)
            if os.path.isdir(f_full):
                for n in depth_names:
                    if n in f: dir_names[0] = f_full
                for n in rgb_names:
                    if n in f: dir_names[1] = f_full
        if dstype == 'mhad':
            # pass
            if None in dir_names:
                for f in filelist:
                    fname, fext = os.path.splitext(f)
                    # print(fname)
                    # print(fext)
                    number_str = re.sub(r'(\D)+', '', fname)
                    # print(number_str)
                    # string_str = re.sub(r'(\d)+', '', fname)
                    string_str = fname[:-5]
                    # print(string_str)
                    if number_str:
                        if string_str not in fname_templates:
                            fname_templates.append(string_str)
                            # print(fname_templates)
                            fname_extensions.append(fext)
                        frame_num = int(number_str[8:13])
                        digits_number = len(number_str[8:13])
                        # print(digits_number)
                        # print(frame_num)
                        if (frame_num > self.limits[1]): self.limits[1] = frame_num
                assert len(fname_templates) == 2
                depth_found = 0
                rgb_found = 0
                for i, f in enumerate(fname_templates):
                    for n in depth_names:
                        if n in f:
                            d_idx = i
                            depth_found += 1
                            break
                    for n in rgb_names:
                        if n in f:
                            rgb_found += 1
                            break
                assert (depth_found == 1) and (rgb_found == 1)
                stream_filename = [
                    os.path.join(path, fname_templates[d_idx] + '%0{0}d'.format(digits_number) + fname_extensions[d_idx]),
                    os.path.join(path,
                                 fname_templates[1 - d_idx] + '%0{0}d'.format(digits_number) + fname_extensions[1 - d_idx])]
        elif dstype == 'normal':
            # pass
            if None in dir_names:
                for f in filelist:
                    fname, fext = os.path.splitext(f)
                    number_str = re.sub(r'(\D)+', '', fname)
                    string_str = re.sub(r'(\d)+', '', fname)
                    if number_str:
                        if string_str not in fname_templates:
                            fname_templates.append(string_str)
                            fname_extensions.append(fext)
                        frame_num = int(number_str)
                        digits_number = len(number_str)
                        print(frame_num)
                        if (frame_num > self.limits[1]): self.limits[1] = frame_num
                assert len(fname_templates) == 2
                depth_found = 0
                rgb_found = 0
                for i, f in enumerate(fname_templates):
                    for n in depth_names:
                        if n in f:
                            d_idx = i
                            depth_found += 1
                            break
                    for n in rgb_names:
                        if n in f:
                            rgb_found += 1
                            break
                assert (depth_found == 1) and (rgb_found == 1)
                stream_filename = [
                    os.path.join(path, fname_templates[d_idx] + '%0{0}d'.format(digits_number) + fname_extensions[d_idx]),
                    os.path.join(path, fname_templates[1 - d_idx] + '%0{0}d'.format(digits_number) + fname_extensions[1 - d_idx])]
        else:

            stream_filename = []
            for d in dir_names:
                filelist = os.listdir(d)
                fname_templates = []
                fname_extensions = []
                for f in filelist:
                    fname, fext = os.path.splitext(f)
                    number_str = re.sub(r'(\D)+', '', fname)
                    string_str = re.sub(r'(\d)+', '', fname)
                    if number_str:
                        if string_str not in fname_templates: fname_templates.append(string_str)
                        if fext not in fname_extensions: fname_extensions.append(fext)
                        frame_num = int(number_str)
                        digits_number = len(number_str)
                        print(frame_num)
                        if (frame_num > self.limits[1]): self.limits[1] = frame_num

                assert len(fname_templates) == 1
                stream_filename.append(os.path.join(d, fname_templates[0] + '%0{0}d'.format(digits_number) + fname_extensions[0]))
        return stream_filename





    @property
    def json_filename(self):
        return os.path.join(self.json_dir, self.did + '.json')



    # def getInitState(self,model_name):
    #     states = self.gt.get_model_states(model_name)
    #     return states[self.limits[0]]

    @staticmethod
    def strip_base_dir(jsondict,json_dir):
        if json_dir[-1] != '/': json_dir += '/'
        if jsondict['gt_filename'] is not None:
            jsondict['gt_filename'] = jsondict['gt_filename'].replace(json_dir, '')
        if jsondict['background'] is not None:
            jsondict['background'] = jsondict['background'].replace(json_dir, '')
        jsondict['calib_filename'] = jsondict['calib_filename'].replace(json_dir, '')
        if jsondict['landmarks'] is not None:
            for (i, l) in jsondict['landmarks'].items():
                l['filename'] = l['filename'].replace(json_dir, '')
                l['calib_filename'] = l['calib_filename'].replace(json_dir, '')
        jsondict['stream_filename'] = [s.replace(json_dir, '') for s in jsondict['stream_filename']]
        return jsondict

    
    def append_base_dir(self):
        if self.gt_filename is not None:
            self.gt_filename = os.path.join(self.json_dir,self.gt_filename)
        if self.background is not None:
            self.background = os.path.join(self.json_dir,self.background)
        self.calib_filename = os.path.join(self.json_dir,self.calib_filename)
        if self.landmarks is not None:
            for (i, l) in self.landmarks.items():
                l['filename'] = os.path.join(self.json_dir,l['filename'])
                l['calib_filename'] = os.path.join(self.json_dir,l['calib_filename'])
        self.stream_filename = [os.path.join(self.json_dir,s) for s in self.stream_filename]
