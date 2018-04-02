import PyMBVCore as core
import numpy as np
import cv2
import zmq
import BlenderMBV.BlenderMBVLib.BlenderMBVConnection as bmc
import BlenderMBV.BlenderMBVLib.BlenderMBVConversions as bmcnv
import abc
import json


class FrameDataOpencv:
    def __init__(self,depth=None,labels=None,rgb=None,n_frame=0):
        self.depth = depth
        self.labels = labels
        self.rgb = rgb
        self.n_frame = n_frame



class ModelTrackingGui():#metaclass=abc.ABCMeta):

    accepted_commands = ['frame', 'init', 'state', 'quit', 'background', 'optimize', 'save']


    class Command:
        def __init__(self,name,data=None):
            assert name in ModelTrackingGui.accepted_commands
            self.name = name
            self.data = data

    #@abc.abstractmethod
    def recv_command(self):
        pass

    #@abc.abstractmethod
    def recv_frame(self):
        pass

    #@abc.abstractmethod
    def send_frame(self): pass



class ModelTrackingGuiNone(ModelTrackingGui):
    def __init__(self, init_frame=0):
        self.next_frame = init_frame
        self.frame_recieved = False

    def recv_command(self):
        if self.frame_recieved:
            self.next_frame += 1
            return ModelTrackingGui.Command('frame')
        else:
            return ModelTrackingGui.Command('init')

    def recv_frame(self):
        self.frame_recieved = True
        return self.next_frame

    def send_frame(self,frame_data=None): pass


class ModelTrackingGuiOpencv(ModelTrackingGui):
    accepted_keys = ['q', 'p', 'n', 'c', 'b', 's']
    visualize_defaults = {'enable': True,
                          'client': 'opencv', 'labels': True, 'depth': True, 'rgb': True,
                          'wait_time': 0}

    def __init__(self, visualize=visualize_defaults, init_frame=0):
        self.visualize = visualize
        self.frame_data = None
        self.next_frame = init_frame

    def recv_command(self):
        cur_command = 'invalid'
        if self.frame_data is not None:
            key = 'a'
            while (key not in ModelTrackingGuiOpencv.accepted_keys):
                key = chr(cv2.waitKey(self.visualize['wait_time']) & 255)
                if key == 'q': cur_command = 'quit'
                if key == 'b': cur_command = 'background'
                if (key == 'p') or (key == 'n'):
                    self.visualize['wait_time'] = 0
                    if (key == 'n'): self.next_frame = self.frame_data.n_frame + 1
                    if (key == 'p'): self.next_frame = self.frame_data.n_frame - 1
                    cur_command = 'frame'
                if (key == 's'):
                    cur_command = 'save'
                if key == 'c':
                    self.next_frame = self.frame_data.n_frame + 1
                    self.visualize['wait_time'] = 10
                    cur_command = 'frame'
                if (self.visualize['wait_time'] != 0) and \
                   (key not in ModelTrackingGuiOpencv.accepted_keys):
                    self.next_frame = self.frame_data.n_frame + 1
                    cur_command = 'frame'
                    break
        else:
            cur_command = 'init'
        return ModelTrackingGui.Command(cur_command)

    def recv_frame(self):
        return self.next_frame

    def send_frame(self,frame_data):
        self.frame_data = frame_data
        if self.visualize['labels'] and (frame_data.labels is not None): cv2.imshow("labels", self.frame_data.labels)
        if self.visualize['depth'] and (frame_data.depth is not None): cv2.imshow("depth", self.frame_data.depth)
        if self.visualize['rgb'] and (frame_data.rgb is not None): cv2.imshow("rgb", self.frame_data.rgb)
        self.next_frame = self.frame_data.n_frame


class ModelTrackingGuiZeromq(ModelTrackingGui):
    sendtimeo = 3000

    def __init__(self):
        self.bmc_server = bmc.ConnectionManager(bmc.server_ports)
        self.bmc_server.sockets["cmd_out"].setsockopt(zmq.SNDTIMEO, ModelTrackingGuiZeromq.sendtimeo)
        self.bmc_server.sockets["data_out"].setsockopt(zmq.SNDTIMEO, ModelTrackingGuiZeromq.sendtimeo)
        self.bmc_server.poller.register(self.bmc_server.sockets["data_in"], zmq.POLLIN)
        self.frame_data = None

    def recv_command(self):
        msg = self.bmc_server.sockets["cmd_in"].recv_string()
        assert msg in ModelTrackingGui.accepted_commands
        #print('Received <{0}> command from client.'.format(msg))
        return ModelTrackingGui.Command(msg)


    def recv_state(self,model3d,state):
        socks = dict(self.bmc_server.poller.poll(1))
        if self.bmc_server.sockets["data_in"] in socks and socks[self.bmc_server.sockets["data_in"]] == zmq.POLLIN:
            blender_model3dmeta = json.loads(self.bmc_server.sockets["data_in"].recv_json())
            state = bmcnv.getModel3dMetaState(blender_model3dmeta,model3d, state)
            print('Received state: ', state)
            return state
        else:
            print('Failed to receive state from client.')
            return None


    def recv_frame(self):
        socks = dict(self.bmc_server.poller.poll(ModelTrackingGuiZeromq.sendtimeo))
        if self.bmc_server.sockets["data_in"] in socks and socks[self.bmc_server.sockets["data_in"]] == zmq.POLLIN:
            n_frame = json.loads(self.bmc_server.sockets["data_in"].recv_json())
            return n_frame
        else:
            print('Failed to receive n_frame from client.')
            return None


    def send_init(self,frame_data_mbv):
        try:
            self.bmc_server.sockets["data_out"].send_json(json.dumps(frame_data_mbv))
        except:
            print('Failed to send init to client.')
        self.frame_data = frame_data_mbv


    def send_frame(self,frame_data_mbv):
        try:
            self.bmc_server.sockets["data_out"].send_json(json.dumps(frame_data_mbv))
        except:
            print('Failed to send frame to client.')
            self.frame_data = frame_data_mbv