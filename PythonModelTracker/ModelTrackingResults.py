import json
import copy


class ModelTrackingResults:
    def __init__(self,did=None):
        self.did = did
        self.models = []
        self.states = {}
        self.landmark_names = {}
        self.landmarks = {}

    def has_state(self, frame,model_name):
        if frame in self.states:
            if model_name in self.states[frame]:
                return True
        return False

    def has_landmarks(self, frame,model_name):
        if frame in self.landmarks:
            if model_name in self.landmarks[frame]:
                return True
        return False

    def get_model_states(self, model_name):
        #min_frame, max_frame = self.get_limits()
        model_states = {}
        for f in self.states:
            #print(f, model_name)
            if self.has_state(f, model_name):
                model_states[f] = self.states[f][model_name]
        return model_states

    def get_model_landmarks(self, model_name):
        #min_frame, max_frame = self.get_limits()
        model_landmarks = {}
        if model_name in self.landmark_names:
            landmark_names = self.landmark_names[model_name]
            for f in self.landmarks:
                #print(f, model_name)
                if self.has_landmarks(f, model_name):
                    model_landmarks[f] = self.landmarks[f][model_name]
        else: landmark_names = []
        return landmark_names,model_landmarks


    def get_limits(self):
        min_frame = min(self.states)
        max_frame = max(self.states)
        return (min_frame, max_frame)



    def add(self,frame,model_name,state):
        if model_name not in self.models: self.models.append(model_name)
        if frame not in self.states: self.states[frame] = {}
        state_list = [i for i in state]
        self.states[frame][model_name] = state_list

    def add_landmark_names(self,model_name, landmark_names):
        self.landmark_names[model_name] = landmark_names

    def add_landmarks(self,frame,model_name,landmarks):
        if model_name not in self.models: self.models.append(model_name)
        assert len(landmarks) == len(self.landmark_names[model_name])
        if frame not in self.landmarks: self.landmarks[frame] = {}
        landmark_list = [[i.x, i.y, i.z] for i in landmarks]
        self.landmarks[frame][model_name] = landmark_list

    def save(self,filename):
        if len(filename)>3:
            json_target = open(filename, 'w')
            json.dump((self.__dict__), json_target, indent=2)
            print('Saving results to: <{}>'.format(filename))
        else:
            print('Cannot save results, invalid filename: <{}>'.format(filename) )


    def load(self, filename):
        json_target = open(filename, 'r')
        self.__dict__ = json.load(json_target)

        states = copy.deepcopy(self.states)
        self.states = {}
        for s in states:
            self.states[int(s)] = states[s]

        landmarks = copy.deepcopy(self.landmarks)
        self.landmarks = {}
        for l in landmarks:
            self.landmarks[int(l)] = landmarks[l]
        #print(self.did, self.models)
         #self.datasets_xml, self.did, self.models, self.states