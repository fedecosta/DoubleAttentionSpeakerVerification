import pickle
import numpy as np
from random import randint, randrange
from torch.utils import data
import soundfile as sf

def featureReader(featurePath, VAD = None):

    with open(featurePath,'rb') as pickleFile:
        features = pickle.load(pickleFile)
        if VAD is not None:
            filtered_features = VAD.filter(features)
        else:
            filtered_features = features

    if filtered_features.shape[1]>0.:
        return np.transpose(filtered_features)
    else:
        return np.transpose(features)

def normalizeFeatures(features, normalization = 'cmn'):

    mean = np.mean(features, axis=0)
    features -= mean 
    if normalization=='cmn':
       return features
    if normalization=='cmvn':
        std = np.std(features, axis=0)
        std = np.where(std>0.01,std,1.0)
        return features/std

class Dataset(data.Dataset):

    def __init__(self, utterances_paths, parameters):
        'Initialization'
        self.utterances_paths = utterances_paths
        self.parameters = parameters
        self.num_samples = len(utterances_paths)

    def __normalize(self, features):

        mean = np.mean(features, axis=0)
        features -= mean 
        if self.parameters.normalization=='cmn':
            return features
        if self.parameters.normalization=='cmvn':
            std = np.std(features, axis=0)
            std = np.where(std>0.01,std,1.0)
            return features/std

    def __sampleSpectogramWindow(self, features):

        # Cut the spectrogram with a fixed length at a random start

        file_size = features.shape[0]
        # TODO why this hardcoded 100?
        windowSizeInFrames = self.parameters.window_size * 100
        index = randint(0, max(0, file_size - windowSizeInFrames - 1))
        a = np.array(range(min(file_size, int(windowSizeInFrames)))) + index
        return features[a,:]

    def __getFeatureVector(self, utterance_path):

        # Load the spectrogram saved in pickle format
        with open(utterance_path + '.pickle', 'rb') as pickle_file:
            features = pickle.load(pickle_file)

        windowedFeatures = self.__sampleSpectogramWindow(self.__normalize(np.transpose(features))) # TODO fix this transpose
        return windowedFeatures            
     
    def __len__(self):

        # Mandatory torch method
        return self.num_samples

    def __getitem__(self, index):
        'Generates one sample of data'

        # Mandatory torch method

        # Each utterance_path is like: path label -1
        # TODO seems that -1 is not necessary?
        utterance_tuple = self.utterances_paths[index].strip().split(' ')
        utterance_path = self.parameters.train_data_dir + '/' + utterance_tuple[0]
        utterance_label = int(utterance_tuple[1])
        
        return self.__getFeatureVector(utterance_path), np.array(utterance_label)

