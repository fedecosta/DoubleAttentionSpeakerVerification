import pickle
import numpy as np
from random import randint, randrange
import os
from torch.utils import data

# TODO understand where is this function used, move to the correct module
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

# TODO understand where is this function used, move to the correct module
# This is exactly the same as Dataset.__normalize
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


    def __len__(self):
        
        # Mandatory torch method

        return self.num_samples


    def normalize(self, features):

        # Cepstral mean normalization
        if self.parameters.normalization == 'cmn':

            # Compute the mean for each frequency band (columns)
            mean = np.mean(features, axis = 0)
            
            # Substract for each column the corresponding column mean
            features = features - mean

        # Cepstral mean and variance normalization
        elif self.parameters.normalization == 'cmvn':

            # Compute the mean for each frequency band (columns)
            mean = np.mean(features, axis = 0)
            
            # Substract for each column the corresponding column mean
            features = features - mean
            
            # Compute the standard deviation for each frequency band (columns)
            std = np.std(features, axis = 0)
            
            # HACK guess this is to avoid zero division overflow
            std = np.where(std > 0.01, std, 1.0)

            # Divide for each column the corresponding column std
            features = features / std

        return features


    def sample_spectogram_window(self, features):

        # Cut the spectrogram with a fixed length at a random start

        file_frames = features.shape[0]
        
        # FIX why this hardcoded 100? 
        # The cutting here is in FRAMES, not secs
        # It would be nice to do the cutting at the feature extractor module
        # It seems that some kind of padding is made with librosa, but it should be done at the feature extractor module also
        sample_size_in_frames = self.parameters.window_size * 100

        # Get a random start point
        index = randint(0, max(0, file_frames - sample_size_in_frames - 1))

        # Generate the index slicing
        a = np.array(range(min(file_frames, int(sample_size_in_frames)))) + index
        
        # Slice the spectrogram
        sliced_spectrogram = features[a,:]

        return sliced_spectrogram


    def get_feature_vector(self, utterance_path):

        # Load the spectrogram saved in pickle format
        with open(utterance_path + '.pickle', 'rb') as pickle_file:
            features_dict = pickle.load(pickle_file)

        features = features_dict["features"]
        features_settings = features_dict["settings"]

        # HACK fix this transpose
        # It seems that the feature extractor's output spectrogram has mel bands as rows
        # Is it possible to do the transpose in that module?
        features = np.transpose(features)
        
        features = self.normalize(features)

        windowedFeatures = self.sample_spectogram_window(features) 

        return windowedFeatures            
     

    def __getitem__(self, index):
        'Generates one sample of data'

        # Mandatory torch method

        # Each utterance_path is like: path label -1
        # TODO seems that -1 is not necessary?
        utterance_tuple = self.utterances_paths[index].strip().split(' ')

        utterance_path = os.path.join(self.parameters.train_data_dir, utterance_tuple[0])
        utterance_label = int(utterance_tuple[1])
        
        return self.get_feature_vector(utterance_path), np.array(utterance_label)

