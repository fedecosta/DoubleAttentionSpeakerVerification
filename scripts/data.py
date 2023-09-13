import pickle
import numpy as np
from random import randint
import os
import torch
from torch.utils import data

from utils import scoreCosineDistance

def feature_reader(featurePath, VAD = None):

    with open(featurePath,'rb') as pickleFile:
        features_dict = pickle.load(pickleFile)

        features = features_dict["features"]
        features_settings = features_dict["settings"]

        if VAD is not None:
            filtered_features = VAD.filter(features)
        else:
            filtered_features = features

    if filtered_features.shape[1]>0.:
        return np.transpose(filtered_features)
    else:
        return np.transpose(features)


def normalize_features(features, normalization = 'cmn'):

        # Cepstral mean normalization
        if normalization == 'cmn':

            # Compute the mean for each frequency band (columns)
            mean = np.mean(features, axis = 0)
            
            # Substract for each column the corresponding column mean
            features = features - mean

        # Cepstral mean and variance normalization
        elif normalization == 'cmvn':

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

        # Cepstral mean and variance normalization and range normalization between -1 and 1
        elif normalization == 'full':

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

            # Range between -1 and 1
            features = features / np.max(np.abs(features), axis = 0)

        return features


class TrainDataset(data.Dataset):

    def __init__(self, utterances_paths, parameters):
        
        self.utterances_paths = utterances_paths
        self.parameters = parameters
        self.num_samples = len(utterances_paths)


    def normalize(self, features):

        normalized_features = normalize_features(features, normalization = self.parameters.normalization)

        return normalized_features


    def sample_spectogram_crop(self, features):

        # Cut the spectrogram with a fixed length at a random start

        file_frames = features.shape[0]
        
        # Get a random start point
        index = randint(0, max(0, file_frames - self.parameters.random_crop_frames - 1))
        
        # Generate the index slicing
        a = np.array(range(min(file_frames, int(self.parameters.random_crop_frames)))) + index
        
        # Slice the spectrogram
        sliced_spectrogram = features[a,:]

        return sliced_spectrogram


    def get_feature_vector(self, utterance_path):

        # Load the spectrogram saved in pickle format
        with open(utterance_path, 'rb') as pickle_file:
            features_dict = pickle.load(pickle_file)

        features = features_dict["features"]
        # features_settings = features_dict["settings"]

        # HACK fix this transpose
        # It seems that the feature extractor's output spectrogram has mel bands as rows
        # Is it possible to do the transpose in that module?
        features = np.transpose(features)
        
        features = self.normalize(features)

        random_crop = self.sample_spectogram_crop(features) 

        return random_crop            
     

    def __getitem__(self, index):
        'Generates one sample of data'

        # Mandatory torch method

        # Each utterance_path is like: path label -1
        # TODO seems that -1 is not necessary?
        utterance_tuple = self.utterances_paths[index].strip().split(' ')

        utterance_path = utterance_tuple[0]
        utterance_label = int(utterance_tuple[1])

        features = self.get_feature_vector(utterance_path)
        labels = np.array(utterance_label)
        
        return features, labels


    def __len__(self):
        
        # Mandatory torch method

        return self.num_samples


class TestDataset(data.Dataset):

    def __init__(self, clients_utterances_paths, impostors_utterances_paths, train_parameters, random_crop_frames, evaluation_type):

        self.clients_utterances_paths = clients_utterances_paths
        self.impostors_utterances_paths = impostors_utterances_paths
        self.parameters = train_parameters
        self.random_crop_frames = random_crop_frames
        self.evaluation_type = evaluation_type
        self.format_input_paths()
        self.num_samples = len(self.formatted_utterances_paths)


    def format_input_paths(self):

        # We label the trial with 1 for clients and 0 for impostors
        formatted_clients = [f"{trial} 1" for trial in self.clients_utterances_paths]
        formatted_impostors = [f"{trial} 0" for trial in self.impostors_utterances_paths]
        self.formatted_utterances_paths = formatted_clients + formatted_impostors


    def normalize(self, features):

        normalized_features = normalize_features(features, normalization = self.parameters.normalization)

        return normalized_features


    def sample_spectogram_crop(self, features):

        # Cut the spectrogram with a fixed length at a random start

        file_frames = features.shape[0]
        
        # Get a random start point
        index = randint(0, max(0, file_frames - self.random_crop_frames - 1))

        # Generate the index slicing
        a = np.array(range(min(file_frames, int(self.random_crop_frames)))) + index
        
        # Slice the spectrogram
        sliced_spectrogram = features[a,:]

        return sliced_spectrogram


    def get_feature_vector(self, utterance_path):

        # Load the spectrogram saved in pickle format
        with open(utterance_path, 'rb') as pickle_file:
            features_dict = pickle.load(pickle_file)

        features = features_dict["features"]
        features_settings = features_dict["settings"]

        # HACK fix this transpose
        # It seems that the feature extractor's output spectrogram has mel bands as rows
        # Is it possible to do the transpose in that module?
        features = np.transpose(features)
        
        features = self.normalize(features)

        if self.evaluation_type == "random_crop":
            features = self.sample_spectogram_crop(features)
        elif self.evaluation_type == "total_length":
            self.random_crop_size = 0
    
        return features   


    def __getitem__(self, index):
        'Generates one sample of data'

        # Mandatory torch method

        # Each line of the dataset is like: speaker_1_path speaker_2_path label \n
        utterance_tuple = self.formatted_utterances_paths[index].strip().replace('\n', '').split(' ')
        speaker_1_utterance_path = utterance_tuple[0]
        speaker_2_utterance_path = utterance_tuple[1]

        utterance_label = int(utterance_tuple[2])

        speaker_1_features = self.get_feature_vector(speaker_1_utterance_path)
        speaker_2_features = self.get_feature_vector(speaker_2_utterance_path)

        # Letting this comment in case wanting to develop a different length padding version
        #speaker_1_features_length = speaker_1_features.shape[0]
        #speaker_2_features_length = speaker_2_features.shape[0]
        
        return (
            torch.from_numpy(speaker_1_features), 
            torch.from_numpy(speaker_2_features), 
            utterance_label,
            )


    def __len__(self):
        
        # Mandatory torch method

        return self.num_samples