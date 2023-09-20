import pickle
import numpy as np
from random import randint
import torch
from torch.utils import data
import torchaudio
import logging

logger = logging.getLogger(__name__)


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

    features[features > 1] = 1
    features = torch.log(features)
    
    # Cepstral mean normalization
    if normalization == 'cmn':

        # Compute the mean for each frequency band (not for eache frame)
        mean = torch.mean(features, dim = 0)
        
        # Substract for each column the corresponding column mean
        features = features - mean

    # Cepstral mean and variance normalization
    elif normalization == 'cmvn':

        # Compute the mean for each frequency band (not for eache frame)
        mean = torch.mean(features, dim = 0)
        
        # Substract for each column the corresponding column mean
        features = features - mean
        
        # Compute the standard deviation for each frequency band (columns)
        std = torch.std(features, dim = 0)
        
        # HACK guess this is to avoid zero division overflow
        std = torch.where(std > 0.01, std, 1.0)

        # Divide for each column the corresponding column std
        features = features / std

    # Cepstral mean and variance normalization and range normalization between -1 and 1
    elif normalization == 'full':

        # Compute the mean for each frequency band (not for eache frame)
        mean = torch.mean(features, dim = 0)
        
        # Substract for each column the corresponding column mean
        features = features - mean
        
        # Compute the standard deviation for each frequency band (columns)
        std = torch.std(features, dim = 0)
        
        # HACK guess this is to avoid zero division overflow
        std = torch.where(std > 0.01, std, 1.0)

        # Divide for each column the corresponding column std
        features = features / std

        # Range between -1 and 1
        maxs, _ = torch.max(torch.abs(features), dim = 1)
        features = features.transpose(0, 1) / maxs
        features = features.transpose(0, 1)

    return features


def sample_random_waveform_crop(waveform, sample_rate, seconds_to_crop):

        '''Cut the waveform with a fixed length at a random start'''

        samples_to_crop = seconds_to_crop * sample_rate
        waveform_total_samples = waveform.shape[1]
        
        # Get a random start point
        index = randint(0, max(0, waveform_total_samples - samples_to_crop - 1))
        
        # Generate the index slicing
        selected_indexes = np.array(range(min(waveform_total_samples, int(samples_to_crop)))) + index
        
        # Slice the waveform
        sliced_waveform = waveform[:, selected_indexes]

        return sliced_waveform
    

class TrainDataset(data.Dataset):

    def __init__(self, utterances_paths, parameters):
        
        self.utterances_paths = utterances_paths
        self.parameters = parameters
        self.num_samples = len(utterances_paths)
        self.init_spectrogram_generator(
            sample_rate = self.parameters.sample_rate,
            n_fft_secs = self.parameters.n_fft_secs,
            win_length_secs = self.parameters.win_length_secs,
            hop_length_secs = self.parameters.hop_length_secs,
            n_mels = self.parameters.n_mels,
        )


    def init_spectrogram_generator(self, sample_rate, n_fft_secs, win_length_secs, hop_length_secs, n_mels):

        # TODO unfix all hardcoded parameters

        self.spectrogram_generator = torchaudio.transforms.MelSpectrogram(
            n_fft = int(n_fft_secs * sample_rate),
            win_length = int(sample_rate * win_length_secs),
            hop_length = int(sample_rate * hop_length_secs),
            n_mels = n_mels,
            mel_scale = "slaney",
            window_fn = torch.hamming_window,
            f_max = sample_rate // 2,
            center = False,
            normalized = False,
            norm = "slaney",
        )


    def normalize_features(self, features, normalization):

        features = normalize_features(features, normalization = normalization)

        return features
    

    def generate_spectrogram(self, waveform):
        
        spectrogram = self.spectrogram_generator(waveform).squeeze(0)
        
        return spectrogram


    def get_feature_vector(self, waveform):

        if False:
            augmented_waveform = self.augment(waveform)
            features = self.generate_spectrogram(augmented_waveform)
        else:
            features = self.generate_spectrogram(waveform)

        # HACK It seems that the output spectrogram has mel bands as rows.
        # is this transpose neccesary?
        features = features.transpose(0, 1)

        return features            
     

    def normalize_waveform(self, waveform, preemphasis_coefficient):

        # TODO analyze is preemphasis helps to achieve better results
        
        if preemphasis_coefficient > 0:
            waveform = waveform * 32768
            waveform[:, 1:] = waveform[:, 1:] - preemphasis_coefficient * waveform[:, :-1]
            waveform[:, 0] = waveform[:, 0] * (1 - preemphasis_coefficient)

        return waveform


    def sample_random_waveform_crop(self, waveform, sample_rate, seconds_to_crop):

        '''Cut the waveform with a fixed length at a random start'''

        sliced_waveform = sample_random_waveform_crop(waveform, sample_rate, seconds_to_crop)

        return sliced_waveform


    def __getitem__(self, index):

        '''Mandatory torch method, generates one sample of data.'''

        # Each utterance_path is like: path label -1
        # TODO seems that -1 is not necessary?
        utterance_tuple = self.utterances_paths[index].strip().split(' ')
        utterance_path = utterance_tuple[0]
        utterance_label = int(utterance_tuple[1])

        # Load the audio into waveform
        waveform, original_sample_rate = torchaudio.load(
            utterance_path)

        # Resample to desired sample_rate
        waveform = torchaudio.functional.resample(
            waveform, 
            orig_freq = original_sample_rate, 
            new_freq = self.parameters.sample_rate,
            )
        
        # Sample a random crop
        waveform = self.sample_random_waveform_crop(
            waveform,
            sample_rate = self.parameters.sample_rate,
            seconds_to_crop = self.parameters.random_crop_secs,
            )

        waveform = self.normalize_waveform(
            waveform,
            preemphasis_coefficient = self.parameters.pre_emph_coef,
            )

        features = self.get_feature_vector(waveform)

        features = self.normalize_features(
            features,
            normalization = self.parameters.normalization)

        labels = np.array(utterance_label)
        
        return features, labels


    def __len__(self):
        
        # Mandatory torch method

        return self.num_samples


class TestDataset(data.Dataset):

    def __init__(self, clients_utterances_paths, impostors_utterances_paths, train_parameters, random_crop_secs, evaluation_type):

        self.clients_utterances_paths = clients_utterances_paths
        self.impostors_utterances_paths = impostors_utterances_paths
        self.parameters = train_parameters
        self.random_crop_secs = random_crop_secs
        self.evaluation_type = evaluation_type
        self.format_input_paths()
        self.num_samples = len(self.formatted_utterances_paths)
        self.init_spectrogram_generator(
            sample_rate = self.parameters.sample_rate,
            n_fft_secs = self.parameters.n_fft_secs,
            win_length_secs = self.parameters.win_length_secs,
            hop_length_secs = self.parameters.hop_length_secs,
            n_mels = self.parameters.n_mels,
        )


    def format_input_paths(self):

        # We label the trial with 1 for clients and 0 for impostors
        formatted_clients = [f"{trial} 1" for trial in self.clients_utterances_paths]
        formatted_impostors = [f"{trial} 0" for trial in self.impostors_utterances_paths]
        self.formatted_utterances_paths = formatted_clients + formatted_impostors


    def normalize(self, features):

        normalized_features = normalize_features(features, normalization = self.parameters.normalization)

        return normalized_features


    def sample_random_waveform_crop(self, waveform, sample_rate, seconds_to_crop):

        '''Cut the waveform with a fixed length at a random start'''

        sliced_waveform = sample_random_waveform_crop(waveform, sample_rate, seconds_to_crop)

        return sliced_waveform


    def init_spectrogram_generator(self, sample_rate, n_fft_secs, win_length_secs, hop_length_secs, n_mels):

        # TODO unfix all hardcoded parameters

        self.spectrogram_generator = torchaudio.transforms.MelSpectrogram(
            n_fft = int(n_fft_secs * sample_rate),
            win_length = int(sample_rate * win_length_secs),
            hop_length = int(sample_rate * hop_length_secs),
            n_mels = n_mels,
            mel_scale = "slaney",
            window_fn = torch.hamming_window,
            f_max = sample_rate // 2,
            center = False,
            normalized = False,
            norm = "slaney",
        )


    def normalize_features(self, features, normalization):

        features = normalize_features(features, normalization = normalization)

        return features


    def generate_spectrogram(self, waveform):
        
        spectrogram = self.spectrogram_generator(waveform).squeeze(0)
        
        return spectrogram


    def get_feature_vector(self, waveform):

        features = self.generate_spectrogram(waveform)

        # HACK It seems that the output spectrogram has mel bands as rows.
        # is this transpose neccesary?
        features = features.transpose(0, 1)

        return features  


    def __getitem__(self, index):
        'Generates one sample of data'

        # Mandatory torch method

        # Each line of the dataset is like: speaker_1_path speaker_2_path label \n
        utterance_tuple = self.formatted_utterances_paths[index].strip().replace('\n', '').split(' ')
        speaker_1_utterance_path = utterance_tuple[0]
        speaker_2_utterance_path = utterance_tuple[1]

        utterance_label = int(utterance_tuple[2])

        # Load the audio into waveform
        waveform_1, original_sample_rate_1 = torchaudio.load(
            speaker_1_utterance_path)

        waveform_2, original_sample_rate_2 = torchaudio.load(
            speaker_2_utterance_path)

        # Resample to desired sample_rate
        waveform_1 = torchaudio.functional.resample(
            waveform_1, 
            orig_freq = original_sample_rate_1, 
            new_freq = self.parameters.sample_rate,
            )

        waveform_2 = torchaudio.functional.resample(
            waveform_2, 
            orig_freq = original_sample_rate_2, 
            new_freq = self.parameters.sample_rate,
            )
        
        # Sample a random crop
        if self.evaluation_type == "random_crop":

            waveform_1 = self.sample_random_waveform_crop(
                waveform_1,
                sample_rate = self.parameters.sample_rate,
                seconds_to_crop = self.parameters.random_crop_secs,
            )

            waveform_2 = self.sample_random_waveform_crop(
                waveform_2,
                sample_rate = self.parameters.sample_rate,
                seconds_to_crop = self.parameters.random_crop_secs,
            )

        elif self.evaluation_type == "total_length":
            
            self.random_crop_size = 0
        
        speaker_1_features = self.get_feature_vector(waveform_1)
        speaker_2_features = self.get_feature_vector(waveform_2)

        speaker_1_features = self.normalize_features(
            speaker_1_features,
            normalization = self.parameters.normalization)

        speaker_2_features = self.normalize_features(
            speaker_2_features,
            normalization = self.parameters.normalization)

        # Letting this comment in case wanting to develop a different length padding version
        #speaker_1_features_length = speaker_1_features.shape[0]
        #speaker_2_features_length = speaker_2_features.shape[0]
        
        return (
            speaker_1_features, 
            speaker_2_features, 
            utterance_label,
            )


    def __len__(self):
        
        # Mandatory torch method

        return self.num_samples