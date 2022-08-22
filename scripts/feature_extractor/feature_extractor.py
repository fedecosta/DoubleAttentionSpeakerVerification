import argparse
import os
import librosa
import numpy as np
import pickle

# In this particular case we ignore warnings of loading a .m4a audio
# Not a good practice
import warnings
warnings.filterwarnings("ignore")

class FeatureExtractor:

    def __init__(self):
        self.parse_args()

        
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Looks for audio files and extract features. \
                It searches audio files in a paths file and dumps the  \
                extracted features in the same directory in a .pickle format.',
            )


    def add_parser_args(self):

        self.parser.add_argument(
            'audio_paths_file_dir',
            type = str, 
            default = 'scripts/feature_extractor/feature_extractor_paths.lst',
            help = '.lst file path containing the audio files paths we want to extract features from.',
            )

        self.parser.add_argument(
            "--sampling_rate", "-sr", 
            type = int,
            default = 16000,
            help = "Audio sampling rate (in Hz).",
            )

        self.parser.add_argument(
            "--n_fft_secs", 
            type = int,
            default = 0.023,
            help = "Length of the windowed signal after padding with zeros (in seconds).",
            )

        self.parser.add_argument(
            "--window", 
            type = str,
            default = "hamming",
            help = "Windowing function (librosa parameter).",
            )

        self.parser.add_argument(
            "--win_length_secs", 
            type = float,
            default = 0.023,
            help = "(In seconds). Each frame of audio is windowed by window of length win_length_secs and then padded with zeros to match n_fft_secs.",
            )

        self.parser.add_argument(
            "--hop_length_secs", 
            type = float,
            default = 0.010,
            help = "Hop length (in seconds).",
            )

        self.parser.add_argument(
            "--pre_emph_coef", 
            type = float,
            default = 0.97,
            help = "Pre-emphasis coefficient.",
            )

        self.parser.add_argument(
            "--n_mels", 
            type = int,
            default = 80,
            help = "Number of Mel bands to generate.",
            )

        self.parser.add_argument(
            "--verbose", 
            action = "store_true",
            help = "Increase output verbosity.",
            )

        
    def parse_args(self):

        self.initialize_parser()
        self.add_parser_args()
        self.params = self.parser.parse_args()


    def count_input_lines(self):
        
        # Doing this to be able to print progress of processing files
        with open(self.params.audio_paths_file_dir, 'r') as file:
            self.total_lines = sum(1 for line in list(file))
            file.close()


    def generate_log_mel_spectrogram(self, samples, sampling_rate):
        
        # Pre-emphasis
        samples *= 32768 #TODO why this number?
        samples[1:] = samples[1:] - self.params.pre_emph_coef * samples[:-1]
        samples[0] *= (1 - self.params.pre_emph_coef)

        # Short time Fourier Transform
        D = librosa.stft(
            samples, 
            n_fft = int(self.params.n_fft_secs * sampling_rate), 
            hop_length = int(self.params.hop_length_secs * sampling_rate),
            win_length = int(self.params.win_length_secs * sampling_rate), 
            window = self.params.window, 
            center = False,
            )

        magnitudes = np.abs(D)
        low_freq = 0
        high_freq = sampling_rate / 2

        mel_spectrogram = librosa.feature.melspectrogram(
            S = magnitudes, 
            sr = sampling_rate, 
            n_mels = self.params.n_mels, 
            fmin = low_freq, 
            fmax = high_freq, 
            norm = None,
            )

        log_mel_spectrogram = np.log(np.maximum(1, mel_spectrogram))
        
        return mel_spectrogram


    def extract_features(self):

        self.count_input_lines()

        with open(self.params.audio_paths_file_dir, 'r') as file:
        
            print(f"[Feature Extractor] {self.total_lines} audios ready for feature extraction.")

            line_num = 0
            for line in file:

                audio_path = line.replace("\n", "")

                if self.params.verbose: print(f"[Feature Extractor] Processing file {audio_path}...")

                # Load the audio
                
                # TODO give the option to load with sf
                # y, sfreq = sf.read('{}'.format(featureFile[:-1]))
                # y, sfreq = sf.read('{}'.format(featureFile))
                
                samples, sampling_rate = librosa.load(
                    f'{audio_path}',
                    sr = self.params.sampling_rate,
                    mono = True, # converts to mono channel
                    ) 

                assert int(sampling_rate) == int(self.params.sampling_rate)

                
                # Create the log mel spectrogram
                log_mel_spectrogram = self.generate_log_mel_spectrogram(
                    samples, 
                    self.params.sampling_rate,
                    )

                # Dump the spectrogram
                # TODO fix this hardcoded -4
                with open(f'{line[:-4]}.pickle', 'wb') as handle:
                    pickle.dump(log_mel_spectrogram, handle)

                if self.params.verbose: print(f"[Feature Extractor] File processed. Dumped pickle in {line[:-4]}.pickle")
                
                progress_pctg = line_num / self.total_lines * 100
                print(f"[Feature Extractor] {progress_pctg:.2f}% audios processed...")
                
                line_num = line_num + 1

            print(f"[Feature Extractor] All audios processed!")
    
    
    def main(self):

        self.extract_features()
        

if __name__=="__main__":

    instance = FeatureExtractor()
    instance.main()
    