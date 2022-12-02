import argparse
import os
import librosa
import numpy as np
import pickle
import datetime

from settings import FEATURE_EXTRACTOR_DEFAULT_SETTINGS

# Set logging config
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)

# Init a wandb project
import wandb
run = wandb.init(project = "speaker_verification", job_type = "dataset")

# In this particular case we ignore warnings of loading a .m4a audio
# Not a good practice
import warnings
warnings.filterwarnings("ignore")

# TODO add the usage instructions in README.md


class FeatureExtractor:

    def __init__(self, params):
        self.params = params
        self.set_other_params()
        self.set_log_file_handler()
        self.config_wandb()


    def set_other_params(self):

        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        self.start_datetime = self.start_datetime.replace("-", "_").replace(" ", "_").replace(":", "_")

        self.dataset_id = f"{self.start_datetime}_{wandb.run.id}_{wandb.run.name}"
        self.params.dump_folder_name = os.path.join(self.params.dump_folder_name, self.dataset_id)

        self.params.audio_paths_file_path = os.path.join(
            self.params.audio_paths_file_folder, 
            self.params.audio_paths_file_name,
            )


    def set_log_file_handler(self):

        # Set a logging file handler
        if not os.path.exists(self.params.log_file_folder):
            os.makedirs(self.params.log_file_folder)
        logger_file_path = os.path.join(self.params.log_file_folder, self.params.log_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
        logger_file_handler.setLevel(logging.INFO) # TODO set the file handler level as a input param
        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler)


    def config_wandb(self):

        # 1 - Save the params
        self.wandb_config = vars(self.params)

        # 2 - Update the wandb config
        wandb.config.update(self.wandb_config)


    def count_input_lines(self):
        
        # Doing this to be able to print progress of processing files
        with open(self.params.audio_paths_file_path, 'r') as file:
            self.total_lines = sum(1 for line in list(file))
            file.close()


    def generate_log_mel_spectrogram(self, samples, sampling_rate):
        
        # Pre-emphasis
        samples *= 32768 # HACK why this number?
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

        # TODO this array has to be trasposed in later methods. why not traspose now?
        log_mel_spectrogram = np.log(np.maximum(1, mel_spectrogram))
        
        return log_mel_spectrogram


    def extract_features(self, audio_path):

        # Load the audio
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

        return log_mel_spectrogram


    def extract_all_features(self):

        # If not exists, create the dump folder
        if not os.path.exists(self.params.dump_folder_name):
            os.makedirs(self.params.dump_folder_name)

        with open(self.params.audio_paths_file_path, 'r') as file:
        
            logger.info(f"{self.total_lines} audios ready for feature extraction.")

            line_num = 0
            progress_pctg_to_print = 0
            for line in file:

                # remove end of line
                audio_path = line.replace("\n", "")
                # Prepend the optional folder directory
                load_audio_path = os.path.join(self.params.prepend_directory, audio_path)

                if self.params.verbose: logger.info(f"Processing file {load_audio_path}...")

                file_dump_path = '.'.join(line.split(".")[:-1]) # remove the file extension
                file_dump_path = file_dump_path + ".pickle" # add the pickle extension
                file_dump_path = os.path.join(self.params.dump_folder_name, file_dump_path)

                # If not exists, create the dump folder (specific to that speaker and interview)
                file_dump_folder = '/'.join(file_dump_path.split("/")[:-1])
                if not os.path.exists(file_dump_folder):
                    os.makedirs(file_dump_folder)

                if (self.params.overwrite == True) or (self.params.overwrite == False and not os.path.exists(file_dump_path)):
                    
                    log_mel_spectrogram = self.extract_features(load_audio_path)

                    info_dict = {}
                    info_dict["features"] = log_mel_spectrogram
                    info_dict["settings"] = self.params
                    
                    # Dump the dict
                    with open(file_dump_path, 'wb') as handle:
                        pickle.dump(info_dict, handle)

                    if self.params.verbose: logger.info(f"File processed. Dumpled pickle in {file_dump_path}")
                    
                progress_pctg = line_num / self.total_lines * 100
                if progress_pctg >=  progress_pctg_to_print:
                    logger.info(f"{progress_pctg:.0f}% audios processed...")
                    progress_pctg_to_print = progress_pctg_to_print + 1
                
                # A flush print have some issues with large datasets
                # print(f"\r {progress_pctg:.1f}% audios processed...", end = '', flush = True)

                line_num = line_num + 1

            logger.info(f"All audios processed!")


    def save_artifact(self):

        # Save dataset as a wandb artifact

        # Define the artifact
        dataset_artifact = wandb.Artifact(
            name = self.dataset_id,
            type = "dataset",
            description = "dataset of spectrograms",
            metadata = self.wandb_config,
        )

        # Add folder directory
        dataset_artifact.add_dir(self.params.dump_folder_name)

        # Log the artifact
        run.log_artifact(dataset_artifact)


    def main(self):

        self.count_input_lines()
        self.extract_all_features()
        self.save_artifact()

               
class ArgsParser:

    def __init__(self):
        self.initialize_parser()

    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Looks for audio files and extract features. \
                It searches audio files in a paths file and dumps the  \
                extracted features in a .pickle file in the same directory.',
            )

    def add_parser_args(self):

        self.parser.add_argument(
            '--audio_paths_file_folder',
            type = str, 
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['audio_paths_file_folder'],
            help = 'Folder containing the .lst file with the audio files paths we want to extract features from.',
            )

        self.parser.add_argument(
            '--audio_paths_file_name',
            type = str, 
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['audio_paths_file_name'],
            help = '.lst file name containing the audio files paths we want to extract features from.',
            )

        self.parser.add_argument(
            '--prepend_directory',
            type = str, 
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['prepend_directory'],
            help = 'Optional folder directory you want to prepend to each line of audio_paths_file_name.',
            )

        self.parser.add_argument(
            '--dump_folder_name',
            type = str, 
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['dump_folder_name'],
            help = 'Folder directory to dump the .pickle files.',
            )

        self.parser.add_argument(
            '--log_file_folder',
            type = str, 
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['log_file_folder'],
            help = 'Name of folder that will contain the log file.',
            )
        
        self.parser.add_argument(
            '--log_file_name',
            type = str, 
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['log_file_name'],
            help = 'Name of the log file.',
            )

        self.parser.add_argument(
            "--sampling_rate", "-sr", 
            type = int,
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['sampling_rate'],
            help = "Audio sampling rate (in Hz).",
            )

        self.parser.add_argument(
            "--n_fft_secs", 
            type = float,
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['n_fft_secs'],
            help = "Length of the windowed signal after padding with zeros (in seconds).\
                int(n_fft_secs x sampling_rate) should be a power of 2 for better performace,\
                 and n_fft_secs must be greater or equal than win_length_secs.",
            )

        self.parser.add_argument(
            "--window", 
            type = str,
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['window'],
            help = "Windowing function (librosa parameter).",
            )

        self.parser.add_argument(
            "--win_length_secs", 
            type = float,
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['win_length_secs'],
            help = "(In seconds). Each frame of audio is windowed by window of length win_length_secs and then padded with zeros to match n_fft_secs.",
            )

        self.parser.add_argument(
            "--hop_length_secs", 
            type = float,
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['hop_length_secs'],
            help = "Hop length (in seconds).",
            )

        self.parser.add_argument(
            "--pre_emph_coef", 
            type = float,
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['pre_emph_coef'],
            help = "Pre-emphasis coefficient.",
            )

        self.parser.add_argument(
            "--n_mels", 
            type = int,
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['n_mels'],
            help = "Number of Mel bands to generate.",
            )

        self.parser.add_argument(
            "--overwrite", 
            action = argparse.BooleanOptionalAction,
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['overwrite'],
            help = "True if you want to overwrite already feature extracted audios.",
            )

        self.parser.add_argument(
            "--verbose", 
            action = argparse.BooleanOptionalAction,
            default = FEATURE_EXTRACTOR_DEFAULT_SETTINGS['verbose'],
            help = "Increase output verbosity.",
            )


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()


if __name__=="__main__":

    args_parser = ArgsParser()
    args_parser.main()
    parameters = args_parser.arguments

    feature_extractor = FeatureExtractor(parameters)
    feature_extractor.main()
    