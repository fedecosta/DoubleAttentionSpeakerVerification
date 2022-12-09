import argparse
import os
import librosa
import numpy as np
import pickle
import datetime

from utils import get_memory_info

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
run = wandb.init(project = "speaker_verification_labels", job_type = "labels")

# In this particular case we ignore warnings of loading a .m4a audio
# Not a good practice
import warnings
warnings.filterwarnings("ignore")

# TODO add the usage instructions in README.md

class LabelsVersioner:

    def __init__(self, input_params):

        self.params = input_params
        self.set_other_params()
        self.set_log_file_handler()

    
    def set_other_params(self):

        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        self.start_datetime = self.start_datetime.replace("-", "_").replace(" ", "_").replace(":", "_")

        self.labels_id = f"{self.start_datetime}_{wandb.run.id}_{wandb.run.name}"
        self.dump_folder = os.path.join(self.params.dump_folder_name, self.labels_id)
        self.params.log_file_name = f"{self.labels_id}.log"

        self.params.labels_file_path = os.path.join(
            self.params.labels_file_folder, 
            self.params.labels_file_name,
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


    def info_mem(self):
        cpu_available_pctg, gpu_free = get_memory_info()
        logger.info(f"CPU available {cpu_available_pctg:.2f}% - GPU free {gpu_free}")


    def load_labels(self):

        logger.info(f"Loading labels from {self.params.labels_file_path}...")

        with open(self.params.labels_file_path, 'r') as data_labels_file:
            self.labels = data_labels_file.readlines()

        logger.info(f"Done!")


    def format_audio_path(self, audio_path):

        # we need to remove the first "/" to join paths
        if audio_path[0] == "/":
            audio_path = audio_path[1:]

        # remove the file extension, if has
        if len(audio_path.split(".")) > 1:
            audio_path = '.'.join(audio_path.split(".")[:-1]) 

        # We prepend prepend_directory to the paths and the file extension
        data_founded = False
        for dir in self.params.prepend_directory:
            
            if data_founded == False:

                for audio_format in ['wav', 'm4a']:

                    audio_file = f"{audio_path}.{audio_format}"
                    complete_audio_file_path = os.path.join(dir, audio_file)

                    if os.path.exists(complete_audio_file_path):
                        data_founded = True
                        break
        
        assert data_founded, f"{audio_path} not founded."

        return complete_audio_file_path


    def get_audio_paths(self):

        logger.info(f"Getting audio paths...")

        self.audio_paths = []
        total_labels = len(self.labels)
        progress_pctg_to_print = 0
        for index, label in enumerate(self.labels):
            
            label = label.replace('\n', '')
            label_chunks = label.split(' ')

            if len(label_chunks) == 3:

                # Training labels:
                # label is of the form '/speaker/interview/file speaker_number -1'
                audio_path = label_chunks[0]
                audio_path = self.format_audio_path(audio_path)
                self.audio_paths.append(audio_path)

            elif len(label_chunks) == 2:

                # Validation or Test labels
                # label is of the form '/speaker/interview/file /speaker/interview/file'
                audio_path_1, audio_path_2 = label_chunks
                audio_path_1 = self.format_audio_path(audio_path_1)
                audio_path_2 = self.format_audio_path(audio_path_2)
                self.audio_paths.append(audio_path_1)
                self.audio_paths.append(audio_path_2)

            else:

                assert False, f"{label} has a not expected structure."
            
            progress_pctg = index / total_labels * 100
            if progress_pctg >=  progress_pctg_to_print:
                logger.info(f"{progress_pctg:.0f}% paths processed...")
                progress_pctg_to_print = progress_pctg_to_print + 1
                self.info_mem()


    def get_dataset_statistics(self, get_num_files = True, get_num_speakers = True, get_duration = True):

        logger.info(f"Getting labels statistics...")

        # num files
        if get_num_files:
            self.params.num_files = len(self.audio_paths)

            logger.info(f"Number of files: {self.params.num_files}")

        
        # num speakers
        if get_num_speakers:
            logger.info(f"Calculating number of speakers...")
            progress_pctg_to_print = 0
            speakers_set = set()
            for index, audio_path in enumerate(self.audio_paths):
                speaker_chunk = [chunk for chunk in audio_path.split("/") if chunk.startswith("id")]
                # Only consider directories with /id.../
                if len(speaker_chunk) > 0: 
                    speaker_label = speaker_chunk[0]
                    speakers_set.add(speaker_label)

                progress_pctg = index / self.params.num_files * 100
                if progress_pctg >=  progress_pctg_to_print:
                    logger.info(f"{progress_pctg:.0f}% audios processed...")
                    progress_pctg_to_print = progress_pctg_to_print + 1
                    self.info_mem()

            self.params.number_of_speakers = len(speakers_set)

            logger.info(f"Number of speakers: {self.params.number_of_speakers}")

        # audio duration in hours
        if get_duration:
            logger.info(f"Estimating duration of audios...")
            self.params.total_duration_hours = 0
            progress_pctg_to_print = 0
            for index, audio_path in enumerate(self.audio_paths):
                
                audio_duration = librosa.get_duration(filename = audio_path) / 3600
                self.params.total_duration_hours = self.params.total_duration_hours + audio_duration

                progress_pctg = index / self.params.num_files * 100
                if progress_pctg >=  progress_pctg_to_print:
                    logger.info(f"{progress_pctg:.0f}% audios processed...")
                    progress_pctg_to_print = progress_pctg_to_print + 1
                    self.info_mem()

            logger.info(f"Total duration of audios (in hours): {self.params.total_duration_hours}")

            
    def get_labels_info(self):

        logger.info(f"Getting labels overall info...")

        self.get_audio_paths()
        self.get_dataset_statistics(
            get_num_files = True, 
            get_num_speakers = True, 
            get_duration = self.params.get_duration,
            )

        logger.info(f"Done.")

        
    def dump_labels(self):

        if not os.path.exists(self.dump_folder):
            os.makedirs(self.dump_folder)
            
        dump_path = os.path.join(self.dump_folder, self.params.labels_file_name)

        logger.info(f"Dumping labels into {dump_path}...")

        with open(dump_path, 'w') as file:
            for line in self.labels: 
                file.write(line)
            file.close()

        logger.info(f"Done!")

    
    def config_wandb(self):

        # 1 - Save the params
        self.wandb_config = vars(self.params)

        # 2 - Update the wandb config
        wandb.config.update(self.wandb_config)

    
    def save_artifact(self):

        # Save labels file as a wandb artifact

        logger.info(f"Saving labels as wandb artifact...")

        # Update and save config parameters
        self.config_wandb()

        # Define the artifact
        artifact = wandb.Artifact(
            name = self.labels_id,
            type = "labels",
            description = "list of labels",
            metadata = self.wandb_config,
        )

        # Add folder directory
        artifact.add_dir(self.dump_folder)

        # Log the artifact
        run.log_artifact(artifact)

        logger.info(f"Done.")

    
    def main(self):

        self.load_labels()
        self.get_labels_info()
        self.dump_labels()
        self.save_artifact()
        
        
class ArgsParser:

    def __init__(self):
        self.initialize_parser()

    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = '.',
            )


    def add_parser_args(self):

        self.parser.add_argument(
            '--labels_file_folder',
            type = str, 
            help = 'Folder containing the .ndx file with the labels.',
            )

        self.parser.add_argument(
            '--labels_file_name',
            type = str, 
            help = '.ndx file name containing the labels.',
            )        

        self.parser.add_argument(
            '--prepend_directory',
            action = 'append',
            type = str, 
            help = 'Optional folder(s) directory you want to prepend to each line of the labels file.',
            )

        self.parser.add_argument(
            '--dump_folder_name',
            type = str, 
            help = 'Folder directory to dump the label file (wandb will take this folder to save the artifact).',
            )

        self.parser.add_argument(
            '--log_file_folder',
            type = str, 
            help = 'Name of folder that will contain the log file.',
            )

        self.parser.add_argument(
            '--get_duration',
            action = argparse.BooleanOptionalAction,
            default = True,
            help = 'Calculate duration (in hours) of the labels.\
                (Can be very time consuming).',
            )


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()


if __name__=="__main__":

    args_parser = ArgsParser()
    args_parser.main()
    input_parameters = args_parser.arguments

    feature_extractor = LabelsVersioner(input_parameters)
    feature_extractor.main()
    