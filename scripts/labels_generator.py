# Imports
# ---------------------------------------------------------------------
import argparse
import os
import warnings
import random
import itertools
import pandas as pd
from collections import OrderedDict

from settings import LABELS_GENERATOR_DEFAULT_SETTINGS
# ---------------------------------------------------------------------

# Logging
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------

# TODO add the usage instructions in README.md

# Classes
# ---------------------------------------------------------------------
class LabelsGenerator:

    def __init__(self, params):
        self.params = params
        self.set_random_seed()
        self.set_log_file_handler()
        self.load_metada()


    def set_random_seed(self):

        '''Set the seed for experimental reproduction.'''

        logger.info("Setting random seed...")

        random.seed(1234)

        logger.info("Random seed setted.")


    def set_log_file_handler(self):

        '''Set a logging file handler.'''

        if not os.path.exists(self.params.log_file_folder):
            os.makedirs(self.params.log_file_folder)
        
        logger_file_name = self.params.log_file_name
        logger_file_path = os.path.join(self.params.log_file_folder, logger_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
        logger_file_handler.setLevel(logging.INFO) # TODO set the file handler level as a input param
        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler) 


    def load_metada(self):

        '''Load speakers metadata (needed for hard validation).'''

        logger.info("Loading metadata...")

        if self.params.sv_hard_pairs:

            logger.info(f"Loading metadata from {self.params.metadata_file_path}...")

            # Load de data
            self.metadata_df = pd.read_csv(self.params.metadata_file_path, sep = ",")

            # Lowercase the columns
            self.metadata_df.columns = [col.lower() for col in self.metadata_df]

            # Use only mandatory columns
            mandatory_columns = ["id", "gender", "nationality"]
            for col in mandatory_columns:
                if col not in self.metadata_df.columns:
                    self.metadata_df[col] = "null_value"
            self.metadata_df = self.metadata_df[mandatory_columns]
            
            # Null processing
            self.metadata_df["id"].fillna("null_value", inplace = True)
            self.metadata_df["gender"].fillna("null_value", inplace = True)
            self.metadata_df["nationality"].fillna("null_value", inplace = True)
            
            id_nulls = (self.metadata_df["id"] == "null_value").sum()
            gender_nulls = (self.metadata_df["gender"] == "null_value").sum()
            nationality_nulls = (self.metadata_df["nationality"] == "null_value").sum()
            logger.info(f"{id_nulls} id nulls")
            logger.info(f"{gender_nulls} gender nulls")
            logger.info(f"{nationality_nulls} nationality nulls")

            # Values cleaning
            self.metadata_df["id"] = self.metadata_df["id"].str.strip()
            self.metadata_df["gender"] = self.metadata_df["gender"].str.strip()
            self.metadata_df["nationality"] = self.metadata_df["nationality"].str.strip()
            self.metadata_df["id"] = self.metadata_df["id"].str.lower()
            self.metadata_df["gender"] = self.metadata_df["gender"].str.lower()
            self.metadata_df["nationality"] = self.metadata_df["nationality"].str.lower()

            logger.info(f"Metadata has {self.metadata_df['id'].nunique()} unique speakers.")

            logger.info("Metadata loaded.")


    def generate_speakers_dict(self, load_path):

        '''Construct a dictionary where each speaker_id is the key and the corresponding value is a dictionary with its information (paths of its features and metadata information).'''

        logger.info("Loading dev data...")
    
        speakers_set = set()
        speakers_dict = OrderedDict() # Using ordered dict for reproducibility
        
        # Search over the folder and extract each speaker information
        total_files = 0
        for (dir_path, dir_names, file_names) in os.walk(load_path):
            
            # Directory should have some /id.../ part
            # HACK fix this id part, it is too much specific
            speaker_chunk = [chunk for chunk in dir_path.split("/") if chunk.startswith("id")]
        
            # Only consider directories with /id.../
            if len(speaker_chunk) > 0: 
            
                # If there is more than one /id.../ chunk we take the first and make a warning at the end
                speaker_id = speaker_chunk[0]
                
                # If speaker_id is looped for the first time, initialize variables
                if speaker_id not in speakers_set:
                    speakers_dict[speaker_id] = {}
                    speakers_dict[speaker_id]["files_paths"] = set()
                    
                # If it is a .pickle file, add the path to speakers_dict
                for file_name in file_names:
                    if file_name.split(".")[-1] in ("pickle", "wav"):     
                        # TODO maybe is better to write only the id.../....pickle part of the path           
                        #file_path = os.path.join(dir_path, file_name.replace(".pickle", ""))
                        file_path = os.path.join(dir_path, file_name)
                        # we want to keep only the data structure directory (speaker_id/interview_id/file), 
                        # not the prepended folder directory
                        file_path = file_path.replace(load_path, "")
                        speakers_dict[speaker_id]["files_paths"].add(file_path)
                        total_files = total_files + 1

                # If speaker_id is looped for the first time, add gender
                if speaker_id not in speakers_set:
                    if self.params.sv_hard_pairs:
                        gender = self.metadata_df[self.metadata_df["id"] == speaker_id.lower().strip()]["gender"].iloc[0]
                        if gender is None: gender = "null_value"
                    else:
                        gender = "null_value"
                    speakers_dict[speaker_id]["gender"] = gender

                # If speaker_id is looped for the first time, add nationality
                if speaker_id not in speakers_set:
                    if self.params.sv_hard_pairs:
                        nationality = self.metadata_df[self.metadata_df["id"] == speaker_id.lower().strip()]["nationality"].iloc[0]
                        if nationality is None: nationality = "null_value"
                    else:
                        nationality = "null_value"
                    speakers_dict[speaker_id]["nationality"] = nationality
                
                # Add speaker_id to set and continue with the loop
                speakers_set.add(speaker_id)

            # If there is some other "/id..." in the directory it should be carefully analized
            if len(speaker_chunk) > 1:
                warnings.warn(f"Ambiguous directory path: {dir_path}. Taking {speaker_id} as id.")
                
        # Transform sets into list for experiment reproducibility (sets seems that are unordered).
        for speaker in speakers_dict.keys():
            speakers_dict[speaker]["files_paths"] = list(speakers_dict[speaker]["files_paths"])
            speakers_dict[speaker]["files_paths"].sort()

        # Count metadata information nulls
        if self.params.sv_hard_pairs:

            null_gender = 0
            for value in speakers_dict.values():
                is_null = int((value["gender"] is None) or (value["gender"] == "null_value"))
                null_gender = null_gender + is_null

            null_nationality = 0
            for value in speakers_dict.values():
                is_null = int((value["nationality"] is None) or (value["nationality"] == "null_value"))
                null_nationality = null_nationality + is_null

        logger.info("Dev data loaded.")
        logger.info(f"Total number of speakers: {len(speakers_set)}")
        logger.info(f"Total number of files: {total_files}")
        if self.params.sv_hard_pairs: logger.info(f"Speakers with null gender: {null_gender} ({null_gender / len(speakers_set) * 100}%)")
        if self.params.sv_hard_pairs: logger.info(f"Speakers with null nationality: {null_nationality} ({null_nationality / len(speakers_set) * 100}%)")
        
        return speakers_dict


    def train_valid_split_dict(self, speakers_dict, train_speakers_pctg, random_split = True):

        '''Split speakers_dict into train_speakers_dict and valid_speakers_dict.'''

        logger.info(f"Spliting data into train and valid...")
        
        # We are going to split speakers_dict in train and valid using speaker_id's
        # ie, all audios of the same speaker goes to same split.
        
        speakers_set = set(speakers_dict.keys())
        num_speakers = len(speakers_set)
        # How many speakers will contain the train split
        train_speakers_final_index = int(num_speakers * train_speakers_pctg)

        # The split can be done using speakers' ids order or randomly
        
        # Order the speakers
        speakers_list = list(speakers_set)
        speakers_list.sort() # we need to sort this list in order to make the experiment reproducible (set changes order every time).
        if random_split == False:
            speakers_list.sort()
        elif random_split == True:
            random.shuffle(speakers_list)

        # Label the speakers. Add 0 to total_speakers-1 labels
        for i, speaker in enumerate(speakers_list):
            speakers_dict[speaker]["speaker_num"] = i

        # Sort in order of labels for better understanding
        speakers_dict = {k: v for k, v in sorted(speakers_dict.items(), key=lambda item: item[1]["speaker_num"])}
            
        train_speakers_dict = speakers_dict.copy()
        valid_speakers_dict = speakers_dict.copy()

        for speaker in speakers_dict.keys():

            speaker_num = speakers_dict[speaker]["speaker_num"]

            if speaker_num >= train_speakers_final_index:
                del train_speakers_dict[speaker]
            else:
                del valid_speakers_dict[speaker]

        train_speakers_num = len(train_speakers_dict.keys())
        valid_speakers_num = len(valid_speakers_dict.keys())
        total_speakers_num = train_speakers_num + valid_speakers_num
        if total_speakers_num != num_speakers:
            raise Exception("total_speakers_num does not match total_speakers_num!")
        train_speakers_pctg = train_speakers_num * 100 / total_speakers_num
        valid_speakers_pctg = valid_speakers_num * 100 / total_speakers_num
        
        train_files_num = len(list(itertools.chain.from_iterable([value["files_paths"] for value in train_speakers_dict.values()])))
        valid_files_num = len(list(itertools.chain.from_iterable([value["files_paths"] for value in valid_speakers_dict.values()])))
        total_files_num = train_files_num + valid_files_num
        train_files_pctg = train_files_num * 100 / total_files_num
        valid_files_pctg = valid_files_num * 100 / total_files_num
        
        logger.info(f"{train_speakers_num} speakers ({train_speakers_pctg:.1f}%) with a total of {train_files_num} files ({train_files_pctg:.1f}%) in training split.")
        logger.info(f"{valid_speakers_num} speakers ({valid_speakers_pctg:.1f}%) with a total of {valid_files_num} files ({valid_files_pctg:.1f}%) in validation split.")
        
        logger.info(f"Data splited.")
        
        return train_speakers_dict, valid_speakers_dict
    
    
    def generate_sc_labels_file(self, dump_file_folder, dump_file_name, speakers_dict, max_lines):
    
        logger.info(f"Generating Speaker Classification labels...")
        
        if not os.path.exists(dump_file_folder):
            os.makedirs(dump_file_folder)
        
        dump_path = os.path.join(dump_file_folder, dump_file_name)
        with open(dump_path, 'w') as f:
            for key, value in speakers_dict.items():
                speaker_num = value["speaker_num"]
                for file_path in value["files_paths"]:
                    line_to_write = f"{file_path} {speaker_num} -1"  
                    f.write(line_to_write)
                    f.write('\n')
            f.close()

        # Reduce the number of lines if needed
        if max_lines > 0:
            
            with open(dump_path, 'r') as f:
                labels = f.readlines()
                f.close()
            
            reduced_labels = random.sample(labels, max_lines)

            with open(dump_path, 'w') as f:
                for line_to_write in reduced_labels: 
                    f.write(line_to_write.replace('\n', ''))
                    f.write('\n')
                f.close()

        logger.info(f"Speaker Classification labels generated.")


    def generate_clients_labels_file(
        self,
        clients_dump_file_folder, clients_dump_file_name,
        speakers_dict, 
        clients_lines_max,
        sv_hard_pairs,
        sv_reduced_pairs,
        ):

        logger.info(f"Generating Speaker Verification clients trials...")

        # Create a dict to keep track of used interviews
        if sv_reduced_pairs:
            used_interviews_dict = {}
            for speaker in speakers_dict.keys():
                used_interviews_dict[speaker] = set()
        
        lines_to_write = []
        for _ in range(clients_lines_max):

            # Choose speaker_1 randomly
            speaker_1 = random.choice(list(speakers_dict.keys()))

            # Get speaker_1 files
            speaker_1_dict = speakers_dict[speaker_1]
            speaker_1_files = speaker_1_dict["files_paths"].copy()

            # Generate elegible files for speaker_1
            if sv_reduced_pairs:
                # We are going to use only one file per interview
                elegible_files = [file for file in speaker_1_files if file.split("/")[1] not in used_interviews_dict[speaker_1]]
            else:
                elegible_files = speaker_1_files.copy()
            
            if len(elegible_files) == 0:
                logger.info(f"No elegible files left for speaker {speaker_1}.")
                continue
            
            # Select the first file
            speaker_1_file_1 = random.choice(elegible_files)
            
            speaker_1_file_1_interview = speaker_1_file_1.split("/")[1]
            # Add the interview to the used interviews set
            if sv_reduced_pairs:
                used_interviews_dict[speaker_1].add(speaker_1_file_1_interview)

            # Generate elegible files for speaker_1 for the second file
            if sv_hard_pairs:
                # To make more difficult clients, we are going to consider only files from different interviews
                # We are assuming that every file path has the form speaker_id/interview_id/file
                if sv_reduced_pairs:
                    # We are going to use only one file per interview
                    elegible_files = [file for file in speaker_1_files if file.split("/")[1] not in used_interviews_dict[speaker_1]]
                else:
                    elegible_files = [file for file in speaker_1_files if file.split("/")[1] != speaker_1_file_1_interview]
            else:
                if sv_reduced_pairs:
                    # We are going to use only one file per interview
                    elegible_files = [file for file in speaker_1_files if file.split("/")[1] not in used_interviews_dict[speaker_1]]
                else:
                    elegible_files = speaker_1_files.copy()

            if len(elegible_files) == 0:
                logger.info(f"No elegible files left for speaker {speaker_1}.")
                continue
            
            # Select the second file
            speaker_1_file_2 = random.choice(elegible_files) 
            
            speaker_1_file_2_interview = speaker_1_file_2.split("/")[1]
            # Add the interview to the used interviews set
            if sv_reduced_pairs:
                used_interviews_dict[speaker_1].add(speaker_1_file_2_interview)

            # We order files paths to avoid duplicated pairs
            ordered_files = [speaker_1_file_1, speaker_1_file_2]
            ordered_files.sort()

            line_to_write = f"{ordered_files[0]} {ordered_files[1]}"

            lines_to_write.append(line_to_write)

        # Remove duplicated trials
        lines_to_write = list(set(lines_to_write))
        lines_to_write.sort() # we need to sort this list to make the experiment reproducible (set() changes order every time).

        logger.info(f"{len(lines_to_write)} lines to write for clients.")

        # Dump the file with the trials
        if not os.path.exists(clients_dump_file_folder):
            os.makedirs(clients_dump_file_folder)
        clients_dump_path = os.path.join(clients_dump_file_folder, clients_dump_file_name)
        with open(clients_dump_path, 'w') as f:
            for line_to_write in lines_to_write: 
                f.write(line_to_write)
                f.write('\n')
            f.close()

        logger.info(f"Speaker Verification clients trials generated.")


    def generate_impostors_labels_file(
        self,
        impostors_dump_file_folder, impostors_dump_file_name,
        speakers_dict, 
        impostors_lines_max,
        sv_hard_pairs,
        sv_reduced_pairs,
        ):

        logger.info(f"Generating Speaker Verification impostors trials...")

        # Create a dict to keep track of used interviews
        if sv_reduced_pairs:
            used_interviews_dict = {}
            for speaker in speakers_dict.keys():
                used_interviews_dict[speaker] = set()

        lines_to_write = []
        for _ in range(impostors_lines_max):

            # Choose speaker_1 randomly
            speaker_1 = random.choice(list(speakers_dict.keys()))

            # Get speaker_1 files
            speaker_1_dict = speakers_dict[speaker_1]
            speaker_1_files = list(speaker_1_dict["files_paths"])

            # Generate elegible files for speaker_1
            if sv_reduced_pairs:
                # We are going to use only one file per interview
                elegible_files = [file for file in speaker_1_files if file.split("/")[1] not in used_interviews_dict[speaker_1]]
            else:
                elegible_files = speaker_1_files.copy()
            
            if len(elegible_files) == 0:
                logger.info(f"No elegible files left for speaker {speaker_1}.")
                continue

            # Select the first file
            speaker_1_file_1 = random.choice(elegible_files)
            
            speaker_1_file_1_interview = speaker_1_file_1.split("/")[1]
            # Add the interview to the used interviews set
            if sv_reduced_pairs:
                used_interviews_dict[speaker_1].add(speaker_1_file_1_interview)
            
            # Generate the second elegible speakers
            if sv_hard_pairs:
                speaker_1_gender = speakers_dict[speaker_1]["gender"]
                speaker_1_nationality = speakers_dict[speaker_1]["nationality"]
                elegible_speakers = [
                    speaker for speaker in speakers_dict.keys() if (speakers_dict[speaker]["gender"] == speaker_1_gender and speakers_dict[speaker]["nationality"] == speaker_1_nationality)
                    ]
            else:
                elegible_speakers = list(speakers_dict.keys())
            elegible_speakers.remove(speaker_1)

            if len(elegible_speakers) == 0:
                logger.info(f"No speaker impostor with gender {speaker_1_gender} and nationality {speaker_1_nationality} founded for speaker {speaker_1}.")
                continue

            # Choose speaker_2 (the impostor)
            speaker_2 = random.choice(elegible_speakers)

            # Get speaker_2 files
            speaker_2_dict = speakers_dict[speaker_2]
            speaker_2_files = list(speaker_2_dict["files_paths"])

            # Generate elegible files for speaker_2 for the second file
            if sv_reduced_pairs:
                # We are going to use only one file per interview
                elegible_files = [file for file in speaker_2_files if file.split("/")[1] not in used_interviews_dict[speaker_2]]
            else:
                elegible_files = speaker_2_files.copy()

            if len(elegible_files) == 0:
                logger.info(f"No elegible files left for speaker {speaker_2}.")
                continue

            # Select the second file
            speaker_2_file_1 = random.choice(elegible_files) 
            
            speaker_2_file_1_interview = speaker_2_file_1.split("/")[1]
            # Add the interview to the used interviews set
            if sv_reduced_pairs:
                used_interviews_dict[speaker_2].add(speaker_2_file_1_interview)

            # We order files paths to avoid duplicated pairs
            ordered_files = [speaker_1_file_1, speaker_2_file_1]
            ordered_files.sort()

            # Saving this code for debugging purposes
            #speaker_1_gender = speakers_dict[speaker_1]["gender"]
            #speaker_1_nationality = speakers_dict[speaker_1]["nationality"]
            #speaker_2_gender = speakers_dict[speaker_2]["gender"]
            #speaker_2_nationality = speakers_dict[speaker_2]["nationality"]

            line_to_write = f"{ordered_files[0]} {ordered_files[1]}"

            lines_to_write.append(line_to_write)

        # Remove duplicated trials
        lines_to_write = list(set(lines_to_write))
        lines_to_write.sort() # we need to sort this list to make the experiment reproducible (set() changes order every time).
        
        logger.info(f"{len(lines_to_write)} lines to write for impostors.")

        # Dump the file with the trials
        if not os.path.exists(impostors_dump_file_folder):
            os.makedirs(impostors_dump_file_folder)
        clients_dump_path = os.path.join(impostors_dump_file_folder, impostors_dump_file_name)
        with open(clients_dump_path, 'w') as f:
            for line_to_write in lines_to_write: 
                f.write(line_to_write)
                f.write('\n')
            f.close()

        logger.info(f"Speaker Verification impostors trials generated.")


    def generate_sv_labels_file(self):

        self.generate_clients_labels_file(
            clients_dump_file_folder = self.params.valid_sv_clients_labels_dump_file_folder, 
            clients_dump_file_name = self.params.valid_sv_clients_labels_dump_file_name, 
            speakers_dict = self.valid_speakers_dict, 
            clients_lines_max = self.params.valid_sv_clients_lines_max, 
            sv_hard_pairs = self.params.sv_hard_pairs,
            sv_reduced_pairs = self.params.sv_reduced_pairs,
        )

        self.generate_impostors_labels_file(
            impostors_dump_file_folder = self.params.valid_sv_impostors_labels_dump_file_folder, 
            impostors_dump_file_name = self.params.valid_sv_impostors_labels_dump_file_name, 
            speakers_dict = self.valid_speakers_dict,
            impostors_lines_max = self.params.valid_sv_impostors_lines_max,
            sv_hard_pairs = self.params.sv_hard_pairs,
            sv_reduced_pairs = self.params.sv_reduced_pairs,
        )


    def generate_training_labels(self):

        logger.info(f"Generating training labels...")

        self.generate_sc_labels_file(
            dump_file_folder = self.params.train_sc_labels_dump_file_folder,
            dump_file_name = self.params.train_sc_labels_dump_file_name, 
            speakers_dict = self.train_speakers_dict,
            max_lines = self.params.train_sc_lines_max,
        )

        logger.info(f"Training labels generated.")


    def generate_validation_labels(self):

        logger.info(f"Generating validation labels...")

        # Generate validation Speaker Classification labels
        self.generate_sc_labels_file(
            dump_file_folder = self.params.valid_sc_labels_dump_file_folder,
            dump_file_name = self.params.valid_sc_labels_dump_file_name, 
            speakers_dict = self.valid_speakers_dict,
            max_lines = self.params.valid_sc_lines_max,
        )

        # Generate validation Speaker Verification clients and impostors labels
        self.generate_sv_labels_file()

        logger.info(f"Validation labels generated.")
        

    def main(self):

        # Generate speakers_dict
        self.dev_speakers_dict = self.generate_speakers_dict(
            load_path = self.params.dev_dataset_folder,
        )

        # Split the dict into train and valid
        self.train_speakers_dict, self.valid_speakers_dict = self.train_valid_split_dict(
                self.dev_speakers_dict, 
                self.params.train_speakers_pctg, 
                self.params.random_split,
            )

        # Generate training labels
        self.generate_training_labels()

        # Generate validation labels
        self.generate_validation_labels()


class ArgsParser:

    def __init__(self):
        
        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Generate labels for training and validation according to what is expected for the Network Training. \
                It takes dev data and splits it into training and validation. \
                All audios of the same speaker goes to same split.\
                Dev data files (.pickle) must be contained in a /id.../ folder to be parsed correctly.\
                For the training split, it generates labels for the Speaker Classification task. \
                Each line of the training file will be of the form: file_path speaker_num -1.\
                For the validation split, it generates labels for the speaker verification task randomly.\
                Each line of the validation file will be of the form: file_path_1 file_path_2.',
            )


    def add_parser_args(self):

        self.parser.add_argument(
            'dev_dataset_folder',
            type = str, 
            help = 'Folder containing the extracted features from the development data.',
            )

        self.parser.add_argument(
            '--train_sc_labels_dump_file_folder', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['train_sc_labels_dump_file_folder'], 
            help = 'Folder where we want to dump the .ndx file with the speaker classification training labels.',
            )

        self.parser.add_argument(
            '--train_sc_labels_dump_file_name', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['train_sc_labels_dump_file_name'], 
            help = 'Name of the .ndx file we want to dump speaker classification training labels into.',
            )

        self.parser.add_argument(
            '--valid_sc_labels_dump_file_folder', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_sc_labels_dump_file_folder'], 
            help = 'Folder where we want to dump the .ndx file with the speaker classification validation labels.',
            )

        self.parser.add_argument(
            '--valid_sc_labels_dump_file_name', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_sc_labels_dump_file_name'], 
            help = 'Name of the .ndx file we want to dump speaker verification classification labels into.',
            )

        self.parser.add_argument(
            '--valid_sv_impostors_labels_dump_file_folder', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_sv_impostors_labels_dump_file_folder'], 
            help = 'Folder where we want to dump the .ndx file with the speaker verification validation impostors labels.',
            )

        self.parser.add_argument(
            '--valid_sv_impostors_labels_dump_file_name', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_sv_impostors_labels_dump_file_name'], 
            help = 'Name of the .ndx file we want to dump speaker verification validation impostors labels into.',
            )
        
        self.parser.add_argument(
            '--valid_sv_clients_labels_dump_file_folder', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_sv_clients_labels_dump_file_folder'], 
            help = 'Folder where we want to dump the .ndx file with the speaker verification validation clients labels.',
            )

        self.parser.add_argument(
            '--valid_sv_clients_labels_dump_file_name', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_sv_clients_labels_dump_file_name'], 
            help = 'Name of the .ndx file we want to dump speaker verification validation clients labels into.',
            )
        
        self.parser.add_argument(
            '--train_speakers_pctg', 
            type = float,
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['train_speakers_pctg'],
            help = 'Proportion of speakers training split of the dev set. It must be a float between 0 and 1.',
            )

        self.parser.add_argument(
            '--random_split', 
            action = argparse.BooleanOptionalAction,
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['random_split'],
            help = 'Randomly split into train and validation.',
            )

        self.parser.add_argument(
            '--train_sc_lines_max', 
            type = int,
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['train_sc_lines_max'],
            help = 'Max number of lines generated for the training speaker classification task. Set to -1 if no max is required.',
            )
        
        self.parser.add_argument(
            '--valid_sc_lines_max', 
            type = int,
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_sc_lines_max'],
            help = 'Max number of lines generated for the validation speaker classification task. Set to -1 if no max is required.',
            )

        self.parser.add_argument(
            '--valid_sv_clients_lines_max', 
            type = int,
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_sv_clients_lines_max'],
            help = 'Max number of clients labels generated.',
            )

        self.parser.add_argument(
            '--valid_sv_impostors_lines_max', 
            type = int,
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_sv_impostors_lines_max'],
            help = 'Max number of impostors labels generated.',
            )

        self.parser.add_argument(
            '--sv_hard_pairs', 
            action = argparse.BooleanOptionalAction,
            help = 'If True, Speaker Verification impostors labels are generated only using same gender and nationality. \
                Speaker Verification clients are generated only using different interviews pairs. \
                In True, metadata_file_path must be specified.',
            )

        self.parser.add_argument(
            '--sv_reduced_pairs', 
            action = argparse.BooleanOptionalAction,
            help = 'If True, Speaker Verification impostors and clients labels are generated using only one file per interview.',
            )

        self.parser.add_argument(
            '--metadata_file_path', 
            type = str, 
            help = '.csv file containg speakers metadata such as gender or nationality.\
                This file must be comma separated and have id, gender, and/or nationality columns.',
            )

        self.parser.add_argument(
            '--log_file_folder',
            type = str, 
            default = './logs/labels_generator/',
            help = 'Name of folder that will contain the log file.',
            )
        
        self.parser.add_argument(
            '--log_file_name',
            type = str, 
            default = 'logs.log',
            help = 'Name of the log file.',
            )

        self.parser.add_argument(
            "--verbose", 
            action = argparse.BooleanOptionalAction,
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['verbose'],
            help = "Increase output verbosity.",
            )


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()
# ---------------------------------------------------------------------

if __name__=="__main__":

    args_parser = ArgsParser()
    args_parser.main()
    parameters = args_parser.arguments

    labels_generator = LabelsGenerator(parameters)
    labels_generator.main()
    














