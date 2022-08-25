# TODO fix this hack
# import sys
# sys.path.append('./scripts/')

import argparse
import os
import warnings
import random
import itertools

from settings import LABELS_GENERATOR_DEFAULT_SETTINGS

# TODO add the usage instructions in README.md


class LabelsGenerator:

    def __init__(self, params):
        self.params = params


    def generate_speakers_dict(self, load_path):

        print("Loading dev data...")
    
        speakers_set = set()
        speakers_dict = {}
        
        for (dir_path, dir_names, file_names) in os.walk(load_path):
            
            
            # Directory should have some /id.../ part
            # TODO fix this id part, it is too much specific
            speaker_chunk = [chunk for chunk in dir_path.split("/") if chunk.startswith("id")]
        
            # Only consider directories with /id.../
            if len(speaker_chunk) > 0: 
            
                speaker_id = speaker_chunk[0]
                
                # If speaker_id is looped for the first time, initialize variables
                if speaker_id not in speakers_set:
                    speakers_dict[speaker_id] = {}
                    speakers_dict[speaker_id]["files_paths"] = set()
                    
                # If it is a .pickle file, add the path to speakers_dict
                for file_name in file_names:
                    if file_name.split(".")[-1] == "pickle":                
                        file_path = dir_path + "/" + file_name.replace(".pickle", "")
                        speakers_dict[speaker_id]["files_paths"].add(file_path)
                
                # Add speaker_id to set and continue with the loop
                speakers_set.add(speaker_id)

            # If there is some other "/id..." in the directory it should be analized
            if len(speaker_chunk) > 1:
                warnings.warn(f"Ambiguous directory path: {dir_path}")
                
        print("Dev data loaded.")
        
        return speakers_dict


    def train_valid_split_dict(self, speakers_dict, train_speakers_pctg, random_split = True):

        print(f"Spliting data into train and valid...")
        
        # We are going to split speakers_dict in train and valid using speaker_id's
        # ie, all audios of the same speaker goes to same split.
        
        speakers_set = set(speakers_dict.keys())
        num_speakers = len(speakers_set)
        # How many speakers will contain the train split
        train_speakers_final_index = int(num_speakers * train_speakers_pctg)

        # The split can be done using speakers' ids order or randomly
        
        # Order the speakers
        speakers_list = list(speakers_set)
        if random_split == False:
            speakers_list.sort()
        elif random_split == True:
            random.shuffle(speakers_list)

        # Add 0 to total_speakers-1 labels
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
        
        
        print(f"{train_speakers_num} speakers ({train_speakers_pctg:.1f}%) with a total of {train_files_num} files ({train_files_pctg:.1f}%) in training split.")
        print(f"{valid_speakers_num} speakers ({valid_speakers_pctg:.1f}%) with a total of {valid_files_num} files ({valid_files_pctg:.1f}%) in validation split.")
        
        print(f"Data splited.")
        
        return train_speakers_dict, valid_speakers_dict
    
    
    def generate_labels_file(self, dump_file_folder, dump_file_name, speakers_dict):
    
        print(f"Generating training labels...")
        
        if not os.path.exists(dump_file_folder):
            os.makedirs(dump_file_folder)
        
        dump_path = dump_file_folder + '/' + dump_file_name
        with open(dump_path, 'w') as f:
            for key, value in speakers_dict.items():
                speaker_num = value["speaker_num"]
                for file_path in value["files_paths"]:
                    line_to_write = f"{file_path} {speaker_num} -1"  
                    f.write(line_to_write)
                    f.write('\n')
            f.close()

        print(f"Training labels generated.")

    
    def generate_clients_impostors_files(
        self,
        impostors_dump_file_folder, impostors_dump_file_name,
        clients_dump_file_folder, clients_dump_file_name,
        speakers_dict, 
        clients_lines_max = None, impostors_lines_max = None):
        
        print(f"Generating valid clients and impostors trials...")
        
        clients_lines_to_write = []
        impostors_lines_to_write = []

        distinct_speakers = list(speakers_dict.keys())

        one_speaker_combinations = [(speaker, speaker) for speaker in distinct_speakers]
        two_speaker_combinations = list(itertools.combinations(distinct_speakers, 2))  
        speaker_combinations = one_speaker_combinations + two_speaker_combinations

        for speaker_1, speaker_2 in speaker_combinations:

            speaker_1_files = speakers_dict[speaker_1]["files_paths"]
            speaker_2_files = speakers_dict[speaker_2]["files_paths"]

            if speaker_1 == speaker_2:
                files_combinations = list(itertools.combinations(speaker_1_files, 2))
                for file_1, file_2 in files_combinations:
                    line_to_write = file_1 + " " + file_2
                    clients_lines_to_write.append(line_to_write)
            else:
                files_combinations = list(itertools.product(speaker_1_files, speaker_2_files))
                for file_1, file_2 in files_combinations:
                    line_to_write = file_1 + " " + file_2
                    impostors_lines_to_write.append(line_to_write)

        if clients_lines_max is not None:
            clients_lines_to_write = random.sample(clients_lines_to_write, clients_lines_max)
        if impostors_lines_max is not None:
            impostors_lines_to_write = random.sample(impostors_lines_to_write, impostors_lines_max)
        
        print(f"{len(clients_lines_to_write)} lines to write for clients.")
        print(f"{len(impostors_lines_to_write)} lines to write for impostors.")
        
        if not os.path.exists(clients_dump_file_folder):
            os.makedirs(clients_dump_file_folder)
        clients_dump_path = clients_dump_file_folder + '/' + clients_dump_file_name
        with open(clients_dump_path, 'w') as f:
            for line_to_write in clients_lines_to_write: 
                f.write(line_to_write)
                f.write('\n')
            f.close()

        if not os.path.exists(impostors_dump_file_folder):
            os.makedirs(impostors_dump_file_folder)
        impostors_dump_path = impostors_dump_file_folder + '/' + impostors_dump_file_name
        with open(impostors_dump_path, 'w') as f:
            for line_to_write in impostors_lines_to_write: 
                f.write(line_to_write)
                f.write('\n')
            f.close()

        print(f"Valid clients and impostors trials generated.")

    
    def main(self):

        self.dev_speakers_dict = self.generate_speakers_dict(
            load_path = self.params.dev_dataset_folder,
        )

        self.num_speakers = len(self.dev_speakers_dict)
        print(f"Total number of distinct speakers loaded: {self.num_speakers}")

        self.train_speakers_dict, self.valid_speakers_dict = self.train_valid_split_dict(
            self.dev_speakers_dict, 
            self.params.train_speakers_pctg, 
            self.params.random_split,
        )
        
        self.generate_labels_file(
            dump_file_folder = self.params.train_labels_dump_file_folder,
            dump_file_name = self.params.train_labels_dump_file_name, 
            speakers_dict = self.train_speakers_dict,
        )
        
        self.generate_clients_impostors_files(
            impostors_dump_file_folder = self.params.valid_impostors_labels_dump_file_folder, 
            impostors_dump_file_name = self.params.valid_impostors_labels_dump_file_name, 
            clients_dump_file_folder = self.params.valid_clients_labels_dump_file_folder, 
            clients_dump_file_name = self.params.valid_clients_labels_dump_file_name, 
            speakers_dict = self.valid_speakers_dict, 
            clients_lines_max = self.params.clients_lines_max, 
            impostors_lines_max = self.params.impostors_lines_max,
        )
        
    
class ArgsParser:

    def __init__(self):
        
        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Generate labels for training and validation according to what is expected for the Network Training. \
                It takes dev data and splits it into training and validation \
                For the training split, it generates labels for the speaker classification task. \
                For the validation split, it generates labels for the speaker verification task.',
            )


    def add_parser_args(self):

        self.parser.add_argument(
            'dev_dataset_folder',
            type = str, 
            help = 'Folder containing the extracted features from the development data.',
            )

        self.parser.add_argument(
            '--train_labels_dump_file_folder', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['train_labels_dump_file_folder'], 
            help = 'Folder where we want to dump the .ndx file with the training labels.',
            )

        self.parser.add_argument(
            '--train_labels_dump_file_name', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['train_labels_dump_file_name'], 
            help = 'Name of the .ndx file we want to dump training labels into.',
            )

        self.parser.add_argument(
            '--valid_impostors_labels_dump_file_folder', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_impostors_labels_dump_file_folder'], 
            help = 'Folder where we want to dump the .ndx file with the validation impostors labels.',
            )

        self.parser.add_argument(
            '--valid_impostors_labels_dump_file_name', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_impostors_labels_dump_file_name'], 
            help = 'Name of the .ndx file we want to dump validation impostors labels into.',
            )
        
        self.parser.add_argument(
            '--valid_clients_labels_dump_file_folder', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_clients_labels_dump_file_folder'], 
            help = 'Folder where we want to dump the .ndx file with the validation clients labels.',
            )

        self.parser.add_argument(
            '--valid_clients_labels_dump_file_name', 
            type = str, 
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['valid_clients_labels_dump_file_name'], 
            help = 'Name of the .ndx file we want to dump validation clients labels into.',
            )
        
        self.parser.add_argument(
            '--train_speakers_pctg', 
            type = float,
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['train_speakers_pctg'],
            help = 'Proportion of training split of the dev set. It must be between 0 and 1',
            )

        self.parser.add_argument(
            '--random_split', 
            action = "store_true",
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['random_split'],
            help = 'Randomly split into train and validation.',
            )

        self.parser.add_argument(
            '--clients_lines_max', 
            type = int,
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['clients_lines_max'],
            help = 'Max number of clients labels generated.',
            )

        self.parser.add_argument(
            '--impostors_lines_max', 
            type = int,
            default = LABELS_GENERATOR_DEFAULT_SETTINGS['impostors_lines_max'],
            help = 'Max number of impostors labels generated.',
            )

        self.parser.add_argument(
            "--verbose", 
            action = "store_true",
            help = "Increase output verbosity.",
            )


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()


if __name__=="__main__":

    args_parser = ArgsParser()
    args_parser.main()
    parameters = args_parser.arguments

    labels_generator = LabelsGenerator(parameters)
    labels_generator.main()
    














