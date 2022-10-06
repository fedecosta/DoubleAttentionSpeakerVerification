import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import time
import datetime

from random import randint, randrange

from model import SpeakerClassifier
from data import normalizeFeatures, featureReader
from utils import scoreCosineDistance, Score, Score_2, generate_model_name
from settings import MODEL_EVALUATOR_DEFAULT_SETTINGS


class ModelEvaluator:

    def __init__(self, input_params):

        self.input_params = input_params
        self.set_device()
        self.set_random_seed()
        self.evaluation_results = {}
        self.start_time = time.time()
        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')

    
    def set_device(self):
    
        print('Setting device...')

        # Set device to GPU or CPU depending on what is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        print(f"Running on {self.device} device.")
    
        if torch.cuda.device_count() > 1:
            print(f"{torch.cuda.device_count()} GPUs available.")
    
        print("Device setted.")

    
    def set_random_seed(self):

        print("Setting random seed...")

        # Set the seed for experimental reproduction
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        print("Random seed setted.")


    def load_checkpoint(self):

        # Load checkpoint
        checkpoint_path = self.input_params.model_checkpoint_path

        print(f"Loading checkpoint from {checkpoint_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location = self.device)

        print(f"Model checkpoint was saved at epoch {self.checkpoint['training_variables']['epoch']}")

        print(f"Checkpoint loaded.")
        
    
    def load_checkpoint_params(self):

        self.params = self.checkpoint['settings']


    def load_checkpoint_network(self):

        try:
            self.net.load_state_dict(self.checkpoint['model'])
        except RuntimeError:    
            self.net.module.load_state_dict(self.checkpoint['model'])


    def load_network(self):

        self.net = SpeakerClassifier(self.params, self.device)
        
        self.load_checkpoint_network()
        
        # Assign model to device
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.net = nn.DataParallel(self.net)


    def sample_spectogram_window(self, features):

        

        # Cut the spectrogram with a fixed length at a random start

        file_frames = features.shape[0]
        
        # FIX why this hardcoded 100? 
        # The cutting here is in FRAMES, not secs
        # It would be nice to do the cutting at the feature extractor module
        # It seems that some kind of padding is made with librosa, but it should be done at the feature extractor module also
        sample_size_in_frames = 3.5 * 100

        # Get a random start point
        # index = randint(0, max(0, file_frames - sample_size_in_frames - 1))
        index = 0

        # Generate the index slicing
        a = np.array(range(min(file_frames, int(sample_size_in_frames)))) + index
        
        # Slice the spectrogram
        sliced_spectrogram = features[a,:]

        return sliced_spectrogram


    def __extractInputFromFeature(self, sline, data_dir):

        features1 = normalizeFeatures(
            featureReader(
                data_dir + '/' + sline[0] + '.pickle'), 
                normalization = self.params.normalization,
                )
        features2 = normalizeFeatures(
            featureReader(
                data_dir + '/' + sline[1] + '.pickle'), 
                normalization = self.params.normalization,
                )

        #features1 = self.sample_spectogram_window(features1)
        #features2 = self.sample_spectogram_window(features2)

        input1 = torch.FloatTensor(features1).to(self.device)
        input2 = torch.FloatTensor(features2).to(self.device)
        
        return input1.unsqueeze(0), input2.unsqueeze(0)


    def __extract_scores(self, trials, data_dir, total_trials):

        scores = []
        for num_line, line in enumerate(trials):

            print(f"\r Extracting score {num_line} of {total_trials - 1}...", end='', flush = True)

            sline = line[:-1].split()

            input1, input2 = self.__extractInputFromFeature(sline, data_dir)

            if torch.cuda.device_count() > 1:
                emb1, emb2 = self.net.module.get_embedding(input1), self.net.module.get_embedding(input2)
            else:
                emb1, emb2 = self.net.get_embedding(input1), self.net.get_embedding(input2)

            dist = scoreCosineDistance(emb1, emb2)
            scores.append(dist.item())
        
        print(f"Scores extracted.")
        
        return scores


    def __calculate_EER(self, CL, IM):

        print("Calculating EER...")

        thresholds = np.arange(-1, 1, 0.01)
        FRR, FAR = np.zeros(len(thresholds)), np.zeros(len(thresholds))
        for idx, th in enumerate(thresholds):
            FRR[idx] = Score(CL, th, 'FRR')
            FAR[idx] = Score(IM, th, 'FAR')

        EER_Idx = np.argwhere(np.diff(np.sign(FAR - FRR)) != 0).reshape(-1)
        if len(EER_Idx) > 0:
            if len(EER_Idx) > 1:
                EER_Idx = EER_Idx[0]
            EER = round((FAR[int(EER_Idx)] + FRR[int(EER_Idx)]) / 2, 4)
        else:
            EER = 50.00

        print("EER calculated.")

        return EER


    def evaluate(self, clients_labels, impostor_labels, data_dir):

        print("Evaluating model...")

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            print("Going to evaluate using these labels:")
            print(f"Clients: {clients_labels}")
            print(f"Impostors: {impostor_labels}")
            print(f"For each row in these labels where are using prefix {data_dir}")

            self.clients_num = sum(1 for line in open(clients_labels))
            self.impostors_num = sum(1 for line in open(impostor_labels))

            print(f"{self.clients_num} test clients to evaluate.")
            print(f"{self.impostors_num} test impostors to evaluate.")

            # EER Validation
            with open(clients_labels,'r') as clients_in, open(impostor_labels,'r') as impostors_in:

                # score clients
                self.CL = self.__extract_scores(clients_in, data_dir, self.clients_num)
                self.IM = self.__extract_scores(impostors_in, data_dir, self.impostors_num)
            
            # Compute EER
            self.EER = self.__calculate_EER(self.CL, self.IM)
            print(f"Model evaluated on test dataset. EER: {self.EER:.2f}")


    def save_report(self):

        print("Creating report...")

        self.end_time = time.time()
        self.end_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        self.elapsed_time_hours = (self.end_time - self.start_time) / 60 / 60
        model_name = generate_model_name(self.params)

        self.evaluation_results['start_datetime'] = self.start_datetime
        self.evaluation_results['end_datetime'] = self.end_datetime
        self.evaluation_results['elapsed_time_hours'] = self.elapsed_time_hours
        self.evaluation_results['model_name'] = model_name
        self.evaluation_results['model_loaded_from'] = self.input_params.model_checkpoint_path
        self.evaluation_results['clients_loaded_from'] = self.input_params.test_clients
        self.evaluation_results['impostors_loaded_from'] = self.input_params.test_impostors
        self.evaluation_results['clients_num'] = self.clients_num
        self.evaluation_results['impostors_num'] = self.impostors_num
        self.evaluation_results['EER'] = self.EER
        #self.evaluation_results['CL'] = self.CL
        #self.evaluation_results['IM'] = self.IM

        
        dump_folder = self.input_params.dump_folder
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)

        dump_file_name = f"report_{model_name}_{self.start_datetime}.json"

        dump_path = os.path.join(dump_folder, dump_file_name)
        
        print(f"Saving file into {dump_path}")
        with open(dump_path, 'w', encoding = 'utf-8') as handle:
            json.dump(self.evaluation_results, handle, ensure_ascii = False, indent = 4)
        print("Saved.")


    def main(self):
        self.load_checkpoint()
        self.load_checkpoint_params()
        self.load_network()
        self.evaluate(
            clients_labels = self.input_params.test_clients,
            impostor_labels = self.input_params.test_impostors, 
            data_dir = self.input_params.data_dir,
            )
        self.save_report()
        

class ArgsParser:

    def __init__(self):
        
        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Evaluate a trained model on a particular dataset.',
            )


    def add_parser_args(self):

        self.parser.add_argument(
            'model_checkpoint_path', 
            type = str, 
            help = 'Complete path where the checkpoint model is saved.'
            ) 

        self.parser.add_argument(
            'test_clients', 
            type = str, 
            help = 'Path of the file containing the clients pairs paths.',
            )

        self.parser.add_argument(
            'test_impostors', 
            type = str,
            help = 'Path of the file containing the impostors pairs paths.',
            )

        self.parser.add_argument(
            '--data_dir', 
            type = str, 
            default = MODEL_EVALUATOR_DEFAULT_SETTINGS["data_dir"],
            help = 'Optional additional directory to prepend to clients and impostors pairs paths.',
            )

        self.parser.add_argument(
            '--dump_folder', 
            type = str, 
            default = MODEL_EVALUATOR_DEFAULT_SETTINGS["dump_folder"],
            help = 'Folder to save the results.'
            )


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()
    

if __name__ == "__main__":

    args_parser = ArgsParser()
    args_parser.main()
    input_params = args_parser.arguments
        
    model_evaluator = ModelEvaluator(input_params)
    model_evaluator.main()