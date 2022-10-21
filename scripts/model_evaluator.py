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
from torch.utils.data import DataLoader

from model import SpeakerClassifier
from data import TestDataset
from utils import scoreCosineDistance, generate_model_name, calculate_EER
from settings import MODEL_EVALUATOR_DEFAULT_SETTINGS

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


class ModelEvaluator:

    def __init__(self, input_params):

        self.input_params = input_params
        self.set_batch_size()
        self.set_device()
        self.set_random_seed()
        self.set_log_file_handler()
        self.evaluation_results = {}
        self.start_time = time.time()
        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        self.load_checkpoint()
        self.load_checkpoint_params()
        self.load_network()

    
    def set_device(self):
    
        logger.info('Setting device...')

        # Set device to GPU or CPU depending on what is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        logger.info(f"Running on {self.device} device.")
    
        if torch.cuda.device_count() > 1:
            logger.info(f"{torch.cuda.device_count()} GPUs available.")
    
        logger.info("Device setted.")

    
    def set_random_seed(self):

        logger.info("Setting random seed...")

        # Set the seed for experimental reproduction
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        logger.info("Random seed setted.")


    def set_log_file_handler(self):

        # Set a logging file handler
        if not os.path.exists(self.input_params.log_file_folder):
            os.makedirs(self.input_params.log_file_folder)
        logger_file_path = os.path.join(self.input_params.log_file_folder, self.input_params.log_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
        logger_file_handler.setLevel(logging.INFO) # TODO set the file handler level as a input param
        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler)


    def set_batch_size(self):

        # If evaluation_type is total_length, batch size must be 1 because we will have different-size samples
        if self.input_params.evaluation_type == "total_length":
            self.input_params.batch_size = 1


    def load_checkpoint(self):

        # Load checkpoint
        checkpoint_path = self.input_params.model_checkpoint_path

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location = self.device)

        logger.info(f"Model checkpoint was saved at epoch {self.checkpoint['training_variables']['epoch']}")

        logger.info(f"Checkpoint loaded.")
        
    
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
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.net = nn.DataParallel(self.net)

    # This function was created to be able to generate batches with different-length samples using padding.
    # It is not used, but saved just in case.
    def collate_batch(self, data):
        """
        data: is a list of tuples with (example, label, length)
        where 'example' is a tensor of arbitrary shape
        and label/length are scalars
        """

        with torch.no_grad():
        
            features_1, lengths_1, features_2, lengths_2, labels,  = zip(*data)

            max_len_1 = max(lengths_1)
            max_len_2 = max(lengths_2)
            
            # We are assuming that all features have the same number of columns
            number_columns = data[0][0].size(1)
            
            padded_features_1 = torch.zeros((len(data), max_len_1, number_columns))
            padded_features_2 = torch.zeros((len(data), max_len_2, number_columns))
            for i in range(len(data)):

                padded_features_1[i][:lengths_1[i],:] = features_1[i]
                padded_features_2[i][:lengths_2[i],:] = features_2[i]
                
            labels = torch.tensor(labels)
            lengths_1 = torch.tensor(lengths_1)
            lengths_2 = torch.tensor(lengths_2)

            return padded_features_1.float(), lengths_1.long(), padded_features_2.float(), lengths_2.long(), labels.long()


    def load_data(self):
            
        logger.info(f'Loading data from {self.input_params.test_clients} and {self.input_params.test_impostors}')

        # Instanciate a Dataset class
        dataset = TestDataset(train_parameters = self.params, input_parameters = self.input_params)

        # Instanciate a DataLoader class
        self.evaluating_generator = DataLoader(
            dataset, 
            batch_size = self.input_params.batch_size,
            shuffle = False,
            num_workers = 1, # TODO set this as a input parameter
            #collate_fn = self.collate_batch,
            )

        logger.info("Data and labels loaded.")


    def calculate_similarities(self):

        logger.info("Extracting embeddings and calculating similarities...")

        similarities = []
        for self.batch_number, (input_1, input_2, label) in enumerate(self.evaluating_generator):

            logger.info(f"Batch {self.batch_number} of {self.total_batches}")
            
            input_1 = input_1.float().to(self.device)
            input_2 = input_2.float().to(self.device)
            label = label.int().to(self.device)

            if torch.cuda.device_count() > 1:
                embedding_1 = self.net.module.get_embedding(input_1)
                embedding_2 = self.net.module.get_embedding(input_2)
            else:
                embedding_1 = self.net.get_embedding(input_1)
                embedding_2 = self.net.get_embedding(input_2)
            
            dist = scoreCosineDistance(embedding_1, embedding_2)

            similarities = similarities + list(zip(dist.cpu().detach().numpy(), label.cpu().detach().numpy()))

        logger.info(f"Embeddings extracted and similarities calculated.")

        return similarities


    def evaluate(self, clients_labels, impostor_labels, data_dir):

        logger.info("Evaluating model...")

        logger.info("Going to evaluate using these labels:")
        logger.info(f"Clients: {clients_labels}")
        logger.info(f"Impostors: {impostor_labels}")
        logger.info(f"For each row in these labels where are using prefix {data_dir}")

        self.clients_num = sum(1 for line in open(clients_labels))
        self.impostors_num = sum(1 for line in open(impostor_labels))

        logger.info(f"{self.clients_num} test clients to evaluate.")
        logger.info(f"{self.impostors_num} test impostors to evaluate.")

        similarities = self.calculate_similarities()
        self.CL = [similarity for similarity, label in similarities if label == 1]
        self.IM = [similarity for similarity, label in similarities if label == 0]
        
        # Compute EER
        self.EER = calculate_EER(self.CL, self.IM)
        logger.info(f"Model evaluated on test dataset. EER: {self.EER:.3f}")


    def save_report(self):

        logger.info("Creating report...")

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
        self.evaluation_results['input_params'] = vars(self.input_params) # convert Namespace object into dictionary
        self.evaluation_results['training_params'] = vars(self.params) # convert Namespace object into dictionary
        
        dump_folder = self.input_params.dump_folder
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)

        dump_file_name = f"report_{model_name}_{self.start_datetime}.json"

        dump_path = os.path.join(dump_folder, dump_file_name)
        
        logger.info(f"Saving file into {dump_path}")
        with open(dump_path, 'w', encoding = 'utf-8') as handle:
            json.dump(self.evaluation_results, handle, ensure_ascii = False, indent = 4)
        logger.info("Saved.")


    def main(self):

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            self.load_data()
            self.total_batches = len(self.evaluating_generator)

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
            nargs = '+',
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

        self.parser.add_argument(
            '--log_file_folder',
            type = str, 
            default = MODEL_EVALUATOR_DEFAULT_SETTINGS['log_file_folder'],
            help = 'Name of folder that will contain the log file.',
            )
        
        self.parser.add_argument(
            '--log_file_name',
            type = str, 
            default = MODEL_EVALUATOR_DEFAULT_SETTINGS['log_file_name'],
            help = 'Name of the log file.',
            )

        self.parser.add_argument(
            '--normalization', 
            type = str, 
            default = MODEL_EVALUATOR_DEFAULT_SETTINGS['normalization'], 
            choices = ['cmn', 'cmvn'],
            help = 'Type of normalization applied to the features. \
                It can be Cepstral Mean Normalization or Cepstral Mean and Variance Normalization',
            )

        self.parser.add_argument(
            '--evaluation_type', 
            type = str, 
            choices = ['random_crop', 'total_length'],
            default = MODEL_EVALUATOR_DEFAULT_SETTINGS['evaluation_type'], 
            help = 'With random_crop the utterances are croped at random with random_crop_size frames before doing the forward pass.\
                In this case, samples are batched with batch_size.\
                With total_length, full length audios are passed through the forward.\
                In this case, samples are automatically batched with batch_size = 1, since they have different lengths.',
            )

        self.parser.add_argument(
            '--batch_size', 
            type = int, 
            default = MODEL_EVALUATOR_DEFAULT_SETTINGS['batch_size'],
            help = "Size of evaluation batches. Automatically set to 1 if evaluation_type is total_length.",
            )

        self.parser.add_argument(
            '--random_crop_size', 
            type = int, 
            default = MODEL_EVALUATOR_DEFAULT_SETTINGS['random_crop_size'], 
            help = 'Cut the input spectrogram with random_crop_size frames length at a random starting point. \
                random_crop_size is measured in frames.',
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