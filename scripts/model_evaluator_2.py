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
from data import normalizeFeatures, featureReader, TestDataset
from utils import scoreCosineDistance, Score, Score_2, generate_model_name
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
        self.set_device()
        self.set_random_seed()
        self.set_log_file_handler()
        self.evaluation_results = {}
        self.start_time = time.time()
        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        self.load_checkpoint()
        self.load_checkpoint_params()
        self.load_network()
        self.load_data()
        self.total_batches = len(self.training_generator)

    
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


    def load_data(self):
            
        logger.info(f'Loading data from {self.input_params.test_clients}')
        
        # Read the paths of the clients audios
        with open(self.input_params.test_clients, 'r') as clients_file:
            clients_trials = clients_file.readlines()

        logger.info(f'Loading data from {self.input_params.test_impostors}')
        
        # Read the paths of the impostors audios
        with open(self.input_params.test_impostors, 'r') as impostors_file:
            impostors_trials = impostors_file.readlines()

        # Instanciate a Dataset class
        dataset = TestDataset(clients_trials, impostors_trials, self.params, self.input_params, self.net)
        
        # Load DataLoader params
        data_loader_parameters = {
            'batch_size': 256, #self.params.batch_size, 
            'shuffle': False, # FIX hardcoded True
            'num_workers': 2, #self.params.num_workers
            }
        
        # Instanciate a DataLoader class
        self.training_generator = DataLoader(
            dataset, 
            **data_loader_parameters,
            )

        logger.info("Data and labels loaded.")


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

        input1 = torch.FloatTensor(features1).to(self.device)
        input2 = torch.FloatTensor(features2).to(self.device)
        
        return input1.unsqueeze(0), input2.unsqueeze(0)


    def calculate_similarities(self):

        logger.info("Extracting embeddings and calculating similarities...")

        similarities = []
        for self.batch_number, (input_1, input_2, label) in enumerate(self.training_generator):

            logger.info(f"Batch {self.batch_number} of {self.total_batches}")
            
            input_1 = input_1.float().to(self.device)
            input_2 = input_2.float().to(self.device)
            label = label.int().to(self.device)

            if torch.cuda.device_count() > 1:
                embedding_1 = self.net.module.get_embedding(input_1)
                embedding_2 = self.net.module.get_embedding(input_2)
            else:
                embedding_1 = self.net.get_embedding(input_1),
                embedding_2 = self.net.get_embedding(input_2),

            dist = scoreCosineDistance(embedding_1, embedding_2)

            similarities = similarities + list(zip(dist.cpu().detach().numpy(), label.cpu().detach().numpy()))

        logger.info(f"Embeddings extracted and similarities calculated.")

        return similarities


    def __calculate_EER(self, CL, IM):

        logger.info("Calculating EER...")

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

        logger.info("EER calculated.")

        return EER


    def evaluate(self, clients_labels, impostor_labels, data_dir):

        logger.info("Evaluating model...")

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

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
            self.EER = self.__calculate_EER(self.CL, self.IM)
            logger.info(f"Model evaluated on test dataset. EER: {self.EER:.2f}")


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
        self.evaluation_results['clients_num'] = self.clients_num
        self.evaluation_results['impostors_num'] = self.impostors_num
        self.evaluation_results['EER'] = self.EER
        #self.evaluation_results['CL'] = str(self.CL)
        #self.evaluation_results['IM'] = str(self.IM)

        
        dump_folder = self.input_params.dump_folder
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)

        dump_file_name = f"report_2_{model_name}_{self.start_datetime}.json"

        dump_path = os.path.join(dump_folder, dump_file_name)
        
        logger.info(f"Saving file into {dump_path}")
        with open(dump_path, 'w', encoding = 'utf-8') as handle:
            json.dump(self.evaluation_results, handle, ensure_ascii = False, indent = 4)
        logger.info("Saved.")


    def main(self):

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


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()
    

if __name__ == "__main__":

    args_parser = ArgsParser()
    args_parser.main()
    input_params = args_parser.arguments
        
    model_evaluator = ModelEvaluator(input_params)
    model_evaluator.main()