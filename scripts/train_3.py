import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from data import Dataset, normalizeFeatures, featureReader
from model import SpeakerClassifier
from utils import getNumberOfSpeakers, Accuracy, scoreCosineDistance, Score
from settings import TRAIN_DEFAULT_SETTINGS

# Set logging config
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%H:%M:%S',
    )

# Set a logging file handler
if not os.path.exists('scripts/logs'):
    os.makedirs('scripts/logs')
logger_file_handler = logging.FileHandler('scripts/logs/train_3.log', mode = 'w')
logger_file_handler.setLevel(logging.DEBUG)
logger_file_handler.setFormatter(logger_formatter)

# Set a logging stram handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.DEBUG)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_file_handler)
logger.addHandler(logger_stream_handler)


class Trainer:

    def __init__(self, params):
        self.params = params
        self.params.num_spkrs = getNumberOfSpeakers(self.params.train_labels_path) 
        self.set_random_seed()
        self.set_device()
        self.__load_data()
        self.__load_network()
        self.__load_loss_function()
        self.__load_optimizer()
        self.__initialize_training_variables()


    # Init methods

    def set_random_seed(self):

        logger.info("Setting random seed...")

        # Set the seed for experimental reproduction
        torch.manual_seed(1234)
        np.random.seed(1234)

        logger.info("Random seed setted.")


    def set_device(self):
        
        logger.info('Setting device...')

        # Set device to GPU or CPU depending on what is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Running on {self.device} device.")
        
        if torch.cuda.device_count() > 1:
            logger.info(f"{torch.cuda.device_count()} GPUs available.")
        
        logger.info("Device setted.")


    def __load_data(self):
            
        logger.info(f'Loading data and labels from {self.params.train_labels_path}')
        
        # Read the paths of the train audios and their labels
        with open(self.params.train_labels_path, 'r') as data_labels_file:
            train_labels = data_labels_file.readlines()

        # Instanciate a Dataset class
        dataset = Dataset(train_labels, self.params)
        
        # Load DataLoader params
        data_loader_parameters = {
            'batch_size': self.params.batch_size, 
            'shuffle': True, # FIX hardcoded True
            'num_workers': self.params.num_workers
            }
        
        # Instanciate a DataLoader class
        self.training_generator = DataLoader(
            dataset, 
            **data_loader_parameters,
            )

        logger.info("Data and labels loaded.")


    def __load_network(self):

        # Load the model (Neural Network)

        logger.info("Loading the network...")

        # Load model class
        self.net = SpeakerClassifier(self.params, self.device)
        
        # Assign model to device
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.net = nn.DataParallel(self.net)
        
        logger.info("Network loaded.")


    # Load the loss function
    def __load_loss_function(self):

        logger.info("Loading the loss function...")

        self.loss_function = nn.CrossEntropyLoss()

        logger.info("Loss function loaded.")


    # Load the optimizer
    def __load_optimizer(self):

        logger.info("Loading the optimizer...")

        if self.params.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay
                )
        if self.params.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.net.parameters(), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay
                )
        if self.params.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.net.parameters(), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay
                )

        logger.info(f"Optimizer {self.params.optimizer} loaded.")


    def __initialize_training_variables(self):

        logger.info("Initializing training variables...")
        
        self.starting_epoch = 0
        self.step = 0
        self.train_eval_metric = 0.0
        self.train_loss = None
        self.valid_eval_metric = 50.0


        logger.info("Training variables initialized.")


    # Training methods
    
    
    def evaluate_training(self, prediction, label):

        logger.info(f"Evaluating training...")

        # Switch to evaluation
        self.net.eval()

        accuracy = Accuracy(prediction, label)
        logger.info(f"Accuracy on training set: {accuracy:.2f}")
        self.train_eval_metric = accuracy

        # Return to training
        self.net.train()


    def __extractInputFromFeature(self, sline):

        # logger.debug("Using __extractInputFromFeature")

        features1 = normalizeFeatures(
            featureReader(
                self.params.valid_data_dir + '/' + sline[0] + '.pickle'), 
                normalization=self.params.normalization,
                )
        features2 = normalizeFeatures(
            featureReader(
                self.params.valid_data_dir + '/' + sline[1] + '.pickle'), 
                normalization=self.params.normalization,
                )

        input1 = torch.FloatTensor(features1).to(self.device)
        input2 = torch.FloatTensor(features2).to(self.device)
        
        # logger.debug("__extractInputFromFeature used")
        
        return input1.unsqueeze(0), input2.unsqueeze(0)


    def __extract_scores(self, trials):

        logger.debug("Using __extract_scores")

        scores = []
        for line in trials:
            sline = line[:-1].split()

            input1, input2 = self.__extractInputFromFeature(sline)

            if torch.cuda.device_count() > 1:
                emb1, emb2 = self.net.module.getEmbedding(input1), self.net.module.getEmbedding(input2)
            else:
                emb1, emb2 = self.net.getEmbedding(input1), self.net.getEmbedding(input2)

            dist = scoreCosineDistance(emb1, emb2)
            scores.append(dist.item())

        logger.debug("__extract_scores used")
        
        return scores


    def __calculate_EER(self, CL, IM):

        logger.debug("Using __calculate_EER")

        thresholds = np.arange(-1,1,0.01)
        FRR, FAR = np.zeros(len(thresholds)), np.zeros(len(thresholds))
        for idx,th in enumerate(thresholds):
            FRR[idx] = Score(CL, th,'FRR')
            FAR[idx] = Score(IM, th,'FAR')

        EER_Idx = np.argwhere(np.diff(np.sign(FAR - FRR)) != 0).reshape(-1)
        if len(EER_Idx)>0:
            if len(EER_Idx)>1:
                EER_Idx = EER_Idx[0]
            EER = round((FAR[int(EER_Idx)] + FRR[int(EER_Idx)])/2,4)
        else:
            EER = 50.00

        logger.debug("__calculate_EER used")

        return EER


    def evaluate_validation(self):

        logger.info(f"Evaluating validation...")

        with torch.no_grad():

            # Switch to evaluation
            self.net.eval()

            # EER Validation
            with open(self.params.valid_clients,'r') as clients_in, open(self.params.valid_impostors,'r') as impostors_in:
                # score clients
                CL = self.__extract_scores(clients_in)
                IM = self.__extract_scores(impostors_in)
            
            # Compute EER
            EER = self.__calculate_EER(CL, IM)
            logger.info(f"EER on validation set: {EER:.2f}")
            self.valid_eval_metric = EER

        # Return to training
        self.net.train()


    def train_single_epoch(self, epoch):

        logger.info(f"Epoch {epoch}...")

        self.net.train()

        for self.batch_number, (input, label) in enumerate(self.training_generator):

            logger.info(f"Batch {self.batch_number} of {len(self.training_generator)}...")

            # Assign input and label to device
            input, label = input.float().to(self.device), label.long().to(self.device)

            # Calculate loss
            prediction, AMPrediction  = self.net(x = input, label = label, step = self.step)
            self.loss = self.loss_function(AMPrediction, label)
            self.train_loss = self.loss.item()
            logger.debug(f"Loss: {self.train_loss:.2f}")

            # Compute backpropagation and update weights
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            # DEBUG
            self.debug_step_info = {}
            self.debug_step_info['epoch'] = epoch
            self.debug_step_info['batch_number'] = self.batch_number
            self.debug_step_info['step'] = self.step
            self.debug_step_info['train_loss'] = self.train_loss
            self.debug_step_info['train_eval_metric'] = self.train_eval_metric
            self.debug_step_info['valid_eval_metric'] = self.valid_eval_metric
            self.debug_info.append(self.debug_step_info)

            self.step = self.step + 1

        self.evaluate_training(prediction, label)
        self.evaluate_validation()
        
        logger.info(f"-"*50)

    
    def train(self, starting_epoch, max_epochs):

        # TODO add a model.summary?

        logger.info(f'Starting training for {max_epochs} epochs.')

        # DEBUG
        self.debug_info = []

        for self.epoch in range(starting_epoch, max_epochs):  
            
            self.train_single_epoch(self.epoch)
            
        logger.info('Training finished!')


    def main(self):

        self.train(self.starting_epoch, self.params.max_epochs)


class ArgsParser:

    def __init__(self):
        
        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Train a VGG based Speaker Embedding Extractor.',
            )


    def add_parser_args(self):

        # TODO complete all helps
        
        # Directory parameters
        
        self.parser.add_argument(
            '--train_data_dir', 
            type = str, default = TRAIN_DEFAULT_SETTINGS['train_data_dir'],
            )
        
        self.parser.add_argument(
            '--train_labels_path', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['train_labels_path'],
            )

        self.parser.add_argument(
            '--valid_data_dir', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['valid_data_dir'], 
            
            )
        self.parser.add_argument(
            '--valid_clients', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['valid_clients'],
            )

        self.parser.add_argument(
            '--valid_impostors', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['valid_impostors'],
            )

        # Network Parameteres
        self.parser.add_argument(
            '--front_end', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['front_end'],
            choices = ['VGG3L','VGG4L', 'VGGNL'], 
            help = 'Type of Front-end used.'
            )

        self.parser.add_argument(
            '--window_size', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['window_size'], 
            help = '', # TODO this must me fixed, used in data
            )
        
        self.parser.add_argument(
            '--normalization', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['normalization'], 
            choices = ['cmn', 'cmvn'],
            help = 'Type of normalization applied to the features. \
                It can be Cepstral Mean Normalization or Cepstral Mean and Variance Normalization'
            )

        self.parser.add_argument(
            '--vgg_n_blocks', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['vgg_n_blocks'],
            help = 'Number of blocks the VGG front-end block will have.',
            )

        self.parser.add_argument(
            '--vgg_start_channels', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['vgg_start_channels'],
            help = 'Number of channels the first VGG convolutional block will have. \
                Each convolutional block will have the double of channels than the previous convolutional block.',
            )

        self.parser.add_argument(
            '--pooling_method', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['pooling_method'], 
            choices = ['Attention', 'MHA', 'DoubleMHA'], 
            help='Type of pooling methods',
            )

        self.parser.add_argument(
            '--heads_number', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['heads_number'],
            )

        self.parser.add_argument(
            '--mask_prob', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['mask_prob'], 
            help = 'Masking Drop Probability. Only Used for Only Double MHA',
            )

        self.parser.add_argument(
            '--embedding_size', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['embedding_size']
            )

        # AMSoftmax Config
        self.parser.add_argument(
            '--scaling_factor', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['scaling_factor'], 
            )

        self.parser.add_argument(
            '--margin_factor', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['margin_factor'],
            )

        self.parser.add_argument(
            '--annealing', 
            action = 'store_true',
            default = TRAIN_DEFAULT_SETTINGS['annealing'],
            )

        # Optimization arguments
        self.parser.add_argument(
            '--optimizer', 
            type = str, 
            choices = ['adam', 'sgd', 'rmsprop'], 
            default = TRAIN_DEFAULT_SETTINGS['optimizer'],
            )

        self.parser.add_argument(
            '--learning_rate', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['learning_rate'],
            )

        self.parser.add_argument(
            '--weight_decay', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['weight_decay'],
            )

        self.parser.add_argument(
            '--max_epochs',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['max_epochs'],
            help = 'Max number of epochs to train.',
            )

        self.parser.add_argument(
            '--batch_size', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['batch_size'],
            help = "Size of training batches.",
            )

        self.parser.add_argument(
            '--num_workers', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['num_workers'],
            )

        self.parser.add_argument(
            "--verbose", 
            action = "store_true", # TODO understand store_true vs store_false
            help = "Increase output verbosity.",
            )


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()


if __name__=="__main__":

    args_parser = ArgsParser()
    args_parser.main()
    parameters = args_parser.arguments

    trainer = Trainer(parameters)
    trainer.main()