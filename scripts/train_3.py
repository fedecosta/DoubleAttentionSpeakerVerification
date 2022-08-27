import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from data import Dataset
from model import SpeakerClassifier
from utils import getNumberOfSpeakers
from settings import TRAIN_DEFAULT_SETTINGS

from log_config import logger_dict
logger = logger_dict['train']


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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Running on {self.device} device.")
        
        if torch.cuda.device_count() > 1:
            logger.info(f"{torch.cuda.device_count()} GPUs available.")
        
        logger.info("Device setted.")


    # Load the Dataset
    def __load_data(self):
            
        logger.info(f'Loading Data and Labels from {self.params.train_labels_path}')
        
        with open(self.params.train_labels_path, 'r') as data_labels_file:
            train_labels = data_labels_file.readlines()

        data_loader_parameters = {
            'batch_size': self.params.batch_size, 
            'shuffle': True, # TODO hardcoded True
            'num_workers': self.params.num_workers
            }
        
        self.training_generator = DataLoader(
            Dataset(train_labels, self.params), 
            **data_loader_parameters,
            )

        logger.info("Data and labels loaded.")


    # Load the model (Neural Network)
    def __load_network(self):

        logger.info("Loading the network...")

        self.net = SpeakerClassifier(self.params, self.device)
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
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
        self.train_accuracy = 0
        self.train_loss = 0

        logger.info("Training variables initialized.")

    
    def train_single_epoch(self, epoch):

        logger.info(f"Epoch {epoch}...")

        self.net.train()

        for self.batch_number, (input, label) in enumerate(self.training_generator):

            logger.info(f"Batch {self.batch_number} of {len(self.training_generator)}...")

            input, label = input.float().to(self.device), label.long().to(self.device)

            # Calculate loss
            prediction, AMPrediction  = self.net(input = input, label = label, step = self.step)




            
            loss = loss_fn(prediction, target)

            # backpropagate error and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(f"loss: {loss.item()}")

        logger.info(f"-"*50)

    
    def train(self, starting_epoch, max_epochs):

        logger.info(f'Starting training for {max_epochs} epochs.')

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
        
        self.parser.add_argument(
            '--train_labels_path', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['train_labels_path']
            )

        # Network Parameteres
        self.parser.add_argument(
            '--front_end', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['front_end'],
            choices = ['VGG3L','VGG4L'], 
            help = 'Kind of Front-end Used'
            )

        self.parser.add_argument(
            '--kernel_size', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['kernel_size'],
            )

        self.parser.add_argument(
            '--pooling_method', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['pooling_method'], 
            choices = ['attention', 'mha', 'dmha'], 
            help='Type of pooling methods',
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
            )

        self.parser.add_argument(
            '--batch_size', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['batch_size'],
            )

        self.parser.add_argument(
            '--num_workers', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['num_workers'],
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

    trainer = Trainer(parameters)
    trainer.main()