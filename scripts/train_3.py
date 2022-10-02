import argparse
import os
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary

from data import Dataset, normalizeFeatures, featureReader
from model import SpeakerClassifier
from utils import get_number_of_speakers, generate_model_name, Accuracy, scoreCosineDistance, Score
from settings import TRAIN_DEFAULT_SETTINGS

# Set logging config
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%H:%M:%S',
    )

# Set a logging file handler
if not os.path.exists('./logs'):
    os.makedirs('./logs')
logger_file_handler = logging.FileHandler('./logs/train_3.log', mode = 'w')
logger_file_handler.setLevel(logging.DEBUG)
logger_file_handler.setFormatter(logger_formatter)

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_file_handler)
logger.addHandler(logger_stream_handler)


class Trainer:

    def __init__(self, input_params):

        self.set_device()
        self.set_random_seed()
        self.set_params(input_params)
        self.load_data()
        self.load_network()
        self.load_loss_function()
        self.load_optimizer()
        self.initialize_training_variables()


    # Init methods


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


    def set_params(self, input_params):

        logger.info("Setting params...")

        self.params = input_params
        self.params.number_speakers = get_number_of_speakers(self.params.train_labels_path)
        self.params.model_name = generate_model_name(self.params)

        if self.params.load_checkpoint == True:

            self.load_checkpoint()
            self.load_checkpoint_params()
            # When we load checkpoint params, all input params are overwriten. 
            # So we need to set load_checkpoint flag to True
            self.params.load_checkpoint = True
        
        logger.info("params setted.")


    def load_checkpoint(self):

        # Load checkpoint
        checkpoint_folder = self.params.model_output_folder
        checkpoint_file_name = f"{self.params.model_name}.chkpt"
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file_name)

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location = self.device)

        logger.info(f"Checkpoint loaded.")


    def load_checkpoint_params(self):

        logger.info(f"Loading checkpoint params...")

        self.params = self.checkpoint['settings']

        logger.info(f"Checkpoint params loaded.")


    def load_data(self):
            
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


    def load_checkpoint_network(self):

        logger.debug(f"Loading checkpoint network...")

        try:
            self.net.load_state_dict(self.checkpoint['model'])
        except RuntimeError:    
            self.net.module.load_state_dict(self.checkpoint['model'])

        logger.debug(f"Checkpoint network loaded.")


    def load_network(self):

        # Load the model (Neural Network)

        logger.info("Loading the network...")

        # Load model class
        self.net = SpeakerClassifier(self.params, self.device)
        
        if self.params.load_checkpoint == True:
            self.load_checkpoint_network()
        
        # Assign model to device
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.net = nn.DataParallel(self.net)
        
        # TODO set the correct size in this call to see the network summary
        # summary(self.net, torch.Size([80, 350]))
        logger.info("Network loaded.")


    def load_loss_function(self):

        logger.info("Loading the loss function...")

        self.loss_function = nn.CrossEntropyLoss()

        logger.info("Loss function loaded.")


    def load_checkpoint_optimizer(self):

        logger.debug(f"Loading checkpoint optimizer...")

        self.optimizer.load_state_dict(self.checkpoint['optimizer'])

        logger.debug(f"Checkpoint optimizer loaded.")


    def load_optimizer(self):

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

        if self.params.load_checkpoint == True:
            self.load_checkpoint_optimizer()

        logger.info(f"Optimizer {self.params.optimizer} loaded.")


    def initialize_training_variables(self):

        logger.info("Initializing training variables...")
        
        if self.params.load_checkpoint == True:

            logger.debug(f"Loading checkpoint training variables...")

            loaded_training_variables = self.checkpoint['training_variables']

            # HACK this can be refined, but we are going to continue training \
            # from the last epoch trained and from the first batch
            # (even if we may already trained with some batches in that epoch in the last training from the checkpoint).
            self.starting_epoch = loaded_training_variables['epoch']
            self.step = loaded_training_variables['step'] + 1 
            self.validations_without_improvement = loaded_training_variables['validations_without_improvement'] 
            self.early_stopping_flag = False
            self.train_loss = loaded_training_variables['train_loss'] 
            self.train_eval_metric = loaded_training_variables['train_eval_metric'] 
            self.valid_eval_metric = loaded_training_variables['valid_eval_metric'] 
            self.best_train_loss = loaded_training_variables['best_train_loss'] 
            self.best_model_train_loss = loaded_training_variables['best_model_train_loss'] 
            self.best_model_train_eval_metric = loaded_training_variables['best_model_train_eval_metric'] 
            self.best_model_valid_eval_metric = loaded_training_variables['best_model_valid_eval_metric']

            logger.info(f"Checkpoint training variables loaded.") 
            logger.info(f"Training will start from:")
            logger.info(f"Epoch {self.starting_epoch}")
            logger.info(f"Step {self.step}")
            logger.info(f"validations_without_improvement {self.validations_without_improvement}")
            logger.info(f"Loss {self.train_loss:.2f}")
            logger.info(f"best_model_train_loss {self.best_model_train_loss:.2f}")
            logger.info(f"best_model_train_eval_metric {self.best_model_train_eval_metric:.2f}")
            logger.info(f"best_model_valid_eval_metric {self.best_model_valid_eval_metric:.2f}")

        else:
            self.starting_epoch = 0
            self.step = 0 
            self.validations_without_improvement = 0 
            self.early_stopping_flag = False
            self.train_loss = None
            self.train_eval_metric = 0.0
            self.valid_eval_metric = 50.0
            self.best_train_loss = np.inf
            self.best_model_train_loss = np.inf
            self.best_model_train_eval_metric = 0.0
            self.best_model_valid_eval_metric = 50.0

        logger.info("Training variables initialized.")


    # Training methods


    def random_slice(self, inputTensor):
        '''Takes a random slice of the tensor cutting the frames axis, from 0 to a random end point.'''

        # TODO is this method usefull? maybe is a feature extractor task

        index = random.randrange(200, self.params.window_size * 100) # HACK fix this harcoded 200
        
        return inputTensor[:,:index,:]


    def evaluate_training(self, prediction, label):

        logger.info(f"Evaluating training...")

        # Switch torch to evaluation mode
        self.net.eval()
        accuracy = Accuracy(prediction, label)
        
        self.train_eval_metric = accuracy

        # Return to torch training mode
        self.net.train()

        logger.info(f"Training evaluated.")
        logger.info(f"Accuracy on training set: {accuracy:.2f}")


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
                emb1, emb2 = self.net.module.get_embedding(input1), self.net.module.get_embedding(input2)
            else:
                emb1, emb2 = self.net.get_embedding(input1), self.net.get_embedding(input2)

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

            # Switch torch to evaluation mode
            self.net.eval()

            # EER Validation
            with open(self.params.valid_clients,'r') as clients_in, open(self.params.valid_impostors,'r') as impostors_in:
                # score clients
                CL = self.__extract_scores(clients_in)
                IM = self.__extract_scores(impostors_in)
            
            # Compute EER
            EER = self.__calculate_EER(CL, IM)
            self.valid_eval_metric = EER

        # Return to training mode
        self.net.train()

        logger.info(f"EER on validation set: {EER:.2f}")
        logger.info(f"Validation evaluated.")


    def evaluate(self, prediction, label):

        self.evaluate_training(prediction, label)
        self.evaluate_validation()


    def save_model(self):

        '''Function to save the model info and optimizer parameters.'''

        model_results = {
            'train_loss' : self.best_model_train_loss,
            'train_eval_metric' : self.best_model_train_eval_metric,
            'valid_eval_metric' : self.best_model_valid_eval_metric,
        }

        training_variables = {
            'epoch': self.epoch,
            'step' : self.step,
            'validations_without_improvement' : self.validations_without_improvement,
            'train_loss' : self.train_loss,
            'train_eval_metric' : self.train_eval_metric,
            'valid_eval_metric' : self.valid_eval_metric,
            'best_train_loss' : self.best_train_loss,
            'best_model_train_loss' : self.best_model_train_loss,
            'best_model_train_eval_metric' : self.best_model_train_eval_metric,
            'best_model_valid_eval_metric' : self.best_model_valid_eval_metric,
        }
        
        if torch.cuda.device_count() > 1:
            checkpoint = {
                'model': self.net.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'settings': self.params,
                'model_results' : model_results,
                'training_variables' : training_variables,
                }
        else:
            checkpoint = {
                'model': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'settings': self.params,
                'model_results' : model_results,
                'training_variables' : training_variables,
                }

        # We will save this checkpoint and it will overwrite the last one of this model
        checkpoint_folder = self.params.model_output_folder
        checkpoint_file_name = f"{self.params.model_name}.chkpt"
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file_name)

        # Create directory if doesn't exists
        if not os.path.exists(self.params.model_output_folder):
            os.makedirs(self.params.model_output_folder)

        logger.info(f"Saving training and model information in {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

        # DEBUG
        # We can also save the checkpoint with epoch and step for debug
        # Warning! It will save a lot of checkpoints
        # checkpoint_file_name_2 = f"{self.params.model_name}_epoch{self.epoch}_step{self.step}.chkpt"
        # checkpoint_path_2 = os.path.join(checkpoint_folder, checkpoint_file_name_2)
        # torch.save(checkpoint, checkpoint_path_2)

        logger.info(f"Training and model information saved.")


    def eval_and_save_best_model(self, prediction, label):

        if self.step > 0 and self.params.eval_and_save_best_model_every > 0 \
            and self.step % self.params.eval_and_save_best_model_every == 0:

            logger.info('Evaluating and saving the new best model (if founded)...')

            # Calculate the evaluation metrics
            self.evaluate(prediction, label)

            # Have we found a better model? (Better in validation metric).
            if self.valid_eval_metric < self.best_model_valid_eval_metric:

                logger.info('We found a better model!')

                # Update best model evaluation metrics
                self.best_model_train_loss = self.train_loss
                self.best_model_train_eval_metric = self.train_eval_metric
                self.best_model_valid_eval_metric = self.valid_eval_metric

                logger.info(f"Best model train loss: {self.best_model_train_loss:.2f}")
                logger.info(f"Best model train evaluation metric: {self.best_model_train_eval_metric:.2f}")
                logger.info(f"Best model validation evaluation metric: {self.best_model_valid_eval_metric:.2f}")
                self.save_model() 

                # Since we found and improvement, validations_without_improvement is reseted.
                self.validations_without_improvement = 0
            
            else:
                # In this case the search didn't improved the model
                # We are one validation closer to do early stopping
                self.validations_without_improvement = self.validations_without_improvement + 1

            logger.info(f"Consecutive validations without improvement: {self.validations_without_improvement}")
            logger.info('Evaluating and saving done.')


    def check_update_optimizer(self):

        if self.validations_without_improvement > 0 and self.params.update_optimizer_every > 0 \
            and self.validations_without_improvement % self.params.update_optimizer_every == 0:

            if self.params.optimizer == 'sgd' or self.params.optimizer == 'adam':

                logger.debug(f"Updating optimizer...")

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                
                
                logger.debug(f"Optimizer updated.")
                logger.info(f"New learning rate: {param_group['lr']}")


    def check_early_stopping(self):

        if self.params.early_stopping > 0 \
            and self.validations_without_improvement >= self.params.early_stopping:

            self.early_stopping_flag = True
            logger.info(f"Doing early stopping after {self.validations_without_improvement} validations without improvement.")


    def check_print_training_info(self):
        
        if self.step > 0 and self.params.print_training_info_every > 0 \
            and self.step % self.params.print_training_info_every == 0:

            logger.info(f"Step: {self.step}")
            logger.info(f"Best loss achieved: {self.best_train_loss:.2f}")
            logger.info(f"Best model training evaluation metric: {self.best_model_train_eval_metric:.2f}")
            logger.info(f"Best model validation evaluation metric: {self.best_model_valid_eval_metric:.2f}")


    def train_single_epoch(self, epoch):

        logger.info(f"Epoch {epoch} of {self.params.max_epochs}...")

        # Switch torch to training mode
        self.net.train()

        for self.batch_number, (input, label) in enumerate(self.training_generator):

            logger.info(f"Batch {self.batch_number} of {len(self.training_generator)}...")

            # Assign input and label to device
            input, label = input.float().to(self.device), label.long().to(self.device)

            # Slice at random using the frames axis, if desired.
            input = self.random_slice(input) if self.params.random_slicing else input

            # logger.debug(f"input size: {input.size()}")

            # Calculate loss
            prediction, AMPrediction  = self.net(input_tensor = input, label = label, step = self.step) # TODO understand diff between prediction and AMPrediction
            self.loss = self.loss_function(AMPrediction, label)
            self.train_loss = self.loss.item()
            logger.info(f"Actual loss: {self.train_loss:.2f}")

            # Compute backpropagation and update weights
            
            # Clears x.grad for every parameter x in the optimizer. 
            # It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
            self.optimizer.zero_grad()
            
            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. 
            # These are accumulated into x.grad for every parameter x.
            self.loss.backward()
            
            # optimizer.step updates the value of x using the gradient x.grad
            self.optimizer.step()

            # Calculate evaluation metrics and save the best model
            self.eval_and_save_best_model(prediction, label)

            # Update best loss
            if self.train_loss < self.best_train_loss:
                self.best_train_loss = self.train_loss

            self.check_update_optimizer()
            self.check_early_stopping()
            self.check_print_training_info()

            # DEBUG
            self.debug_step_info = {}
            self.debug_step_info['epoch'] = epoch
            self.debug_step_info['batch_number'] = self.batch_number
            self.debug_step_info['step'] = self.step
            self.debug_step_info['train_loss'] = self.train_loss
            self.debug_step_info['train_eval_metric'] = self.train_eval_metric
            self.debug_step_info['valid_eval_metric'] = self.valid_eval_metric
            self.debug_step_info['best_train_loss'] = self.best_train_loss
            self.debug_step_info['best_model_train_eval_metric'] = self.best_model_train_eval_metric
            self.debug_step_info['best_model_valid_eval_metric'] = self.best_model_valid_eval_metric
            self.debug_step_info['validations_without_improvement'] = self.validations_without_improvement
            self.debug_info.append(self.debug_step_info)

            if self.early_stopping_flag == True: 
                break

            self.step = self.step + 1
        
        # DEBUG
        #self.debug_step_info['train_eval_metric'] = self.train_eval_metric
        #self.debug_step_info['valid_eval_metric'] = self.valid_eval_metric
        #self.debug_info[-1] = self.debug_step_info

        logger.info(f"-"*50)
        logger.info(f"Epoch {epoch} finished with:")
        logger.info(f"Loss {self.train_loss:.2f}")
        logger.info(f"Best model training evaluation metric: {self.best_model_train_eval_metric:.2f}")
        logger.info(f"Best model validation evaluation metric: {self.best_model_valid_eval_metric:.2f}")
        logger.info(f"-"*50)

    
    def train(self, starting_epoch, max_epochs):

        logger.info(f'Starting training for {max_epochs} epochs.')

        # DEBUG
        self.debug_info = []

        for self.epoch in range(starting_epoch, max_epochs):  
            
            self.train_single_epoch(self.epoch)

            if self.early_stopping_flag == True: 
                break
            
        logger.info('Training finished!')


    # TODO I think that this is redundant, parameters are saved with the model in checkpoint
    def save_input_params(self):
        '''Save the input argparse params into a pickle file.'''

        # Create directory if doesn't exists
        if not os.path.exists(self.params.model_output_folder):
            os.makedirs(self.params.model_output_folder)

        # Save argparse input params
        config_file_name = f"{self.params.model_name}_config.pickle" 
        config_file_dir = os.path.join(self.params.model_output_folder, config_file_name)
        with open(config_file_dir, 'wb') as handle:
            pickle.dump(self.params, handle, protocol = pickle.HIGHEST_PROTOCOL)

    
    # TODO I think that this is redundant, parameters are saved with the model in checkpoint
    def load_input_params(self, load_folder, load_file_name):

        load_path = os.path.join(load_folder, load_file_name)
        file = open(load_path,'rb')
        namespace = pickle.load(file)

        return namespace
    

    def main(self):

        # self.save_input_params() TODO I think this is useless
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
            '--train_labels_path', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['train_labels_path'],
            help = 'Path of the file containing the training examples paths and labels.',
            )
        
        self.parser.add_argument(
            '--train_data_dir', 
            type = str, default = TRAIN_DEFAULT_SETTINGS['train_data_dir'],
            help = 'Optional additional directory to prepend to the train_labels_path paths.',
            )
        
        self.parser.add_argument(
            '--valid_clients', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['valid_clients'],
            help = 'Path of the file containing the validation clients pairs paths.',
            )

        self.parser.add_argument(
            '--valid_impostors', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['valid_impostors'],
            help = 'Path of the file containing the validation impostors pairs paths.',
            )

        self.parser.add_argument(
            '--valid_data_dir', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['valid_data_dir'], 
            help = 'Optional additional directory to prepend to valid_clients and valid_impostors paths.',
            )

        self.parser.add_argument(
            '--model_output_folder', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['model_output_folder'], 
            help = 'Directory where model outputs and configs are saved.',
            )

        # Training Parameters

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
            '--eval_and_save_best_model_every', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['eval_and_save_best_model_every'],
            help = "The model is evaluated on train and validation sets every eval_and_save_best_model_every steps. \
                Set to 0 if you don't want to execute this utility.",
            )
        
        self.parser.add_argument(
            '--print_training_info_every', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['print_training_info_every'],
            help = "Training info is printed every print_training_info_every steps. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--early_stopping', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['early_stopping'],
            help = "Training is stopped if there are early_stopping consectuive validations without improvement. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--update_optimizer_every', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['update_optimizer_every'],
            help = "Some optimizer parameters will be updated every update_optimizer_every consectuive validations without improvement. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--load_checkpoint',
            action = 'store_true',
            default = TRAIN_DEFAULT_SETTINGS['load_checkpoint'],
            help = 'Set to True if you want to load a previous checkpoint and continue training from that point. \
                Loaded parameters will overwrite all inputted parameters.',
            )

        # Data Parameters

        self.parser.add_argument(
            '--window_size', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['window_size'], 
            help = 'Cut the input spectrogram with window_size length at a random starting point. \
                window_size is measured in #frames / 100.', # TODO this should be improved, it is used at data.py
            )

        self.parser.add_argument(
            '--normalization', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['normalization'], 
            choices = ['cmn', 'cmvn'],
            help = 'Type of normalization applied to the features. \
                It can be Cepstral Mean Normalization or Cepstral Mean and Variance Normalization',
            )

        self.parser.add_argument(
            '--num_workers', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['num_workers'],
            help = 'num_workers to be used by the data loader'
            )

        self.parser.add_argument(
            '--random_slicing', 
            action = 'store_true',
            default = TRAIN_DEFAULT_SETTINGS['random_slicing'],
            help = 'Whether to do random slicing or not. This slice the inputs randomly at frames axis.',
            )

        # Network Parameters

        self.parser.add_argument(
            '--model_name_prefix', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['model_name_prefix'], 
            help = 'Give the model a name prefix for saving it.'
            )

        self.parser.add_argument(
            '--front_end', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['front_end'],
            choices = ['VGGNL'], 
            help = 'Type of Front-end used. VGGNL for a N-block VGG architecture.'
            )
            
        self.parser.add_argument(
            '--vgg_n_blocks', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['vgg_n_blocks'],
            help = 'Number of blocks the VGG front-end block will have.',
            )

        self.parser.add_argument(
            '--vgg_channels', 
            nargs = '+',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['vgg_channels'],
            help = 'Number of channels each VGG convolutional block will have. \
                The number of channels must be passed in order and consisently with vgg_n_blocks.',
            )

        self.parser.add_argument(
            '--pooling_method', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['pooling_method'], 
            choices = ['Attention', 'MHA', 'DoubleMHA'], 
            help = 'Type of pooling method.',
            )

        self.parser.add_argument(
            '--heads_number', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['heads_number'],
            help = 'Number of heads for the pooling method (only for MHA and DoubleMHA options).',
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
            default = TRAIN_DEFAULT_SETTINGS['embedding_size'],
            help = 'Size of the embedding that the system will generate.',
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

        # Verbosity and debug Parameters
        
        self.parser.add_argument(
            "--verbose", 
            action = "store_true", # TODO understand store_true vs store_false
            default = TRAIN_DEFAULT_SETTINGS['verbose'],
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