import argparse
import os
import numpy as np
import random
import pickle
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary

from data import Dataset, normalizeFeatures, featureReader
from model import SpeakerClassifier
from utils import get_number_of_speakers, generate_model_name, Accuracy, scoreCosineDistance, Score, get_memory_info
from settings import TRAIN_DEFAULT_SETTINGS

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
run = wandb.init(project = "speaker_verification_models", job_type = "training")


class Trainer:

    def __init__(self, input_params):

        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        self.set_device()
        self.set_random_seed()
        self.set_params(input_params)
        self.set_log_file_handler()
        self.load_data()
        self.load_network()
        self.load_loss_function()
        self.load_optimizer()
        self.initialize_training_variables()
        self.config_wandb()


    # Init methods


    def info_mem(self, step = None):
        cpu_available_pctg, gpu_free = get_memory_info()
        if step is not None:
            logger.info(f"Step {self.step}: CPU available {cpu_available_pctg:.2f}% - GPU free {gpu_free}")
        else:
            logger.info(f"CPU available {cpu_available_pctg:.2f}% - GPU free {gpu_free}")


    def set_params(self, input_params):

        logger.info("Setting params...")

        self.params = input_params
        self.params.number_speakers = get_number_of_speakers(self.params.train_labels_path)
        self.params.model_name = generate_model_name(
            self.params, 
            start_datetime = self.start_datetime, 
            wandb_run_id = wandb.run.id,
            )

        if self.params.load_checkpoint == True:

            self.load_checkpoint()
            self.load_checkpoint_params()
            # When we load checkpoint params, all input params are overwriten. 
            # So we need to set load_checkpoint flag to True
            self.params.load_checkpoint = True
        
        logger.info("params setted.")


    def load_checkpoint(self):

        # Load checkpoint
        checkpoint_folder = self.params.checkpoint_file_folder
        checkpoint_file_name = self.params.checkpoint_file_name
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file_name)

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location = self.device)

        logger.info(f"Checkpoint loaded.")


    def load_checkpoint_params(self):

        logger.info(f"Loading checkpoint params...")

        self.params = self.checkpoint['settings']

        logger.info(f"Checkpoint params loaded.")


    def set_log_file_handler(self):

        # Set a logging file handler

        if not os.path.exists(self.params.log_file_folder):
            os.makedirs(self.params.log_file_folder)
        
        logger_file_name = f"{self.params.model_name}.log"
        logger_file_path = os.path.join(self.params.log_file_folder, logger_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
        logger_file_handler.setLevel(logging.INFO) # TODO set the file handler level as a input param
        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler)


    def set_device(self):
        
        logger.info('Setting device...')

        # Set device to GPU or CPU depending on what is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Running on {self.device} device.")
        
        self.gpus = torch.cuda.device_count()
        logger.info(f"{self.gpus} GPUs available.")
        
        logger.info("Device setted.")

    
    def set_random_seed(self):

        logger.info("Setting random seed...")

        # Set the seed for experimental reproduction
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        logger.info("Random seed setted.")


    def set_random_crop_size(self, pickle_path):

        logger.info(f'Defining the random crop size in frames from a sample...')
        logger.info(f'(We are assuming that all files in the dataset have the same spectrogram settings).')
        
        # Takes a path of a .pickle file that has a spectrogram matrix and its settings parameters.
        # Given a random crop size in seconds, we calculate the equivalent size in frames.
        # For this it is crucial to know the setting parameters of the spectrogram.

        # Load the file
        with open(pickle_path, 'rb') as pickle_file:
            features_dict = pickle.load(pickle_file)
            
        # Unpack the spectrogram and the settings
        features = features_dict["features"]
        features_settings = features_dict["settings"]
        
        # Get the setting parameters
        file_frames = features.shape[0]
        sampling_rate = features_settings.sampling_rate
        hop_length = int(features_settings.hop_length_secs * sampling_rate)
        n_fft = int(features_settings.n_fft_secs * sampling_rate)

        # Estimate the random crop size in frames
        estimated_samples = (file_frames - 1) * hop_length + n_fft
        estimated_audio_length_secs = estimated_samples / sampling_rate
        estimated_frames_1_sec = file_frames / estimated_audio_length_secs 
        
        self.params.random_crop_frames = int(self.params.random_crop_secs * estimated_frames_1_sec)

        logger.info(f'Random crop size calculated: {self.params.random_crop_frames} frames (eq to {self.params.random_crop_secs} seconds).')
        

    def format_train_labels(self):

        # Read the paths of the train audios and their labels
        with open(self.params.train_labels_path, 'r') as data_labels_file:
            train_labels = data_labels_file.readlines()

        # If train labels are of the form /speaker/interview/file we need to remove the first "/" to join paths
        self.train_labels = []
        for train_label in train_labels:
            if train_label[0] == "/":
                train_label = train_label[1:]
            self.train_labels.append(train_label)
        
        # We prepend train_data_dir to the paths
        self.train_labels = [os.path.join(self.params.train_data_dir, train_label) for train_label in self.train_labels]


    def format_valid_labels(self, labels_path):

        # Read the paths of the train audios and their labels
        with open(labels_path, 'r') as data_labels_file:
            valid_labels = data_labels_file.readlines()

        # If valid labels are of the form /speaker/interview/file /speaker/interview/file we need to remove the first "/" of each case to join paths
        final_valid_labels = []
        for valid_label in valid_labels:
            speaker_1 = valid_label.split(" ")[0]
            speaker_2 = valid_label.split(" ")[1]
            if speaker_1[0] == "/":
                speaker_1 = speaker_1[1:]
            if speaker_2[0] == "/":
                speaker_2 = speaker_2[1:]
            speaker_1 = os.path.join(self.params.valid_data_dir, speaker_1)
            speaker_2 = os.path.join(self.params.valid_data_dir, speaker_2)
            
            final_valid_labels.append(f"{speaker_1} {speaker_2}")
    
        return final_valid_labels


    def format_labels(self):

        self.format_train_labels()
        self.valid_clients_labels = self.format_valid_labels(self.params.valid_clients_path)
        self.valid_impostors_labels = self.format_valid_labels(self.params.valid_impostors_path)
        

    def load_data(self):
            
        logger.info(f'Loading data and labels from {self.params.train_labels_path}')

        self.format_labels()

        # Get one sample to calculate the random crop size in frames for all the dataset
        representative_sample = self.train_labels[0]
        pickle_path = representative_sample.replace('\n', '').split(' ')[0]
        self.set_random_crop_size(pickle_path)

        # Instanciate a Dataset class
        dataset = Dataset(self.train_labels, self.params)
        
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

        del representative_sample
        del dataset

        logger.info("Data and labels loaded.")


    def load_checkpoint_network(self):

        logger.info(f"Loading checkpoint network...")

        try:
            self.net.load_state_dict(self.checkpoint['model'])
        except RuntimeError:    
            self.net.module.load_state_dict(self.checkpoint['model'])

        logger.info(f"Checkpoint network loaded.")


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
        
        summary(self.net, input_tensor = (350, 80))

        # Calculate trainable parameters (to estimate model complexity)
        self.total_trainable_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )

        logger.info("Network loaded.")


    def load_loss_function(self):

        logger.info("Loading the loss function...")

        self.loss_function = nn.CrossEntropyLoss()

        logger.info("Loss function loaded.")


    def load_checkpoint_optimizer(self):

        logger.info(f"Loading checkpoint optimizer...")

        self.optimizer.load_state_dict(self.checkpoint['optimizer'])

        logger.info(f"Checkpoint optimizer loaded.")


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

            logger.info(f"Loading checkpoint training variables...")

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
            logger.info(f"Loss {self.train_loss:.3f}")
            logger.info(f"best_model_train_loss {self.best_model_train_loss:.3f}")
            logger.info(f"best_model_train_eval_metric {self.best_model_train_eval_metric:.3f}")
            logger.info(f"best_model_valid_eval_metric {self.best_model_valid_eval_metric:.3f}")

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
        
        self.total_batches = len(self.training_generator)

        logger.info("Training variables initialized.")


    def config_wandb(self):

        # 1 - Save the params
        self.wandb_config = vars(self.params)

        # 2 - Save the feature extraction configuration params

        # Get one sample to get the feature extraction configuration for all the dataset
        representative_sample = self.train_labels[0]
        pickle_path = representative_sample.replace('\n', '').split(' ')[0]

        # Load the file
        with open(pickle_path, 'rb') as pickle_file:
            features_dict = pickle.load(pickle_file)
            
        # Unpack the spectrogram and the features settings
        # features = features_dict["features"]
        dev_features_settings = features_dict["settings"]

        self.wandb_config["dev_features_settings"] = vars(dev_features_settings)

        # 3 - Save additional params

        self.wandb_config["total_trainable_params"] = self.total_trainable_params
        self.wandb_config["gpus"] = self.gpus

        # 4 - Update the wandb config
        wandb.config.update(self.wandb_config)

        del representative_sample
        del features_dict
        del dev_features_settings


    # Training methods


    def evaluate_training(self, prediction, label):

        logger.info(f"Evaluating training...")

        # Switch torch to evaluation mode
        self.net.eval()
        accuracy = Accuracy(prediction, label)
        
        self.train_eval_metric = accuracy

        # Return to torch training mode
        self.net.train()

        logger.info(f"Training evaluated.")
        logger.info(f"Accuracy on training set: {accuracy:.3f}")
        #self.info_mem(self.step)


    def extractInputFromFeature(self, sline):

        logger.debug("Using extractInputFromFeature")

        features1 = normalizeFeatures(
            featureReader(sline[0]), 
            normalization = self.params.normalization,
            )
        features2 = normalizeFeatures(
            featureReader(sline[1]), 
            normalization = self.params.normalization,
            )

        input1 = torch.FloatTensor(features1).to(self.device)
        input2 = torch.FloatTensor(features2).to(self.device)
        
        logger.debug("extractInputFromFeature used")
        
        return input1.unsqueeze(0), input2.unsqueeze(0)


    def extract_scores(self, trials):

        logger.debug("Using extract_scores")

        scores = []
        for line in trials:
            sline = line[:-1].split()

            input1, input2 = self.extractInputFromFeature(sline)

            if torch.cuda.device_count() > 1:
                emb1, emb2 = self.net.module.get_embedding(input1), self.net.module.get_embedding(input2)
            else:
                emb1, emb2 = self.net.get_embedding(input1), self.net.get_embedding(input2)

            dist = scoreCosineDistance(emb1, emb2)
            scores.append(dist.item())

        logger.debug("extract_scores used")
        
        return scores


    def calculate_EER(self, CL, IM):

        logger.debug("Using calculate_EER")

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

        logger.debug("calculate_EER used")

        return EER


    def evaluate_validation(self):

        logger.info(f"Evaluating validation...")

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            # EER Validation score clients
            CL = self.extract_scores(self.valid_clients_labels)
            IM = self.extract_scores(self.valid_impostors_labels)
            
            # Compute EER
            EER = self.calculate_EER(CL, IM)
            self.valid_eval_metric = EER

        # Return to training mode
        self.net.train()

        logger.info(f"EER on validation set: {EER:.3f}")
        logger.info(f"Validation evaluated.")
        #self.info_mem(self.step)


    def evaluate(self, prediction, label):

        self.evaluate_training(prediction, label)
        self.evaluate_validation()


    def save_model(self):

        '''Function to save the model info and optimizer parameters.'''

        # 1 - Add all the info that will be saved in checkpoint 
        
        model_results = {
            'best_model_train_loss' : self.best_model_train_loss,
            'best_model_train_eval_metric' : self.best_model_train_eval_metric,
            'best_model_valid_eval_metric' : self.best_model_valid_eval_metric,
        }

        training_variables = {
            'epoch': self.epoch,
            'batch_number' : self.batch_number,
            'step' : self.step,
            'validations_without_improvement' : self.validations_without_improvement,
            'train_loss' : self.train_loss,
            'train_eval_metric' : self.train_eval_metric,
            'valid_eval_metric' : self.valid_eval_metric,
            'best_train_loss' : self.best_train_loss,
            'best_model_train_loss' : self.best_model_train_loss,
            'best_model_train_eval_metric' : self.best_model_train_eval_metric,
            'best_model_valid_eval_metric' : self.best_model_valid_eval_metric,
            'total_trainable_params' : self.total_trainable_params,
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

        end_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        checkpoint['start_datetime'] = self.start_datetime
        checkpoint['end_datetime'] = end_datetime

        # 2 - Save the checkpoint locally

        checkpoint_folder = os.path.join(self.params.model_output_folder, self.params.model_name)
        checkpoint_file_name = f"{self.params.model_name}.chkpt"
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file_name)

        # Create directory if doesn't exists
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        logger.info(f"Saving training and model information in {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Done.")

        # 3 - Save checkpoint as a wandb artifact

        # Add checkpoint data to wandb config
        #self.wandb_config["model_results"] = model_results
        #self.wandb_config["training_variables"] = training_variables
        #wandb.config.update(self.wandb_config)

        # Define the artifact
        trained_model_artifact = wandb.Artifact(
            name = self.params.model_name,
            type = "trained_model",
            description = self.params.model_name_prefix, # TODO set as an argparse input param
            metadata = self.wandb_config,
        )

        # Add folder directory
        trained_model_artifact.add_dir(checkpoint_folder)

        # Log the artifact
        run.log_artifact(trained_model_artifact)

        # Delete variables to free memory
        del model_results
        del training_variables
        del checkpoint
        del trained_model_artifact

        logger.info(f"Training and model information saved.")
        #self.info_mem(self.step)


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

                logger.info(f"Best model train loss: {self.best_model_train_loss:.3f}")
                logger.info(f"Best model train evaluation metric: {self.best_model_train_eval_metric:.3f}")
                logger.info(f"Best model validation evaluation metric: {self.best_model_valid_eval_metric:.3f}")

                self.save_model() 

                # Since we found and improvement, validations_without_improvement is reseted.
                self.validations_without_improvement = 0
            
            else:
                # In this case the search didn't improved the model
                # We are one validation closer to do early stopping
                self.validations_without_improvement = self.validations_without_improvement + 1

            logger.info(f"Consecutive validations without improvement: {self.validations_without_improvement}")
            logger.info('Evaluating and saving done.')
            #self.info_mem(self.step)


    def check_update_optimizer(self):

        if self.validations_without_improvement > 0 and self.params.update_optimizer_every > 0 \
            and self.validations_without_improvement % self.params.update_optimizer_every == 0:

            if self.params.optimizer == 'sgd' or self.params.optimizer == 'adam':

                logger.info(f"Updating optimizer...")

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                
                
                logger.info(f"Optimizer updated.")
                logger.info(f"New learning rate: {param_group['lr']}")


    def check_early_stopping(self):

        if self.params.early_stopping > 0 \
            and self.validations_without_improvement >= self.params.early_stopping:

            self.early_stopping_flag = True
            logger.info(f"Doing early stopping after {self.validations_without_improvement} validations without improvement.")


    def check_print_training_info(self):
        
        if self.step > 0 and self.params.print_training_info_every > 0 \
            and self.step % self.params.print_training_info_every == 0:

            info_to_print = f"Epoch {self.epoch} of {self.params.max_epochs}, "
            info_to_print = info_to_print + f"batch {self.batch_number} of {self.total_batches}, "
            info_to_print = info_to_print + f"step {self.step}, "
            info_to_print = info_to_print + f"Loss {self.train_loss:.3f}, "
            info_to_print = info_to_print + f"Best EER {self.best_model_valid_eval_metric:.3f}..."

            logger.info(info_to_print)
            
            # Uncomment for memory usage info 
            #self.info_mem(self.step)

            
    def train_single_epoch(self, epoch):

        logger.info(f"Epoch {epoch} of {self.params.max_epochs}...")

        # Switch torch to training mode
        self.net.train()

        for self.batch_number, (input, label) in enumerate(self.training_generator):

            # Assign input and label to device
            input, label = input.float().to(self.device), label.long().to(self.device)

            # Calculate prediction and loss
            prediction, inner_products_m_s  = self.net(input_tensor = input, label = label)
            self.loss = self.loss_function(inner_products_m_s, label)
            self.train_loss = self.loss.item()

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

            try:
                wandb.log(
                    {
                        "epoch" : self.epoch,
                        "batch_number" : self.batch_number,
                        "loss": self.train_loss,
                        "train_eval_metric" : self.train_eval_metric,
                        "valid_eval_metric" : self.valid_eval_metric,
                        'best_model_train_loss' : self.best_model_train_loss,
                        'best_model_train_eval_metric' : self.best_model_train_eval_metric,
                        'best_model_valid_eval_metric' : self.best_model_valid_eval_metric,
                    },
                    step = self.step
                    )
            except Exception as e:
                logger.error('Failed at wandb.log: '+ str(e))

            if self.early_stopping_flag == True: 
                break
            
            self.step = self.step + 1

        logger.info(f"-"*50)
        logger.info(f"Epoch {epoch} finished with:")
        logger.info(f"Loss {self.train_loss:.3f}")
        logger.info(f"Best model training evaluation metric: {self.best_model_train_eval_metric:.3f}")
        logger.info(f"Best model validation evaluation metric: {self.best_model_valid_eval_metric:.3f}")
        logger.info(f"-"*50)

    
    def train(self, starting_epoch, max_epochs):

        logger.info(f'Starting training for {max_epochs} epochs.')

        for self.epoch in range(starting_epoch, max_epochs):  
            
            self.train_single_epoch(self.epoch)

            if self.early_stopping_flag == True: 
                break
            
        logger.info('Training finished!')
    

    def delete_version_artifacts(self):

        logger.info(f'Starting to delete not latest checkpoint version artifacts...')

        # We want to keep only the latest checkpoint because of wandb memory storage limit

        api = wandb.Api()
        actual_run = api.run(f"{run.entity}/{run.project}/{run.id}")
        
        # We need to finish the run and let wandb upload all files
        run.finish()

        for artifact_version in actual_run.logged_artifacts():
            
            if 'latest' in artifact_version.aliases:
                latest_version = True
            else:
                latest_version = False

            if latest_version == False:
                logger.info(f'Deleting not latest artifact {artifact_version.name} from wandb...')
                artifact_version.delete(delete_aliases=True)
                logger.info(f'Deleted.')

        logger.info(f'All not latest artifacts deleted.')


    def main(self):

        self.train(self.starting_epoch, self.params.max_epochs)
        self.delete_version_artifacts()


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
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['train_data_dir'],
            help = 'Optional additional directory to prepend to the train_labels_path paths.',
            )
        
        self.parser.add_argument(
            '--valid_clients_path', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['valid_clients_path'],
            help = 'Path of the file containing the validation clients pairs paths.',
            )

        self.parser.add_argument(
            '--valid_impostors_path', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['valid_impostors_path'],
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

        self.parser.add_argument(
            '--log_file_folder',
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['log_file_folder'],
            help = 'Name of folder that will contain the log file.',
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
            help = "The model is evaluated on train and validation sets and saved every eval_and_save_best_model_every steps. \
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
            help = "Some optimizer parameters will be updated every update_optimizer_every consecutive validations without improvement. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--load_checkpoint',
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['load_checkpoint'],
            help = 'Set to True if you want to load a previous checkpoint and continue training from that point. \
                Loaded parameters will overwrite all inputted parameters.',
            )

        self.parser.add_argument(
            '--checkpoint_file_folder',
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['checkpoint_file_folder'],
            help = 'Name of folder that contain the model checkpoint file. Mandatory if load_checkpoint is True.',
            )
        
        self.parser.add_argument(
            '--checkpoint_file_name',
            type = str, 
            help = 'Name of the model checkpoint file. Mandatory if load_checkpoint is True.',
            )

        # Data Parameters

        self.parser.add_argument(
            '--n_mels', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['n_mels'], 
            help = 'Spectrograms mel bands.'
            )

        self.parser.add_argument(
            '--random_crop_secs', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['random_crop_secs'], 
            help = 'Cut the input spectrogram with random_crop_secs length at a random starting point.'
            )

        self.parser.add_argument(
            '--normalization', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['normalization'], 
            choices = ['cmn', 'cmvn'],
            help = 'Type of normalization applied to the features when evaluating in validation. \
                It can be Cepstral Mean Normalization or Cepstral Mean and Variance Normalization',
            )

        self.parser.add_argument(
            '--num_workers', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['num_workers'],
            help = 'num_workers to be used by the data loader.'
            )

        # Network Parameters

        self.parser.add_argument(
            '--model_name_prefix', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['model_name_prefix'], 
            help = 'Give the model a name prefix when saving it.'
            )

        self.parser.add_argument(
            '--embedding_size', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['embedding_size'],
            help = 'Size of the embedding that the system will generate.',
            )

        self.parser.add_argument(
            '--front_end', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['front_end'],
            choices = ['VGGNL', 'PatchsGenerator'], 
            help = 'Type of Front-end used. \
                VGGNL for a N-block VGG architecture, PatchsGenerator for Visual Transformer architecture.'
            )
            
        self.parser.add_argument(
            '--vgg_n_blocks', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['vgg_n_blocks'],
            help = 'Number of blocks the VGG front-end block will have.\
                Each block consists in two convolutional layers followed by a max pooling layer.',
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
            '--patchs_generator_patch_width', 
            type = int,
            help = 'Width of each patch token to use with at the PatchsGenerator front-end. \
                (Only used when front_end = PatchsGenerator.',
            )

        self.parser.add_argument(
            '--pooling_method', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['pooling_method'], 
            choices = ['Attention', 'MHA', 'DoubleMHA', 'SelfAttentionAttentionPooling', 'MultiHeadAttentionAttentionPooling', 'TransformerStackedAttentionPooling'], 
            help = 'Type of pooling method.',
            )

        self.parser.add_argument(
            '--pooling_output_size', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['pooling_output_size'], 
            help = 'Each output vector of the pooling component will have 1 x pooling_output_size dimension.',
            )

        self.parser.add_argument(
            '--pooling_heads_number', 
            type = int, 
            #default = TRAIN_DEFAULT_SETTINGS['pooling_heads_number'],
            help = 'Number of heads for the attention layer of the pooling component \
                (only for MHA based pooling_method options).',
            )

        self.parser.add_argument(
            '--pooling_mask_prob', 
            type = float, 
            #default = TRAIN_DEFAULT_SETTINGS['pooling_mask_prob'], 
            help = 'Masking head drop probability. Only used for pooling_method = Double MHA',
            )

        self.parser.add_argument(
            '--pooling_positional_encoding', 
            action = argparse.BooleanOptionalAction,
            #default = TRAIN_DEFAULT_SETTINGS['pooling_positional_encoding'], 
            help = 'Wether to use positional encoding in the attention layer of the pooling component.'
            )

        self.parser.add_argument(
            '--transformer_n_blocks', 
            type = int, 
            #default = TRAIN_DEFAULT_SETTINGS['transformer_n_blocks'],
            help = 'Number of transformer blocks to stack in the attention component of the pooling_method. \
                (Only for pooling_method = TransformerStackedAttentionPooling).',
            )

        self.parser.add_argument(
            '--transformer_expansion_coef', 
            type = int, 
            #default = TRAIN_DEFAULT_SETTINGS['transformer_expansion_coef'], 
            help = "Number you want to multiply by the size of the hidden layer of the transformer block's feed forward net. \
                (Only for pooling_method = TransformerStackedAttentionPooling)."
            )

        self.parser.add_argument(
            '--transformer_attention_type', 
            type = str, 
            #default = TRAIN_DEFAULT_SETTINGS['transformer_attention_type'], 
            choices = ['SelfAttention', 'MultiHeadAttention'],
            help = 'Type of Attention to use in the attention component of the transformer block.\
                (Only for pooling_method = TransformerStackedAttentionPooling).'
            )
        
        self.parser.add_argument(
            '--transformer_drop_out', 
            type = float, 
            #default = TRAIN_DEFAULT_SETTINGS['transformer_drop_out'], 
            help = 'Dropout probability to use in the feed forward component of the transformer block.\
                (Only for pooling_method = TransformerStackedAttentionPooling).'
            )

        self.parser.add_argument(
            '--bottleneck_drop_out', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['bottleneck_drop_out'], 
            help = 'Dropout probability to use in each layer of the final fully connected bottleneck.'
            )

        # AMSoftmax Config

        self.parser.add_argument(
            '--scaling_factor', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['scaling_factor'], 
            help = 'Scaling factor of the AM-Softmax (referred as s in the AM-Softmax definition).'
            )

        self.parser.add_argument(
            '--margin_factor', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['margin_factor'],
            help = 'Margin factor of the AM-Softmax (referred as m in the AM-Softmax definition).'
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
            action = argparse.BooleanOptionalAction,
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