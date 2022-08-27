import os
import sys
import argparse
import numpy as np
import random
import pickle
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader

from data import Dataset, normalizeFeatures, featureReader
from model import SpeakerClassifier
from utils import getModelName, getNumberOfSpeakers, Accuracy, Score, scoreCosineDistance, chkptsave

class Trainer:

    def __init__(self, params, device):

        self.params = params
        self.device = device
        self.__load_data()
        self.__load_network()
        self.__load_criterion()
        self.__load_optimizer()
        self.__initialize_training_variables()

        self.break_epoch = 10000000
        self.break_batch = 10000000

    
    # Init methods


    # Load the Dataset
    def __load_data(self):
            
        print("Using __load_data")
        print(f'Loading Data and Labels from {self.params.train_labels_path}')
        
        with open(self.params.train_labels_path, 'r') as data_labels_file:
            train_labels = data_labels_file.readlines()

        data_loader_parameters = {
            'batch_size': self.params.batch_size, 
            'shuffle': True, 
            'num_workers': self.params.num_workers
            }
        
        self.training_generator = DataLoader(
            Dataset(train_labels, self.params), 
            **data_loader_parameters,
            )

        print("Data and labels loaded.")
        print("__load_data used")


    # Load the model (Neural Network)
    def __load_network(self):

        print("Using __load_network")
        print("Loading the network...")

        self.net = SpeakerClassifier(self.params, self.device)
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            #print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = nn.DataParallel(self.net)
        
        print("Network loaded.")
        print("__load_network used")


    # Load the loss function
    def __load_criterion(self):

        print("Using __load_criterion")
        print("Loading the loss function...")

        self.criterion = nn.CrossEntropyLoss()

        print("Loss function loaded.")
        print("__load_criterion used")


    # Load the optimizer
    def __load_optimizer(self):

        print("Using __load_optimizer")
        print("Loading the optimizer...")

        if self.params.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay
                )
        if self.params.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.net.parameters(), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay
                )
        if self.params.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(
                self.net.parameters(), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay
                )

        print(f"Optimizer {self.params.optimizer} loaded.")
        print("__load_optimizer used")


    def __load_previous_states(self):

        print("Using __load_previous_states")
        print("Loading previous states...")

        list_files = os.listdir(self.params.out_dir)
        list_files = [self.params.out_dir + '/' + f for f in list_files if '.chkpt' in f]
        if list_files:
            file2load = max(list_files, key=os.path.getctime)
            checkpoint = torch.load(file2load, map_location=self.device)
            try:
                self.net.load_state_dict(checkpoint['model'])
            except RuntimeError:
                self.net.module.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.params = checkpoint['settings']
            self.starting_epoch = checkpoint['epoch']+1
            self.step = checkpoint['step']
            print('Model "%s" is Loaded for requeue process' % file2load)
        else:
            self.step = 0
            self.starting_epoch = 1

        print("Previous states loaded.")
        print("__load_previous_states used")


    def __initialize_training_variables(self):

        print("Using __initialize_training_variables")
        print("Initializing training variables...")

        if self.params.requeue: # in the case we want to continue an unfinished training
            self.__load_previous_states()
        else:
            self.step = 0
            self.starting_epoch = 0

        self.best_EER = 50.0
        self.stopping = 0.0

        print("Training variables initialized.")
        print("__initialize_training_variables used")


    # Training methods

    # Reset some variables
    def __initialize_batch_variables(self):

        print("Using __initialize_batch_variables")
        print("Initializing batch variables...")

        self.print_time = time.time()
        self.train_loss = 0.0
        self.train_accuracy = 0.0
        self.train_batch_number = 0

        print("Batch variables initialized.")
        print("__initialize_batch_variables used")

    # TODO understand this function
    def __randomSlice(self, inputTensor):

        print("Using __randomSlice")

        index = random.randrange(200,self.params.window_size*100)
        
        print("__randomSlice used")
        return inputTensor[:,:index,:]   


    def __extractInputFromFeature(self, sline):

        #print("Using __extractInputFromFeature")

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
        
        #print("__extractInputFromFeature used")
        
        return input1.unsqueeze(0), input2.unsqueeze(0)


    def __extract_scores(self, trials):

        print("Using __extract_scores")

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

        print("__extract_scores used")
        
        return scores


    def __calculate_EER(self, CL, IM):

        print("Using __calculate_EER")

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

        print("__calculate_EER used")

        return EER
    

    def __getAnnealedFactor(self):

        print("Using __getAnnealedFactor")

        if torch.cuda.device_count() > 1:
            annealed_factor = self.net.module.predictionLayer.getAnnealedFactor(self.step)   
        else:
            annealed_factor = self.net.predictionLayer.getAnnealedFactor(self.step)    

        print("__getAnnealedFactor used")

        return annealed_factor


    def __validate(self):

        print("Using __validate")

        with torch.no_grad():
            valid_time = time.time()
            self.net.eval()
            # EER Validation
            with open(params.valid_clients,'r') as clients_in, open(params.valid_impostors,'r') as impostors_in:
                # score clients
                CL = self.__extract_scores(clients_in)
                IM = self.__extract_scores(impostors_in)
            # Compute EER
            EER = self.__calculate_EER(CL, IM)
            print(f"EER is {EER}")
            
            annealedFactor = self.__getAnnealedFactor()
            print('Annealed Factor is {}.'.format(annealedFactor))
            print('--Validation Epoch:{epoch: d}, Updates:{Num_Batch: d}, EER:{eer: 3.3f}, elapse:{elapse: 3.3f} min'. \
                format(
                    epoch=self.epoch, 
                    Num_Batch=self.step, 
                    eer=EER, 
                    elapse=(time.time()-valid_time)/60)
                    )

            # early stopping and save the best model
            if EER < self.best_EER:
                self.best_EER = EER
                self.stopping = 0
                print('We found a better model!')
                chkptsave(params, self.net, self.optimizer, self.epoch, self.step)
            else:
                self.stopping += 1
                print('Better Accuracy is: {}. {} epochs of no improvement'.format(self.best_EER, self.stopping))
            
            self.print_time = time.time()
            self.net.train()

            print("__validate used")
    

    # Updates the learning rate
    def __update_optimizer(self):

        print("Using __update_optimizer")

        if self.params.optimizer == 'SGD' or self.params.optimizer == 'Adam':
            for paramGroup in self.optimizer.param_groups:
                paramGroup['lr'] *= 0.5
            print('New Learning Rate: {}'.format(paramGroup['lr']))

        print("__update_optimizer used")


    # Updates the learning rate if some conditions are true
    def __updateTrainningVariables(self):

        print("Using __updateTrainningVariables")

        if (self.stopping+1)% 15 ==0:
            self.__update_optimizer()

        print("__updateTrainningVariables used")


    def __update(self):

        print("Using __update")

        print("Using optimizer.step")
        self.optimizer.step()
        
        print("Using optimizer.zero_grad")
        self.optimizer.zero_grad()
        
        self.step += 1

        if self.step % int(self.params.print_every) == 0:
            print(
                'Training Epoch:{epoch: d}, Updates:{Num_Batch: d} -----> \
                xent:{xnet: .3f}, Accuracy:{acc: .2f}, elapse:{elapse: 3.3f} min'. \
                format(
                    epoch = self.epoch, 
                    Num_Batch = self.step, 
                    xnet = self.train_loss / self.train_batch_number, 
                    acc = self.train_accuracy * 100 / self.train_batch_number, 
                    elapse = (time.time() - self.print_time) / 60
                    )
            )
            self.__initialize_batch_variables()

        # validation
        if self.step % self.params.validate_every == 0:
            self.__validate()
        
        print("__update used")


    def train_single_epoch(self):

        print(f"Epoch {self.epoch}...")

        self.net.train()

        self.__initialize_batch_variables()

        for batch, (input, label) in enumerate(self.training_generator):

            print(f"Batch {batch} of {len(self.training_generator)}...")

            print("Initial variables:")
            print(f"self.train_accuracy: {self.train_accuracy}")
            print(f"self.train_loss: {self.train_loss}")

            input, label = input.float().to(self.device), label.long().to(self.device)

            # TODO understand this
            input = self.__randomSlice(input) if self.params.randomSlicing else input 

            # Make predcitions on this batch
            prediction, AMPrediction  = self.net(input, label = label, step = self.step)

            # Calculate the loss
            loss = self.criterion(AMPrediction, label)
            print(f"Loss calculated: loss {loss.item()}")
            self.losses.append(loss.item())

            # Compute backpropagation
            # is optimiser.zero_grad() missing here?
            print("Using loss.backward")
            loss.backward()
            # is optimiser.step() missing here?

            # why is accuracy sum on every batch? 
            accuracy = Accuracy(prediction, label)
            print(f"Accuracy calculated: accuracy {accuracy}")
            self.train_accuracy = self.train_accuracy + accuracy
            
            # why is loss sum on every batch?
            self.train_loss = self.train_loss + loss.item()

            print("Sum updated variables:")
            print(f"self.train_accuracy: {self.train_accuracy}")
            print(f"self.train_loss: {self.train_loss}")

            self.train_batch_number = self.train_batch_number + 1
            if self.train_batch_number % self.params.gradientAccumulation == 0:
                
                self.__update()

                print("Updated variables:")
                print(f"self.train_accuracy: {self.train_accuracy}")
                print(f"self.train_loss: {self.train_loss}")

            # Early stopping check
            if self.stopping > self.params.early_stopping:
                print("Early stopping.")
                print('--Best Model EER%%: %.2f' %(self.best_EER))
                break
            
            self.__updateTrainningVariables()

            print("Final variables:")
            print(f"self.train_accuracy: {self.train_accuracy}")
            print(f"self.train_loss: {self.train_loss}")

            if self.epoch >= self.break_epoch and batch >= self.break_batch: 
                print("Exit because of break point")
                break
        
        print("-"*50)


    def train(self):

        print(f'Starting training for {self.params.max_epochs} epochs.')

        self.losses = []

        for self.epoch in range(self.starting_epoch, self.params.max_epochs):  
            
            self.train_single_epoch()

            if self.epoch == self.break_epoch: break
            
        print('Training finished!')

        print("Final model evaluation.")
        self.__validate()

        print(f"Losses: {self.losses}")










def main(opt):

    # Set the seed for experimental reproduction
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    # Set the device
    print('Defining device...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device} device.")
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs available.")
    print("Device defined.")

    print('Loading Trainer class...')
    trainer = Trainer(opt, device)
    print("Trainer class loaded.")

    trainer.train()


if __name__=="__main__":

    # TODO refactor this to a class

    parser = argparse.ArgumentParser(description='Train a VGG based Speaker Embedding Extractor')
   
    parser.add_argument('--train_data_dir', type=str, default='', help='data directory.')
    parser.add_argument('--train_labels_path', type = str, default = 'scripts/labels/train/train_labels.ndx')
    parser.add_argument('--valid_data_dir', type=str, default='', help='data directory.')
    parser.add_argument('--valid_clients', type = str, default='scripts/labels/valid/valid_clients_labels.ndx')
    parser.add_argument('--valid_impostors', type = str, default='scripts/labels/valid/valid_impostors_labels.ndx')

    parser.add_argument('--data_mode', type = str, default = 'normal', choices=['normal','window'])
    parser.add_argument('--out_dir', type=str, default='./models/model1', help='directory where data is saved')
    parser.add_argument('--model_name', type=str, default='CNN', help='Model associated to the model builded')
    parser.add_argument('--front_end', type=str, default='VGG4L', choices = ['VGG3L','VGG4L'], help='Kind of Front-end Used')
    
    # Network Parameteres
    parser.add_argument('--window_size', type=float, default=3.5, help='number of seconds per window')
    parser.add_argument('--randomSlicing',action='store_true')
    parser.add_argument('--normalization', type=str, default='cmn', choices=['cmn', 'cmvn'])
    parser.add_argument('--kernel_size', type=int, default=1024)
    parser.add_argument('--embedding_size', type=int, default=400)
    parser.add_argument('--heads_number', type=int, default=32)
    parser.add_argument('--pooling_method', type=str, default='DoubleMHA', choices=['Attention', 'MHA', 'DoubleMHA'], help='Type of pooling methods')
    parser.add_argument('--mask_prob', type=float, default=0.3, help='Masking Drop Probability. Only Used for Only Double MHA')
 
    # AMSoftmax Config
    parser.add_argument('--scalingFactor', type=float, default=30.0, help='')
    parser.add_argument('--marginFactor', type=float, default=0.4, help='')
    parser.add_argument('--annealing', action='store_true')

    # Optimization 
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD', 'RMSprop'], default='Adam')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='number of sequences to train on in parallel')
    parser.add_argument('--gradientAccumulation', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=1000000, help='number of full passes through the trainning data')
    parser.add_argument('--early_stopping', type=int, default=25, help='-1 if not early stopping')
    parser.add_argument('--print_every', type = int, default = 1000)
    parser.add_argument('--requeue',action='store_true', help='restart from the last model for requeue on slurm')
    parser.add_argument('--validate_every', type = int, default = 10000)
    parser.add_argument('--num_workers', type = int, default = 2)
    
    # parse input params
    params = parser.parse_args()
    params.model_name = getModelName(params)
    params.num_spkrs = getNumberOfSpeakers(params.train_labels_path) 
    print('{} Speaker Labels'.format(params.num_spkrs))

    # Set model's config dump paths
    if not os.path.exists(params.out_dir):
        os.makedirs(params.out_dir)
    with open(params.out_dir + '/' + params.model_name + '_config.pkl', 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    main(params)
