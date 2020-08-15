import os
import sys
import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
sys.path.append('./scripts/')
from data import *
from model import SpeakerClassifier
from loss import *
from evaluate import *

def score(emb1, emb2):

    dist = F.cosine_similarity(emb1,emb2, dim=-1, eps=1e-08)
    return dist

def chkptsave(opt,model,optimizer,epoch,step):
    ''' function to save the model and optimizer parameters '''
    if torch.cuda.device_count() > 1:
        checkpoint = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'settings': opt,
            'epoch': epoch,
            'step':step}
    else:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'settings': opt,
            'epoch': epoch,
            'step':step}

    torch.save(checkpoint,'{}/{}_{}.chkpt'.format(opt.out_dir, opt.model_name,step))
    
def Accuracy(pred, labels):

    acc = 0.0
    num_pred = pred.size()[0]
    pred = torch.max(pred, 1)[1]
    for idx in range(num_pred):
        if pred[idx].item() == labels[idx].item():
            acc += 1

    return acc/num_pred

class Trainer:

    def __init__(self, params, device):

        self.params = params
        self.device = device
        self.__load_network()
        self.__load_data()
        self.__load_optimizer()
        self.__load_criterion()
        self.__initialize_training_variables()

    def __load_previous_states(self):

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

    def __initialize_training_variables(self):

        if self.params.requeue:
            self.__load_previous_states()
        else:
            self.step = 0
            self.starting_epoch = 0

        self.best_EER = 50.0
        self.stopping = 0.0


    def __load_network(self):

        self.net = SpeakerClassifier(self.params, self.device)
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = nn.DataParallel(self.net)


    def __load_data(self):
        print('Loading Data and Labels')
        with open(self.params.train_labels_path, 'r') as data_labels_file:
            train_labels=data_labels_file.readlines()

        data_loader_parameters = {'batch_size': self.params.batch_size, 'shuffle': True, 'num_workers': self.params.num_workers}
        self.training_generator = DataLoader(Dataset(train_labels, self.params), **data_loader_parameters)


    def __load_optimizer(self):
        if self.params.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        if self.params.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        if self.params.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)

    def __update_optimizer(self):

        if self.params.optimizer == 'SGD' or self.params.optimizer == 'Adam':
            for paramGroup in self.optimizer.param_groups:
                paramGroup['lr'] *= 0.5
            print('New Learning Rate: {}'.format(paramGroup['lr']))
    
    def __load_criterion(self):
        if params.loss == 'Softmax':
            self.criterion = nn.CrossEntropyLoss()
        elif params.loss=='AMSoftmax':
            self.criterion = AMSoftmax(s=self.params.scalingFactor, m=self.params.marginFactor).to(self.device)
        elif params.loss=='AMSoftmaxV2':
            self.criterion = AMSoftmax(s=self.params.scalingFactor, m=self.params.marginFactor).to(self.device)

    def __initialize_batch_variables(self):

        self.print_time = time.time()
        self.train_loss = 0.0
        self.train_accuracy = 0.0
        self.train_batch = 0

    def __extractInputFromFeature(self, sline):

        features1 = normalizeFeatures(featureReader(self.params.valid_data_dir + '/' + sline[0] + '.pickle'), normalization=self.params.normalization)
        features2 = normalizeFeatures(featureReader(self.params.valid_data_dir + '/' + sline[1] + '.pickle'), normalization=self.params.normalization)

        input1 = torch.FloatTensor(features1).to(self.device)
        input2 = torch.FloatTensor(features2).to(self.device)
        
        return input1.unsqueeze(0), input2.unsqueeze(0)

    def __extract_scores(self, trials):

        scores = []
        for line in trials:
            sline = line[:-1].split()

            input1, input2 = self.__extractInputFromFeature(sline)

            _, emb1, _ = self.net.module.getEmbedding(input1)
            _, emb2, _ = self.net.module.getEmbedding(input2)

            dist = score(emb1, emb2)
            scores.append(dist.item())

        return scores

    def __calculate_EER(self, CL, IM):

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
        return EER

    def __validate(self):

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

            print('--Validation Epoch:{epoch: d}, Updates:{Num_Batch: d}, EER:{eer: 3.3f}, elapse:{elapse: 3.3f} min'.format(epoch=self.epoch, Num_Batch=self.step, eer=EER, elapse=(time.time()-valid_time)/60))
            # early stopping and save the best model
            if EER < self.best_EER:
                self.best_EER = EER
                self.stopping = 0
                print('We found a better model!')
                chkptsave(params, self.net, self.optimizer, self.epoch, self.step)
            else:
                self.stopping += 1

            self.print_time = time.time()
            self.net.train()

    def __update(self):

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1

        if self.step % int(self.params.print_every) == 0:
            print('Training Epoch:{epoch: d}, Updates:{Num_Batch: d} -----> xent:{xnet: .3f}, Accuracy:{acc: .2f}, elapse:{elapse: 3.3f} min'.format(epoch=self.epoch, Num_Batch=self.step, xnet=self.train_loss / self.train_batch, acc=self.train_accuracy *100/ self.train_batch, elapse=(time.time()-self.print_time)/60))
            self.__initialize_batch_variables()

        # validation
        if self.step % self.params.validate_every == 0:
            self.__validate()

    def train(self):

        print('Start Training')
        for self.epoch in range(self.starting_epoch, self.params.max_epochs):  # loop over the dataset multiple times
            self.net.train()
            self.__initialize_batch_variables()
            for input, label in self.training_generator:
                self.train_batch += 1
                input, label = input.float().to(self.device), label.long().to(self.device)

                _, alignment, pred = self.net(input, label=label)
                loss = self.criterion(pred, label)
                loss.backward()
                self.train_accuracy += Accuracy(pred, label)
                self.train_loss += loss.item()

                if self.train_batch % self.params.gradientAccumulation == 0:
                    self.__update()

            if self.stopping > self.params.early_stopping:
                print('--Best Model EER%%: %.2f' %(self.best_EER))
                break
            
            if self.stopping > 10:
                self.__update_optimizer()

        print('Finished Training')

def main(opt):

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    print('Defining Device')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    print('Loading Trainer')
    trainer = Trainer(opt, device)
    trainer.train()

def getModelName(params):

    model_name = params.model_name

    model_name = model_name + '_{}'.format(params.front_end) + '_{}'.format(params.window_size) + '_{}batchSize'.format(params.batch_size*params.gradientAccumulation) + '_{}lr'.format(params.learning_rate) + '_{}weightDecay'.format(params.weight_decay) + '_{}kernel'.format(params.kernel_size) +'_{}embSize'.format(params.embedding_size) + '_{}'.format(params.loss)

    if params.loss == 'AMSoftmax':
        model_name += '_{}s'.format(params.scalingFactor) + '_{}m'.format(params.marginFactor) 
  
    model_name = model_name + '_{}'.format(params.loss) + '_{}'.format(params.pooling_method) + '_{}'.format(params.heads_number)

    return model_name

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a CNN Syamese')
   
    parser.add_argument('--train_data_dir', type=str, default='/scratch/speaker_databases/', help='data directory.')
    parser.add_argument('--valid_data_dir', type=str, default='/scratch/speaker_databases/VoxCeleb-1/wav', help='data directory.')
    parser.add_argument('--train_labels_path', type = str, default = 'labels/Vox2.ndx')
    parser.add_argument('--data_mode', type = str, default = 'normal', choices=['normal','window'])
    parser.add_argument('--valid_clients', type = str, default='labels/clients.ndx')
    parser.add_argument('--valid_impostors', type = str, default='labels/impostors.ndx')
    parser.add_argument('--out_dir', type=str, default='./models/model6', help='directory where data is saved')
    parser.add_argument('--model_name', type=str, default='CNN', help='Model associated to the model builded')
    parser.add_argument('--front_end', type=str, default='VGG4L', choices = ['VGG3L','VGG4L'], help='Kind of Front-end Used')
    
    # Network Parameteres
    parser.add_argument('--window_size', type=float, default=3.5, help='number of seconds per window')
    parser.add_argument('--normalization', type=str, default='cmn', choices=['cmn', 'cmvn'])
    parser.add_argument('--kernel_size', type=int, default=1024)
    parser.add_argument('--embedding_size', type=int, default=400)
    parser.add_argument('--heads_number', type=int, default=16)
    parser.add_argument('--pooling_method', type=str, default='Statistical', choices=['Attention', 'Statistical', 'MHA', 'DoubleMHA'], help='Type of pooling methods')
    parser.add_argument('--mask_prob', type=float, default=0.25, help='Masking Drop Probability. Only Used for Only Double MHA')
 
    # Losses 
    parser.add_argument('--loss', type=str, choices=['Softmax', 'AMSoftmax'], default='AMSoftmax', help='type of loss function')
    # AMSoftmax Config
    parser.add_argument('--scalingFactor', type=float, default=5.0, help='')
    parser.add_argument('--marginFactor', type=float, default=0.1, help='')

    # Optimization 
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD', 'RMSprop'], default='Adam')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='number of sequences to train on in parallel')
    parser.add_argument('--gradientAccumulation', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=1000000, help='number of full passes through the trainning data')
    parser.add_argument('--early_stopping', type=int, default=15, help='-1 if not early stopping')
    parser.add_argument('--print_every', type = int, default = 1000)
    parser.add_argument('--requeue',action='store_true', help='restart from the last model for requeue on slurm')
    parser.add_argument('--validate_every', type = int, default = 10000)
    parser.add_argument('--num_workers', type = int, default = 2)
    
    # parse input params
    params=parser.parse_args()
    params.model_name = getModelName(params)
    params.num_spkrs = 5994

    if not os.path.exists(params.out_dir):
        os.makedirs(params.out_dir)

    with open(params.out_dir + '/' + params.model_name + '_config.pkl', 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    main(params)
