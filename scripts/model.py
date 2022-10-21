import torch
from torch import nn
from torch.nn import functional as F
from poolings import Attention, MultiHeadAttention, DoubleMHA
from CNNs import VGGNL
from loss import AMSoftmax

class SpeakerClassifier(nn.Module):

    def __init__(self, parameters, device):
        super().__init__()
       
        parameters.feature_size = 80 # HACK hardcoded 80 for mel bands. Read the number of mel bands from the input spectrogram 
        self.device = device
        
        self.__initFrontEnd(parameters)        
        self.__initPoolingLayers(parameters)
        self.__initFullyConnectedBlock(parameters)
        
        self.predictionLayer = AMSoftmax(
            parameters.embedding_size, 
            parameters.number_speakers, 
            s = parameters.scaling_factor, 
            m = parameters.margin_factor, 
            annealing = parameters.annealing
            )
 

    def __initFrontEnd(self, parameters):

        if parameters.front_end == 'VGGNL':

            # Set the front-end component that will take the spectrogram and generate complex features
            self.front_end = VGGNL(parameters.vgg_n_blocks, parameters.vgg_channels)
                
            # Calculate the size of the hidden state vectors (output of the front-end)
            self.hidden_states_dimension = self.front_end.get_hidden_states_dimension(
                parameters.feature_size, 
                )
            
            #print(f"[model] hidden_states_dimension: {self.hidden_states_dimension}")
            

    def __initPoolingLayers(self, parameters):    

        # Set the pooling component that will take the front-end features and summarize them in a context vector

        self.pooling_method = parameters.pooling_method

        if self.pooling_method == 'Attention':
            self.poolingLayer = Attention(self.hidden_states_dimension)
        elif self.pooling_method == 'MHA':
            self.poolingLayer = MultiHeadAttention(self.hidden_states_dimension, parameters.heads_number)
        elif self.pooling_method == 'DoubleMHA':
            self.poolingLayer = DoubleMHA(self.hidden_states_dimension, parameters.heads_number, mask_prob = parameters.mask_prob)
            self.hidden_states_dimension = self.hidden_states_dimension // parameters.heads_number


    def __initFullyConnectedBlock(self, parameters):

        # Set the set of fully connected layers that will take the pooling context vector

        # TODO abstract the FC component in a class with a forward method like the other components
        # TODO Get also de RELUs in this class
        # Should we batch norm and relu the last layer?
        self.fc1 = nn.Linear(self.hidden_states_dimension, parameters.embedding_size)
        self.b1 = nn.BatchNorm1d(parameters.embedding_size)
        self.fc2 = nn.Linear(parameters.embedding_size, parameters.embedding_size)
        self.b2 = nn.BatchNorm1d(parameters.embedding_size)
        self.fc3 = nn.Linear(parameters.embedding_size, parameters.embedding_size)
        self.b3 = nn.BatchNorm1d(parameters.embedding_size)


    def forward(self, input_tensor, label = None, step = 0):

        # Mandatory torch method
        # Set the net's forward pass

        encoder_output = self.front_end(input_tensor)

        # print(f"[model] encoder_output size: {encoder_output.size()}")

        # TODO seems that alignment is not used anywhere
        embedding_0, alignment = self.poolingLayer(encoder_output)

        # TODO should we use relu and bn in every layer?

        embedding_1 = self.fc1(embedding_0)
        embedding_1 = F.relu(embedding_1)
        embedding_1 = self.b1(embedding_1)

        embedding_2 = self.fc2(embedding_1)
        embedding_2 = F.relu(embedding_2)
        embedding_2 = self.b2(embedding_2)

        embedding_3 = self.fc3(embedding_2)
        embedding_3 = self.b3(embedding_3)

        prediction, ouput_tensor = self.predictionLayer(embedding_3, label, step)
    
        return prediction, ouput_tensor


    # TODO not used in this class. where?
    def get_embedding(self, input_tensor):

        # TODO should we use relu and bn in every layer?d

        encoder_output = self.front_end(input_tensor)

        embedding_0, alignment = self.poolingLayer(encoder_output)

        embedding_1 = self.fc1(embedding_0)
        embedding_1 = F.relu(embedding_1)
        embedding_1 = self.b1(embedding_1)

        embedding_2 = self.fc2(embedding_1)
        embedding_2 = F.relu(embedding_2)
        embedding_2 = self.b2(embedding_2)
    
        return embedding_2 

