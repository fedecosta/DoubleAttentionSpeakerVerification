import torch
from torch import nn
from torch.nn import functional as F
from poolings import Attention, MultiHeadAttention, DoubleMHA
from poolings import SelfAttentionAttentionPooling, MultiHeadAttentionAttentionPooling, TransformerStackedAttentionPooling
from front_end import VGGNL, PatchsGenerator
from loss import AMSoftmax

class SpeakerClassifier(nn.Module):

    def __init__(self, parameters, device):
        super().__init__()
     
        self.device = device
        
        self.__initFrontEnd(parameters)        
        self.__initPoolingLayers(parameters, device = self.device)
        self.__initFullyConnectedBlock(parameters)
        
        self.am_softmax_layer = AMSoftmax(
            parameters.embedding_size, 
            parameters.number_speakers, 
            s = parameters.scaling_factor, 
            m = parameters.margin_factor, 
            )
 

    def __initFrontEnd(self, parameters):

        if parameters.front_end == 'VGGNL':

            # Set the front-end component that will take the spectrogram and generate complex features
            self.front_end = VGGNL(parameters.vgg_n_blocks, parameters.vgg_channels)
                
            # Calculate the size of the hidden state vectors (output of the front-end)
            self.hidden_states_dimension = self.front_end.get_hidden_states_dimension(
                parameters.n_mels, 
                )

        if parameters.front_end == 'PatchsGenerator':

            self.front_end = PatchsGenerator(parameters.patchs_generator_patch_width)

            self.hidden_states_dimension = int(parameters.n_mels * parameters.patchs_generator_patch_width)
  

    def __initPoolingLayers(self, parameters, device):    

        # Set the pooling component that will take the front-end features and summarize them in a context vector

        self.pooling_method = parameters.pooling_method

        # Old Pooling classes
        if self.pooling_method == 'Attention':
            self.poolingLayer = Attention(self.hidden_states_dimension)
        elif self.pooling_method == 'MHA':
            self.poolingLayer = MultiHeadAttention(self.hidden_states_dimension, parameters.pooling_heads_number)
        elif self.pooling_method == 'DoubleMHA':
            self.poolingLayer = DoubleMHA(self.hidden_states_dimension, parameters.pooling_heads_number, mask_prob = parameters.pooling_mask_prob)
            self.hidden_states_dimension = self.hidden_states_dimension // parameters.pooling_heads_number
        # New Pooling classes
        elif self.pooling_method == 'SelfAttentionAttentionPooling':
            self.poolingLayer = SelfAttentionAttentionPooling(
                emb_in = self.hidden_states_dimension,
                emb_out = parameters.pooling_output_size,
                positional_encoding = parameters.pooling_positional_encoding,
                device = device,
                )
        elif self.pooling_method == 'MultiHeadAttentionAttentionPooling':
            self.poolingLayer = MultiHeadAttentionAttentionPooling(
                emb_in = self.hidden_states_dimension,
                emb_out = parameters.pooling_output_size,
                heads = parameters.pooling_heads_number,
                positional_encoding = parameters.pooling_positional_encoding,
                device = device,
                )
        elif self.pooling_method == 'TransformerStackedAttentionPooling':
            self.poolingLayer = TransformerStackedAttentionPooling(
                emb_in = self.hidden_states_dimension,
                emb_out = parameters.pooling_output_size,
                n_blocks = parameters.transformer_n_blocks, 
                expansion_coef = parameters.transformer_expansion_coef, 
                attention_type = parameters.transformer_attention_type, 
                drop_out_p = parameters.transformer_drop_out, 
                heads = parameters.pooling_heads_number,
                positional_encoding = parameters.pooling_positional_encoding,
                device = device,
                )


    def __initFullyConnectedBlock(self, parameters):

        # Set the set of fully connected layers that will take the pooling context vector

        # TODO abstract the FC component in a class with a forward method like the other components
        # TODO Get also de RELUs in this class
        # Should we batch norm and relu the last layer?

        if self.pooling_method in ('SelfAttentionAttentionPooling', 'MultiHeadAttentionAttentionPooling', 'TransformerStackedAttentionPooling'):
            # New Pooling classes output size is different from old poolings
            self.fc1 = nn.Linear(parameters.pooling_output_size, parameters.embedding_size)
        else:
            self.fc1 = nn.Linear(self.hidden_states_dimension, parameters.embedding_size)
        self.b1 = nn.BatchNorm1d(parameters.embedding_size)
        self.fc2 = nn.Linear(parameters.embedding_size, parameters.embedding_size)
        self.b2 = nn.BatchNorm1d(parameters.embedding_size)
        self.fc3 = nn.Linear(parameters.embedding_size, parameters.embedding_size)
        self.b3 = nn.BatchNorm1d(parameters.embedding_size)

        self.drop_out = nn.Dropout(parameters.bottleneck_drop_out)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_tensor, label = None):

        # Mandatory torch method
        # Set the net's forward pass

        encoder_output = self.front_end(input_tensor)

        # TODO seems that alignment is not used anywhere
        embedding_0, alignment = self.poolingLayer(encoder_output)

        # TODO should we use relu and bn in every layer?
        embedding_0 = self.drop_out(embedding_0)
        embedding_1 = self.fc1(embedding_0)
        embedding_1 = F.relu(embedding_1)
        embedding_1 = self.b1(embedding_1)

        embedding_2 = self.drop_out(embedding_1)
        embedding_2 = self.fc2(embedding_2)
        embedding_2 = F.relu(embedding_2)
        embedding_2 = self.b2(embedding_2)

        embedding_3 = self.drop_out(embedding_2)
        embedding_3 = self.fc3(embedding_3)
        embedding_3 = self.b3(embedding_3)

        inner_products, inner_products_m_s = self.am_softmax_layer(embedding_3, label)

        probs = self.softmax(inner_products)
    
        # returning also inner_products_m_s to use them at the AM-Softmax loss calculation 
        return probs, inner_products_m_s


    # TODO not used in this class. where? in model evaluator
    def get_embedding(self, input_tensor):

        # TODO should we use relu and bn in every layer?d

        encoder_output = self.front_end(input_tensor)

        # TODO seems that alignment is not used anywhere
        embedding_0, alignment = self.poolingLayer(encoder_output)

        # TODO should we use relu and bn in every layer?

        # NO DROPOUT HERE
        embedding_1 = self.fc1(embedding_0)
        embedding_1 = F.relu(embedding_1)
        embedding_1 = self.b1(embedding_1)

        embedding_2 = self.fc2(embedding_1)
        embedding_2 = F.relu(embedding_2)
        embedding_2 = self.b2(embedding_2)
    
        return embedding_2 

