import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import math
import copy

def new_parameter(*size):
    #print("[poolings] Using new_parameter...")
    out = torch.nn.Parameter(torch.FloatTensor(*size))
    #print(f"[poolings] out size : {out.size()}")
    torch.nn.init.xavier_normal_(out)
    #print(f"[poolings] out size after xn init: {out.size()}")
    return out


class Attention(nn.Module):

    def __init__(self, embedding_size):

        super().__init__()
        self.embedding_size = embedding_size
        self.att = new_parameter(self.embedding_size, 1)

    def forward(self, ht):
        #print(f"[poolings] ht size : {ht.size()}")
        attention_score = torch.matmul(ht, self.att)
        #print(f"[poolings] attention_score size : {attention_score.size()}")
        attention_score = attention_score.squeeze()
        #print(f"[poolings] attention_score size : {attention_score.size()}")
        attention_score = F.softmax(attention_score, dim = -1)
        #print(f"[poolings] attention_score size : {attention_score.size()}")
        attention_score = attention_score.view(ht.size(0), ht.size(1), 1)
        #print(f"[poolings] attention_score size : {attention_score.size()}")
        ct = torch.sum(ht * attention_score, dim = 1)
        #print(f"[poolings] ct size : {ct.size()}")

        #print(f"[poolings] Attention output size : {ct.size()}")
        
        return ct, attention_score


class HeadAttention(nn.Module):

    def __init__(self, encoder_size, heads_number, mask_prob = 0.25, attentionSmoothing=False):

        super(HeadAttention, self).__init__()
        self.embedding_size = encoder_size//heads_number
        self.att=new_parameter(self.embedding_size,1)
        self.mask_prob = int(1/mask_prob)
        self.attentionSmoothing = attentionSmoothing

    def __maskAttention(self, attention_score, mask_value = -float('inf')):
        
        # seems that this works only with GPU (what about CPU?)
        mask = torch.cuda.FloatTensor(attention_score.size()).random_(self.mask_prob)>0
        attention_score[~mask] = mask_value
        return attention_score

    def __narrowAttention(self, new_ht):

        attention_score = torch.matmul(new_ht, self.att).squeeze()
        if self.training:
            attention_score = self.__maskAttention(attention_score)
        attention_score = F.softmax(attention_score, dim=-1).view(new_ht.size(0), new_ht.size(1),1)
        return attention_score 

    def __wideAttention(self):

        attention_score = torch.matmul(new_ht, self.att).squeeze()
        if self.training:
            attention_score = self.__maskAttention(attention_score, mask_value = -1)
        attention_score /= torch.sum(attention_score, dim=1).unsqueeze(1)
        return attention_score.view(new_ht.size(0), new_ht.size(1),1)

    def forward(self,ht):

        if self.attentionSmoothing:
            attention_score = self.__wideAttention(ht)
        else:
            attention_score = self.__narrowAttention(ht)

        weighted_ht = ht * attention_score
        ct = torch.sum(weighted_ht,dim=1)

        # print(f"[poolings] HeadAttention output size : {ct.size()}")

        return ct, attention_score


def innerKeyValueAttention(query, key, value):

    d_k = query.size(-1)
    scores = torch.diagonal(torch.matmul(key, query) / math.sqrt(d_k), dim1=-2, dim2=-1).view(value.size(0),value.size(1), value.size(2))
    p_attn = F.softmax(scores, dim = -2)
    weighted_vector = value * p_attn.unsqueeze(-1)
    ct = torch.sum(weighted_vector, dim=1)
    return ct, p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, encoder_size, heads_number):
        super(MultiHeadAttention, self).__init__()
        self.encoder_size = encoder_size
        assert self.encoder_size % heads_number == 0 # d_model
        self.head_size = self.encoder_size // heads_number 
        self.heads_number = heads_number
        self.query = new_parameter(self.head_size, self.heads_number)
        self.aligmment = None
        
    def getAlignments(self,ht): 
        batch_size = ht.size(0)
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size,-1,self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)
        return self.alignment 
    
    def getHeadsContextVectors(self,ht):    
        batch_size = ht.size(0)
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size,-1,self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)
        return headsContextVectors

    def forward(self, ht):
        headsContextVectors = self.getHeadsContextVectors(ht)

        #print(f"[poolings] MultiHeadAttention output size : {headsContextVectors.view(headsContextVectors.size(0),-1).size()}")

        return headsContextVectors.view(headsContextVectors.size(0),-1), copy.copy(self.alignment)


class DoubleMHA(nn.Module):
    def __init__(self, encoder_size, heads_number, mask_prob=0.2):
        super(DoubleMHA, self).__init__()
        self.heads_number = heads_number
        self.utteranceAttention = MultiHeadAttention(encoder_size, heads_number)
        self.heads_size = encoder_size // heads_number
        self.headsAttention = HeadAttention(encoder_size, heads_number, mask_prob=mask_prob, attentionSmoothing=False)

    def getAlignments(self, x):

        utteranceRepresentation, alignment = self.utteranceAttention(x)
        headAlignments = self.headsAttention(utteranceRepresentation.view(utteranceRepresentation.size(0), self.heads_number, self.heads_size))[1]
        return alignment, headAlignments

    def forward(self, x):
        utteranceRepresentation, alignment = self.utteranceAttention(x)
        compressedRepresentation = self.headsAttention(utteranceRepresentation.view(utteranceRepresentation.size(0), self.heads_number, self.heads_size))[0]    
        
        # print(f"[poolings] DoubleMHA output size : {compressedRepresentation.size()}")
        
        return compressedRepresentation, alignment


# ----------------------------------------------------------------------------
# New Classes
# Based on https://peterbloem.nl/blog/transformers

# TODO make dim asserts in every new class

# 1 - Attention blocks (sequence to sequence blocks, the input dimension is the same than the output dimension)

class SelfAttention(nn.Module):

    """
    Sequence to sequence component, the input dimension is the same than the output dimension.
    Self-attention without trainable parameters.
    """

    def __init__(self):

        super().__init__()


    def forward(self, x):

        raw_weights = torch.bmm(x, x.transpose(1, 2))

        weights = F.softmax(raw_weights, dim = 2)

        output = torch.bmm(weights, x)

        return output


class MultiHeadAttention(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        emb_in is the dimension of every input vector (embedding).
        heads is the number of heads to use in the Multi-Head Attention.
    """

    def __init__(self, emb_in, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # we force the same input and output dimension
        self.heads = heads

        self.init_matrix_transformations()
    

    def init_matrix_transformations(self):

        # Matrix transformations to stack every head keys, queries and values matrices
        self.to_keys = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)
        self.to_queries = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)
        self.to_values = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)

        # Linear projection. For each input vector we get self.heads heads, we project them into only one.
        self.unify_heads = nn.Linear(self.heads * self.emb_out, self.emb_out)
    
    
    def forward(self, x):

        b, t, e = x.size()
        assert e == self.emb_in, f'[MultiHeadAttention] Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        keys = self.to_keys(x).view(b, t, self.heads, self.emb_out)
        queries = self.to_queries(x).view(b, t, self.heads, self.emb_out)
        values = self.to_values(x).view(b, t, self.heads, self.emb_out)

        # 1 - Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)

        # - Instead of dividing the dot products by sqrt(e), we scale the queries and keys.
        #   This should be more memory efficient
        queries = queries / (self.emb_out ** (1/4))
        keys    = keys / (self.emb_out ** (1/4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * self.heads, t, t), f'[MultiHeadAttention] Matrix has size {dot.size()}, expected {(b * self.heads, t, t)}.'

        dot = F.softmax(dot, dim = 2) # dot now has row-wise self-attention probabilities

        # 2 - Apply the self attention to the values
        output = torch.bmm(dot, values).view(b, self.heads, t, self.emb_out)

        # swap h, t back
        output = output.transpose(1, 2).contiguous().view(b, t, self.heads * self.emb_out)

        # unify heads
        output = self.unify_heads(output)

        return output


class TransformerBlock(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        One Transformer block.
        emb_in is the dimension of every input vector (embedding).
        expansion_coef is the number you want to multiply the size of the hidden layer of the feed forward net.
        attention_type is the type of attention to use in the attention component.
        heads is the number of heads to use in the attention component, if Multi-Head Attention is used.
    """

    def __init__(self, emb_in, expansion_coef, attention_type, heads = None):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # we want the same dimension
        self.expansion_coef = expansion_coef
        self.attention_type = attention_type
        self.heads = heads

        self.init_attention_layer()
        self.init_norm_layers()
        self.init_feed_forward_layer()


    def init_attention_layer(self):

        if self.attention_type == "SelfAttention":
            self.attention_layer = SelfAttention()
        elif self.attention_type == "MultiHeadAttention":
            self.attention_layer = MultiHeadAttention(self.emb_in, self.heads)


    def init_norm_layers(self):

        self.norm1 = nn.LayerNorm(self.emb_out)
        self.norm2 = nn.LayerNorm(self.emb_out)


    def init_feed_forward_layer(self):

        self.feed_forward_layer = nn.Sequential(
            nn.Linear(self.emb_out, self.expansion_coef * self.emb_out),
            nn.ReLU(),
            nn.Linear(self.expansion_coef * self.emb_out, self.emb_out),
            )


    def forward(self, x):

        # Pass through the attention component
        attention_layer_output = self.attention_layer(x)

        # Make the skip connection
        skip_connection_1 = attention_layer_output + x

        # Normalization layer
        normalized_1 = self.norm1(skip_connection_1)

        # Feed forward component
        feed_forward = self.feed_forward_layer(normalized_1)
        
        # Make the skip connection
        skip_connection_2 = feed_forward + normalized_1

        # Normalization layer
        norm_attended_2 = self.norm2(skip_connection_2)

        # Output
        output = norm_attended_2

        return output


class TransformerStacked(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        Stack of n_blocks Transformer blocks.
        emb_in is the dimension of every input vector (embedding).
        expansion_coef is the number you want to multiply the size of the hidden layer of the feed forward net.
        attention_type is the type of attention to use in the attention component.
        heads is the number of heads to use in the attention component, if Multi-Head Attention is used.
    """
  
    def __init__(self, emb_in, n_blocks, expansion_coef, attention_type, heads = None):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # we force the same input and output dimension
        self.n_blocks = n_blocks
        self.expansion_coef = expansion_coef
        self.attention_type = attention_type
        self.heads = heads

        self.init_transformer_blocks()


    def init_transformer_block(self, emb_in, expansion_coef, attention_type, heads = None):

        # Init one transformer block

        transformer_block = TransformerBlock(emb_in, expansion_coef, attention_type, heads)

        return transformer_block


    def init_transformer_blocks(self):

        self.transformer_blocks = nn.Sequential()

        for num_block in range(self.n_blocks):

            transformer_block_name = f"transformer_block_{num_block}"
            transformer_block = self.init_transformer_block(self.emb_in, self.expansion_coef, self.attention_type, self.heads)
                
            self.transformer_blocks.add_module(transformer_block_name, transformer_block)


    def forward(self, x):

        transformer_output = self.transformer_blocks(x)

        output = transformer_output

        return output






