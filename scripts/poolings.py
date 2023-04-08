import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math

# Based on https://peterbloem.nl/blog/transformers
# TODO make dim asserts in every new class

# 1 - Attention components (sequence to sequence blocks, the input dimension is the same than the output dimension)

class SelfAttention(nn.Module):

    """
    Sequence to sequence component, the input dimension is the same than the output dimension.
    Sequence length is not fixed.
    Self-attention without trainable parameters.
    """

    def __init__(self):

        super().__init__()


    def forward(self, x):

        raw_weights = torch.bmm(x, x.transpose(1, 2))

        weights = F.softmax(raw_weights, dim = 2)

        output = torch.bmm(weights, x)

        return output


class MultiHeadAttention2(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
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
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

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

        assert dot.size() == (b * self.heads, t, t), f'Matrix has size {dot.size()}, expected {(b * self.heads, t, t)}.'

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
        Sequence length is not fixed.
        One Transformer block.
        emb_in is the dimension of every input vector (embedding).
        expansion_coef is the number you want to multiply the size of the hidden layer of the feed forward net.
        attention_type is the type of attention to use in the attention component.
        heads is the number of heads to use in the attention component, if Multi-Head Attention is used.
    """

    def __init__(self, emb_in, expansion_coef, attention_type, drop_out_p, heads = None):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # we want the same dimension
        self.expansion_coef = expansion_coef
        self.attention_type = attention_type
        self.drop_out_p = drop_out_p
        self.heads = heads
        

        self.init_attention_layer()
        self.init_norm_layers()
        self.init_feed_forward_layer()
        self.drop_out = nn.Dropout(drop_out_p)


    def init_attention_layer(self):

        if self.attention_type == "SelfAttention":
            self.attention_layer = SelfAttention()
        elif self.attention_type == "MultiHeadAttention":
            self.attention_layer = MultiHeadAttention2(self.emb_in, self.heads)


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

        b, t, e = x.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        # Pass through the attention component
        attention_layer_output = self.attention_layer(x)

        # Make the skip connection
        skip_connection_1 = attention_layer_output + x

        # Normalization layer
        normalized_1 = self.norm1(skip_connection_1)

        # Feed forward component
        feed_forward = self.feed_forward_layer(self.drop_out(normalized_1))
        
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
        Sequence length is not fixed.
        Stack of n_blocks Transformer blocks.
        emb_in is the dimension of every input vector (embedding).
        expansion_coef is the number you want to multiply the size of the hidden layer of the feed forward net.
        attention_type is the type of attention to use in the attention component.
        heads is the number of heads to use in the attention component, if Multi-Head Attention is used.
    """
  
    def __init__(self, emb_in, n_blocks, expansion_coef, attention_type, drop_out_p, heads = None):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # we force the same input and output dimension
        self.n_blocks = n_blocks
        self.expansion_coef = expansion_coef
        self.attention_type = attention_type
        self.drop_out_p = drop_out_p
        self.heads = heads

        self.init_transformer_blocks()


    def init_transformer_block(self, emb_in, expansion_coef, attention_type, drop_out_p, heads = None):

        # Init one transformer block

        transformer_block = TransformerBlock(emb_in, expansion_coef, attention_type, drop_out_p, heads)

        return transformer_block


    def init_transformer_blocks(self):

        self.transformer_blocks = nn.Sequential()

        for num_block in range(self.n_blocks):

            transformer_block_name = f"transformer_block_{num_block}"
            transformer_block = self.init_transformer_block(self.emb_in, self.expansion_coef, self.attention_type, self.drop_out_p, self.heads)
                
            self.transformer_blocks.add_module(transformer_block_name, transformer_block)


    def forward(self, x):

        b, t, e = x.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        transformer_output = self.transformer_blocks(x)

        output = transformer_output

        return output


# 2 - Pooling components (sequence to one components, the input dimension is the same than the output dimension)

class StatisticalPooling(nn.Module):

    """
        Sequence to one component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Given n vectors, takes their average as output.
        emb_in is the dimension of every input vector (embedding).
    """

    def __init__(self, emb_in):

        super().__init__()
        
        self.emb_in = emb_in 


    def forward(self, x):

        b, t, e = x.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        # Get the average of the input vectors (dim = 0 is the batch dimension)
        output = x.mean(dim = 1)

        return output


class AttentionPooling(nn.Module):

    """
        Sequence to one component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Given n vectors, takes their weighted average as output. These weights comes from an attention mechanism.
        It can be seen as a One Head Self-Attention, where a unique query is used and input vectors are the values and keys.   
        emb_in is the dimension of every input vector (embedding).
    """

    def __init__(self, emb_in):

        super().__init__()

        self.emb_in = emb_in
        self.init_query()

        
    def init_query(self):

        # Init the unique trainable query.
        self.query = torch.nn.Parameter(torch.FloatTensor(self.emb_in, 1))
        torch.nn.init.xavier_normal_(self.query)


    def forward(self, x):

        b, t, e = x.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        attention_scores = torch.matmul(x, self.query)
        attention_scores = attention_scores.squeeze(dim = -1)
        attention_scores = F.softmax(attention_scores, dim = 1)
        attention_scores = attention_scores.unsqueeze(dim = -1)

        output = torch.bmm(attention_scores.transpose(1, 2), x)
        output = output.view(output.size()[0], output.size()[1] * output.size()[2])
        
        return output


# 3 - Pooling Systems (sequence to one components, the input dimension can be different than the output dimension)
# Consists of an Attention component followed by a Pooling component 

# HACK The following classes are constructed in this way to keep the old classes running. 
# Must refactor all the module eventually to allow choosing an Attention and Pooling component as a argparse input param.

class SelfAttentionAttentionPooling(nn.Module):

    """
        Sequence to one component, the input dimension can be different than the output dimension.
        Sequence length is not fixed.
        Consists of an SelfAttention component followed by a AttentionPooling component.
        emb_in is the dimension of every input vector (embedding).
        emb_out is the dimension of the final output vector (embedding).
    """

    def __init__(self, emb_in, emb_out, positional_encoding, device):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_out
        self.positional_encoding = positional_encoding
        self.device = device
        self.init_linear_projection()
        self.init_positional_encoding()
        self.init_attention_layer()
        self.init_pooling_layer()

    
    def init_linear_projection(self):
    
      self.projection = nn.Linear(self.emb_in, self.emb_out, bias=False)


    def init_positional_encoding(self):

        if self.positional_encoding:
            self.positional_encoding_layer = PositionalEncoding(
                d_model = self.emb_out, 
                device = self.device,
                p_drop_out = 0.0, 
                max_len = 5000,
                )


    def init_attention_layer(self):
      
      self.attention_layer = SelfAttention()


    def init_pooling_layer(self):

      self.pooling_layer = AttentionPooling(self.emb_out)


    def forward(self, x):

        b, t, e = x.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'
      
        x = self.projection(x)

        if self.positional_encoding:
            x = self.positional_encoding_layer(x)

        output = self.attention_layer(x)

        output = self.pooling_layer(output)

        assert output.size()[1] == self.emb_out, f'Output embedding dim ({output.size()[1]}) should match layer embedding dim ({self.emb_out})'
        
        # Returing a tuple to keep old classes running
        return output, None


class MultiHeadAttentionAttentionPooling(nn.Module):

    """
        Sequence to one component, the input dimension can be different than the output dimension.
        Sequence length is not fixed.
        Consists of an MultiHeadAttention component followed by a AttentionPooling component.
        emb_in is the dimension of every input vector (embedding).
        emb_out is the dimension of the final output vector (embedding).
        heads is the number of heads to use in the Multi-Head Attention component.
    """

    def __init__(self, emb_in, emb_out, heads, positional_encoding, device):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_out
        self.heads = heads
        self.positional_encoding = positional_encoding
        self.device = device

        self.init_linear_projection()
        self.init_positional_encoding()
        self.init_attention_layer()
        self.init_pooling_layer()

    
    def init_linear_projection(self):
    
      self.projection = nn.Linear(self.emb_in, self.emb_out, bias=False)


    def init_positional_encoding(self):

        if self.positional_encoding:
            self.positional_encoding_layer = PositionalEncoding(
                d_model = self.emb_out, 
                device = self.device,
                p_drop_out = 0.0, 
                max_len = 5000,
                )


    def init_attention_layer(self):
      
      self.attention_layer = MultiHeadAttention2(self.emb_out, self.heads)


    def init_pooling_layer(self):

      self.pooling_layer = AttentionPooling(self.emb_out)


    def forward(self, x):

        b, t, e = x.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        x = self.projection(x)

        if self.positional_encoding:
            x = self.positional_encoding_layer(x)

        output = self.attention_layer(x)

        output = self.pooling_layer(output)

        assert output.size()[1] == self.emb_out, f'Output embedding dim ({output.size()[1]}) should match layer embedding dim ({self.emb_out})'

        # Returing a tuple to keep old classes running
        return output, None


class TransformerStackedAttentionPooling(nn.Module):

    """
        Sequence to one component, the input dimension can be different than the output dimension.
        Sequence length is not fixed.
        Consists of an TransformerStacked component followed by a AttentionPooling component.
        emb_in is the dimension of every input vector (embedding).
        emb_out is the dimension of the final output vector (embedding).
        n_blocks is the number of Transformer blocks to use.
        expansion_coef is the number you want to multiply the size of the hidden layer of the feed forward net.
        attention_type is the type of attention to use in the attention component.
        heads is the number of heads to use in the attention component, if Multi-Head Attention is used.
    """

    def __init__(self, emb_in, emb_out, n_blocks, expansion_coef, attention_type, drop_out_p, heads, positional_encoding, device):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_out
        self.n_blocks = n_blocks
        self.expansion_coef = expansion_coef
        self.attention_type = attention_type
        self.drop_out_p = drop_out_p
        self.heads = heads
        self.positional_encoding = positional_encoding
        self.device = device
        
        self.init_linear_projection()
        self.init_positional_encoding()
        self.init_attention_layer()
        self.init_pooling_layer()

    
    def init_linear_projection(self):
    
      self.projection = nn.Linear(self.emb_in, self.emb_out, bias=False)


    def init_positional_encoding(self):

        if self.positional_encoding:
            self.positional_encoding_layer = PositionalEncoding(
                d_model = self.emb_out, 
                device = self.device,
                p_drop_out = 0.0, 
                max_len = 5000,
                )


    def init_attention_layer(self):
      
      self.attention_layer = TransformerStacked(self.emb_out, self.n_blocks, self.expansion_coef, self.attention_type, self.drop_out_p, self.heads)


    def init_pooling_layer(self):

      self.pooling_layer = AttentionPooling(self.emb_out)


    def forward(self, x):
      
        b, t, e = x.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        x = self.projection(x)

        if self.positional_encoding:
            x = self.positional_encoding_layer(x)

        output = self.attention_layer(x)

        output = self.pooling_layer(output)

        assert output.size()[1] == self.emb_out, f'Output embedding dim ({output.size()[1]}) should match layer embedding dim ({self.emb_out})'

        # Returing a tuple to keep old classes running
        return output, None


class PositionalEncoding(nn.Module):
    "Implement the PE function from http://nlp.seas.harvard.edu/annotated-transformer/#positional-encoding."

    def __init__(self, d_model, device, p_drop_out = 0.0, max_len = 5000):
        
        super().__init__()

        self.d_model = d_model
        self.device = device
        self.p_drop_out = p_drop_out
        self.max_len = max_len
        self.dropout = nn.Dropout(p = p_drop_out)
        self.compute_pe_matrix(max_len = self.max_len, d_model = self.d_model)

    def compute_pe_matrix(self, max_len, d_model):

        # Compute the positional encodings once in log space.

        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

        # adds self.pe to the state_dict so that its included when serialized to disk
        # self.register_buffer("positional_encoding", self.pe)

    def forward(self, x):

        # if the input has more positions than previously calculated in pe we enlarge pe
        if x.size(1) > self.pe.size(0):
            self.compute_pe_matrix(max_len = x.size(1) + 100, d_model = self.d_model)

        x = x + self.pe[:, : x.size(1)].requires_grad_(False).to(device = self.device)
        
        return self.dropout(x)
